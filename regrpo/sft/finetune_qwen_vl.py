# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""RoT-aware Qwen-VL LoRA SFT.

This training entrypoint requires the MAT-Agent Qwen-VL environment
(`transformers==4.50.2`, Qwen-VL dependencies, DeepSpeed/PEFT as needed). It is
not run in CI. Heavy training dependencies are imported lazily so the module can
be imported in the repo's lightweight test environment.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any

from regrpo.common.io import read_json

IGNORE_TOKEN_ID = -100
_IMG_TAG_RE = re.compile(r"(?:Picture\s+\d+:\s*)?<img>.*?</img>\s*", re.DOTALL)
DEFAULT_IMAGE_MIN_PIXELS = 4 * 28 * 28
DEFAULT_IMAGE_MAX_PIXELS = 768 * 28 * 28


def preprocess(
    records: list[dict],
    tokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
) -> dict:
    """Tokenize Qwen-VL records with clean/RoT-aware assistant loss masks."""

    import torch

    roles = {
        "user": "<|im_start|>user",
        "assistant": "<|im_start|>assistant",
        "system": "<|im_start|>system",
    }
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    input_ids, targets = [], []
    for record in records:
        source = list(record["conversations"])
        train_indices = _trainable_qwen_vl_indices(record)
        indexed_source = list(enumerate(source))
        if indexed_source and indexed_source[0][1].get("from") != "user":
            indexed_source = indexed_source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [IGNORE_TOKEN_ID] * len(system)
        for original_index, sentence in indexed_source:
            role_name = sentence["from"]
            role = roles[role_name]
            role_ids = tokenizer(role).input_ids
            content_ids = tokenizer(sentence["value"]).input_ids
            _input_id = role_ids + nl_tokens + content_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role_name in {"user", "system"}:
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
            elif role_name == "assistant":
                if original_index in train_indices:
                    _target = (
                        [IGNORE_TOKEN_ID] * (len(role_ids) + len(nl_tokens))
                        + content_ids
                        + [IGNORE_TOKEN_ID] * (1 + len(nl_tokens))
                    )
                else:
                    _target = [IGNORE_TOKEN_ID] * len(_input_id)
            else:
                raise NotImplementedError(f"unsupported role: {sentence.get('from')!r}")
            target += _target
        if len(input_id) != len(target):
            raise AssertionError("input/target length mismatch")
        input_id += [tokenizer.pad_token_id] * max(0, max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * max(0, max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.int)
    targets_tensor = torch.tensor(targets, dtype=torch.int)
    return {
        "input_ids": input_ids_tensor,
        "labels": targets_tensor,
        "attention_mask": input_ids_tensor.ne(tokenizer.pad_token_id),
    }


def preprocess_multimodal_record(
    record: dict,
    processor,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
    image_min_pixels: int | None = DEFAULT_IMAGE_MIN_PIXELS,
    image_max_pixels: int | None = DEFAULT_IMAGE_MAX_PIXELS,
) -> dict:
    """Preprocess one Qwen2-VL SFT record with optional real image tensors."""

    import torch

    tokenizer = _processor_tokenizer(processor)
    messages, indexed_source = _qwen2vl_sft_messages(
        record,
        system_message=system_message,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_paths = _record_image_paths_from_messages(messages)
    if image_paths:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs, video_inputs = None, None
    encoded = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask[0]
    labels = torch.full_like(input_ids, IGNORE_TOKEN_ID)
    _mask_trainable_assistant_spans(record, indexed_source, tokenizer, input_ids, labels)

    if input_ids.shape[0] > max_len:
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]

    item = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
    for key in ("pixel_values", "image_grid_thw"):
        if key in encoded:
            item[key] = encoded[key]
    return item


def train(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    imports = _lazy_training_imports()
    torch = imports["torch"]
    transformers = imports["transformers"]
    Trainer = imports["Trainer"]
    GPTQConfig = imports["GPTQConfig"]
    LoraConfig = imports["LoraConfig"]
    get_peft_model = imports["get_peft_model"]
    prepare_model_for_kbit_training = imports["prepare_model_for_kbit_training"]
    deepspeed = imports["deepspeed"]
    zero = imports["zero"]
    ZeroParamStatus = imports["ZeroParamStatus"]
    DistributedType = imports["DistributedType"]

    if args.deepspeed and args.q_lora:
        logging.warning("QLoRA with DeepSpeed follows MAT-Agent compatibility behavior.")

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        max_steps=args.steps,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if args.gradient_checkpointing
        else None,
        deepspeed=args.deepspeed,
        report_to=args.report_to,
        remove_unused_columns=False,
    )
    training_args.use_lora = args.use_lora
    training_args.fix_vit = args.fix_vit
    if args.deepspeed and args.q_lora:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    _ = compute_dtype
    device_map = None
    if args.q_lora:
        import os

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if world_size != 1 else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are not compatible with QLoRA.")

    config = transformers.AutoConfig.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
        cache_dir=args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(bits=4, disable_exllama=True)
        if args.use_lora and args.q_lora
        else None,
    )
    if not args.use_lora and args.fix_vit and hasattr(model, "transformer") and hasattr(
        model.transformer, "visual"
    ):
        model.transformer.visual.requires_grad_(False)
        if hasattr(model.transformer.visual, "attn_pool"):
            model.transformer.visual.attn_pool.requires_grad_(True)

    if args.use_images:
        processor_kwargs = {}
        if args.image_min_pixels is not None:
            processor_kwargs["min_pixels"] = args.image_min_pixels
        if args.image_max_pixels is not None:
            processor_kwargs["max_pixels"] = args.image_max_pixels
        processor = transformers.AutoProcessor.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
            **processor_kwargs,
        )
        tokenizer = _processor_tokenizer(processor)
        tokenizer.model_max_length = args.max_len
        tokenizer.padding_side = "right"
    else:
        processor = None
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            model_max_length=args.max_len,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
    if hasattr(tokenizer, "eod_id") and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.im_start_id = tokenizer.encode("<|im_start|>")[0]
    tokenizer.im_end_id = tokenizer.encode("<|im_end|>")[0]

    if args.use_lora:
        if args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing
            )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=_qwen2vl_lora_targets(
                args.lora_layers,
                include_mlp=args.lora_target_mlp,
            ),
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=_split_csv_arg(args.modules_to_save),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        if args.gradient_checkpointing:
            model.enable_input_require_grads()

    raw_data = read_json(args.data)
    if args.use_images:
        train_dataset = _multimodal_dataset_cls(imports["Dataset"])(
            raw_data,
            processor,
            args.max_len,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        )
        data_collator = Qwen2VLSFTDataCollator(tokenizer)
    else:
        train_dataset = _dataset_cls(imports["Dataset"])(raw_data, tokenizer, args.max_len)
        data_collator = None
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    _safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=args.output_dir,
        bias=args.lora_bias,
        deepspeed=deepspeed,
        zero=zero,
        zero_param_status=ZeroParamStatus,
    )


def main(argv: list[str] | None = None) -> None:
    train(argv)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Converted Qwen-VL JSON from convert_qwen_vl")
    parser.add_argument("--model", required=True, help="Qwen2-VL/Qwen-VL model path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fix-vit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", default="none")
    parser.add_argument("--lora-layers", type=int, default=28)
    parser.add_argument(
        "--lora-target-mlp",
        action="store_true",
        help="Also target text MLP projections (gate_proj/up_proj/down_proj).",
    )
    parser.add_argument(
        "--modules-to-save",
        default=None,
        help="Comma-separated non-LoRA modules to save, e.g. 'wte,lm_head'.",
    )
    parser.add_argument("--q-lora", action="store_true")
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="Use Qwen2-VL processor/process_vision_info to feed real image pixels.",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=DEFAULT_IMAGE_MIN_PIXELS,
        help="Qwen2-VL per-image min_pixels override for --use-images.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=DEFAULT_IMAGE_MAX_PIXELS,
        help="Qwen2-VL per-image max_pixels override for --use-images.",
    )
    return parser.parse_args(argv)


def _trainable_qwen_vl_indices(record: dict) -> set[int]:
    policy = record.get("mask_policy", "clean")
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("record must contain conversations")
    if policy == "clean":
        return {idx for idx, turn in enumerate(conversations) if turn.get("from") == "assistant"}
    if policy == "rot":
        target = record.get("train_turn_index")
        if not isinstance(target, int):
            raise ValueError("RoT Qwen-VL record must contain integer train_turn_index")
        if target < 0 or target >= len(conversations):
            raise ValueError("train_turn_index out of range")
        turn = conversations[target]
        if turn.get("from") != "assistant" or not str(turn.get("value", "")).startswith("Reflection:"):
            raise ValueError("train_turn_index must point to the Reflection assistant turn")
        return {target}
    raise ValueError(f"unknown mask_policy: {policy!r}")


def _lazy_training_imports() -> dict[str, Any]:
    import torch
    import transformers
    from accelerate.utils import DistributedType
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import GPTQConfig, Trainer
    from transformers.integrations import deepspeed

    return {
        "torch": torch,
        "transformers": transformers,
        "DistributedType": DistributedType,
        "zero": zero,
        "ZeroParamStatus": ZeroParamStatus,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "Dataset": Dataset,
        "GPTQConfig": GPTQConfig,
        "Trainer": Trainer,
        "deepspeed": deepspeed,
    }


def _dataset_cls(dataset_base):
    class QwenVLSupervisedDataset(dataset_base):
        def __init__(self, raw_data: list[dict], tokenizer, max_len: int):
            super().__init__()
            self.data = preprocess(raw_data, tokenizer, max_len)

        def __len__(self) -> int:
            return len(self.data["input_ids"])

        def __getitem__(self, index: int) -> dict:
            return {
                "input_ids": self.data["input_ids"][index],
                "labels": self.data["labels"][index],
                "attention_mask": self.data["attention_mask"][index],
            }

    return QwenVLSupervisedDataset


def _multimodal_dataset_cls(dataset_base):
    class Qwen2VLSupervisedDataset(dataset_base):
        def __init__(
            self,
            raw_data: list[dict],
            processor,
            max_len: int,
            image_min_pixels: int | None = DEFAULT_IMAGE_MIN_PIXELS,
            image_max_pixels: int | None = DEFAULT_IMAGE_MAX_PIXELS,
        ):
            super().__init__()
            self.raw_data = raw_data
            self.processor = processor
            self.max_len = max_len
            self.image_min_pixels = image_min_pixels
            self.image_max_pixels = image_max_pixels

        def __len__(self) -> int:
            return len(self.raw_data)

        def __getitem__(self, index: int) -> dict:
            return preprocess_multimodal_record(
                self.raw_data[index],
                self.processor,
                self.max_len,
                image_min_pixels=self.image_min_pixels,
                image_max_pixels=self.image_max_pixels,
            )

    return Qwen2VLSupervisedDataset


class Qwen2VLSFTDataCollator:
    """Pad text tensors and concatenate Qwen2-VL image tensors across examples."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        import torch

        input_ids = [item["input_ids"].long() for item in features]
        labels = [item["labels"].long() for item in features]
        attention_mask = [item["attention_mask"].long() for item in features]
        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_TOKEN_ID,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
        }
        pixel_values = [item["pixel_values"] for item in features if item.get("pixel_values") is not None]
        image_grid_thw = [
            item["image_grid_thw"] for item in features if item.get("image_grid_thw") is not None
        ]
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grid_thw:
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        return batch


def _qwen2vl_sft_messages(
    record: dict,
    system_message: str = "You are a helpful assistant.",
    image_min_pixels: int | None = DEFAULT_IMAGE_MIN_PIXELS,
    image_max_pixels: int | None = DEFAULT_IMAGE_MAX_PIXELS,
) -> tuple[list[dict], list[tuple[int, dict]]]:
    source = list(record["conversations"])
    indexed_source = list(enumerate(source))
    if indexed_source and indexed_source[0][1].get("from") != "user":
        indexed_source = indexed_source[1:]
    image_paths = _usable_image_paths(_image_paths_from_value(record.get("image")))
    messages = [{"role": "system", "content": system_message}]
    inserted_images = False
    for _, sentence in indexed_source:
        role = str(sentence["from"])
        value = str(sentence.get("value", ""))
        if role == "user" and image_paths and not inserted_images:
            content = [
                _qwen2vl_image_message(
                    path,
                    min_pixels=image_min_pixels,
                    max_pixels=image_max_pixels,
                )
                for path in image_paths
            ]
            content.append({"type": "text", "text": _strip_inline_img_tags(value)})
            inserted_images = True
        else:
            content = value
        messages.append({"role": role, "content": content})
    return messages, indexed_source


def _qwen2vl_image_message(
    path: str,
    *,
    min_pixels: int | None = DEFAULT_IMAGE_MIN_PIXELS,
    max_pixels: int | None = DEFAULT_IMAGE_MAX_PIXELS,
) -> dict:
    item = {"type": "image", "image": path}
    if min_pixels is not None:
        item["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        item["max_pixels"] = int(max_pixels)
    return item


def _mask_trainable_assistant_spans(
    record: dict,
    indexed_source: list[tuple[int, dict]],
    tokenizer,
    input_ids,
    labels,
) -> None:
    train_indices = _trainable_qwen_vl_indices(record)
    input_list = [int(token) for token in input_ids.tolist()]
    cursor = 0
    for original_index, sentence in indexed_source:
        if sentence.get("from") != "assistant":
            continue
        content_ids = tokenizer(str(sentence.get("value", "")), add_special_tokens=False).input_ids
        if not content_ids:
            continue
        start = _find_subsequence(input_list, content_ids, cursor)
        if start < 0:
            raise ValueError(
                "could not align assistant content tokens in multimodal Qwen-VL input "
                f"for record {record.get('id', '')!r}, turn {original_index}"
            )
        end = start + len(content_ids)
        if original_index in train_indices:
            labels[start:end] = input_ids[start:end]
        cursor = end


def _find_subsequence(values: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return start
    last = len(values) - len(needle)
    for index in range(max(start, 0), last + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def _strip_inline_img_tags(text: str) -> str:
    return _IMG_TAG_RE.sub("", text).strip()


def _record_image_paths_from_messages(messages: list[dict]) -> list[str]:
    paths = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image" and item.get("image"):
                paths.append(str(item["image"]))
    return _usable_image_paths(paths)


def _image_paths_from_value(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(item) for item in value.values() if item]
    if isinstance(value, (list, tuple)):
        paths = []
        for item in value:
            paths.extend(_image_paths_from_value(item))
        return paths
    return []


def _usable_image_paths(paths: list[str]) -> list[str]:
    usable = []
    for path in paths:
        if _is_usable_image_source(path):
            usable.append(path)
    return usable


def _is_usable_image_source(path: str) -> bool:
    if path.startswith(("http://", "https://", "data:")):
        return True
    if os.path.exists(path):
        return True
    warnings.warn(
        "skipping missing local image file in Qwen-VL SFT training: " f"{path}",
        RuntimeWarning,
        stacklevel=3,
    )
    return False


def _processor_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def _qwen2vl_lora_targets(layer_count: int, include_mlp: bool = False) -> list[str]:
    targets = []
    for idx in range(layer_count):
        targets.extend(
            [
                f"model.layers.{idx}.self_attn.q_proj",
                f"model.layers.{idx}.self_attn.k_proj",
                f"model.layers.{idx}.self_attn.v_proj",
                f"model.layers.{idx}.self_attn.o_proj",
            ]
        )
        if include_mlp:
            targets.extend(
                [
                    f"model.layers.{idx}.mlp.gate_proj",
                    f"model.layers.{idx}.mlp.up_proj",
                    f"model.layers.{idx}.mlp.down_proj",
                ]
            )
    return targets


def _split_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _safe_save_model_for_hf_trainer(
    trainer,
    output_dir: str | Path,
    bias: str,
    deepspeed,
    zero,
    zero_param_status,
) -> None:
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    elif trainer.args.use_lora:
        state_dict = _get_peft_state_maybe_zero_3(
            trainer.model.named_parameters(), bias, zero, zero_param_status
        )
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(str(output_dir), state_dict=state_dict)


def _get_peft_state_maybe_zero_3(named_params, bias: str, zero, zero_param_status) -> dict:
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                lora_bias_names.add(k.split("lora_")[0] + "bias")
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    return {
        key: _maybe_zero_3(value, zero=zero, zero_param_status=zero_param_status)
        for key, value in to_return.items()
    }


def _maybe_zero_3(param, zero, zero_param_status):
    if hasattr(param, "ds_id"):
        if param.ds_status != zero_param_status.NOT_AVAILABLE:
            raise AssertionError("unexpected DeepSpeed parameter state")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


if __name__ == "__main__":
    main()
