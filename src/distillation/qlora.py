"""QLoRA configuration and utilities for memory-efficient fine-tuning."""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning."""

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # nf4 or fp4
    bnb_4bit_compute_dtype: str = "bfloat16"  # bfloat16, float16, float32
    bnb_4bit_use_double_quant: bool = True  # Nested quantization

    # LoRA
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Alpha scaling
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None  # Auto-detect if None

    # Training
    gradient_checkpointing: bool = True

    def get_bnb_config(self) -> BitsAndBytesConfig:
        """Get BitsAndBytes quantization config."""
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)

        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )

    def get_lora_config(self, model_type: str = "t5") -> LoraConfig:
        """Get LoRA config for the specified model type."""
        # Default target modules for T5
        if self.target_modules is None:
            if "t5" in model_type.lower():
                target_modules = ["q", "v", "k", "o", "wi", "wo"]
            else:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        else:
            target_modules = self.target_modules

        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )


def setup_qlora_model(
    model,
    qlora_config: QLoRAConfig,
    model_type: str = "t5",
) -> tuple:
    """
    Prepare model for QLoRA training.

    Args:
        model: The base model (already loaded with quantization)
        qlora_config: QLoRA configuration
        model_type: Model type for target module detection

    Returns:
        Tuple of (peft_model, lora_config)
    """
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=qlora_config.gradient_checkpointing,
    )

    # Get LoRA config
    lora_config = qlora_config.get_lora_config(model_type)

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params, all_params = get_trainable_params(model)
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    print(f"Total params: {all_params:,}")

    return model, lora_config


def get_trainable_params(model) -> tuple[int, int]:
    """Get count of trainable and total parameters."""
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return trainable_params, all_params


def load_model_for_qlora(
    model_name: str,
    qlora_config: QLoRAConfig,
    device_map: str = "auto",
):
    """
    Load a model with quantization for QLoRA training.

    Args:
        model_name: HuggingFace model name
        qlora_config: QLoRA configuration
        device_map: Device mapping strategy

    Returns:
        Quantized model ready for QLoRA setup
    """
    from transformers import T5ForConditionalGeneration, AutoTokenizer

    # Get quantization config
    bnb_config = qlora_config.get_bnb_config()

    # Load model with quantization
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def merge_and_save_qlora(
    model,
    output_path: str,
    tokenizer=None,
):
    """
    Merge LoRA weights into base model and save.

    Args:
        model: PEFT model with LoRA adapters
        output_path: Path to save merged model
        tokenizer: Optional tokenizer to save
    """
    from pathlib import Path

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Merge LoRA weights
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(output_path)

    if tokenizer:
        tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")


def save_qlora_adapters(
    model,
    output_path: str,
):
    """
    Save only the LoRA adapters (much smaller than full model).

    Args:
        model: PEFT model with LoRA adapters
        output_path: Path to save adapters
    """
    from pathlib import Path

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    print(f"LoRA adapters saved to {output_path}")


def load_qlora_for_inference(
    base_model_name: str,
    adapter_path: str,
    device: str = "auto",
    merge: bool = True,
):
    """
    Load a QLoRA model for inference.

    Args:
        base_model_name: Original base model name
        adapter_path: Path to saved LoRA adapters
        device: Device to load on
        merge: Whether to merge adapters (faster inference)

    Returns:
        Model ready for inference
    """
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    from peft import PeftModel

    # Load base model (can load in fp16 for inference)
    model = T5ForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)

    if merge:
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer
