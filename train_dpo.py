"""
train_dpo.py -- DPO fine-tuning using sparse-personality generated pairs.

Trains models using QLoRA to fit in limited VRAM.
Uses DPO pairs from sparse-personality baselines to fix behavioral flaws.

Usage:
    python train_dpo.py qwen25_7b
    python train_dpo.py --model Qwen/Qwen2.5-7B-Instruct --dpo-pairs baselines/qwen25_7b_dpo.jsonl
    python train_dpo.py olmo2_7b --epochs 5 --duplication 4
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

TRAINABLE_MODELS = {
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
    "olmo3_7b": "allenai/OLMo-3-7B-Instruct-20251121",
    "gemma3_4b": "google/gemma-3-4b-it",
}


def load_dpo_pairs(jsonl_path):
    """Load DPO pairs from sparse-personality export."""
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pairs.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            })
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(description="DPO fine-tuning with sparse-personality pairs")
    parser.add_argument("name", nargs="?", help="Short model name from TRAINABLE_MODELS")
    parser.add_argument("--model", help="HuggingFace model ID (overrides name lookup)")
    parser.add_argument("--dpo-pairs", help="Path to DPO JSONL file (default: baselines/{name}_dpo.jsonl)")
    parser.add_argument("--output-dir", help="Output directory (default: trained/{name}-personality)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--duplication", type=int, default=8, help="Data duplication factor (default: 8)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).parent

    # Resolve model name and ID
    if args.name and args.name in TRAINABLE_MODELS:
        model_name = TRAINABLE_MODELS[args.name]
        short_name = args.name
    elif args.model:
        model_name = args.model
        short_name = model_name.split("/")[-1].lower().replace("-", "_")
    elif args.name:
        print(f"Unknown model '{args.name}'. Available: {', '.join(TRAINABLE_MODELS.keys())}")
        sys.exit(1)
    else:
        print("Provide a model name or --model. Available:")
        for k, v in TRAINABLE_MODELS.items():
            print(f"  {k}: {v}")
        sys.exit(1)

    if args.model:
        model_name = args.model

    dpo_path = Path(args.dpo_pairs) if args.dpo_pairs else base_dir / "baselines" / f"{short_name}_dpo.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "trained" / f"{short_name}-personality"

    if not dpo_path.exists():
        print(f"DPO pairs not found: {dpo_path}")
        sys.exit(1)

    pairs = load_dpo_pairs(dpo_path)
    print(f"Loaded {len(pairs)} DPO pairs from {dpo_path}")

    if len(pairs) < 5:
        print("Not enough pairs for meaningful training. Need at least 5.")
        sys.exit(1)

    pairs = pairs * args.duplication
    print(f"Expanded to {len(pairs)} training examples ({args.duplication}x duplication)")

    dataset = Dataset.from_list(pairs)

    print(f"\nLoading {model_name} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        beta=0.1,
        max_length=512,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )

    print("\nStarting DPO training...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"\nSaving to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Done! Model saved. Run sparse-personality against it to see the improvement.")


if __name__ == "__main__":
    main()
