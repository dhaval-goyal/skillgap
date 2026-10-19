"""
Reproducible DistilBERT training script for Technical Knowledge Extraction (tags_knowledge).

Trained on Hugging Face dataset 'jjzha/skillspan' targeting the 'tags_knowledge'
entity field (software, tools, technical concepts, databases).

Key Principles:
1. Controlled comparison: Uses distilbert-base-uncased, lr=5e-5, epochs=3, seed=42.
2. Separate checkpoint: Saved to 'checkpoints/knowledge_experiment' (preserving original checkpoint).
3. Split discipline: Validation set used strictly for checkpoint selection (best F1).
   Held-out test set evaluated once at the conclusion.
4. Entity-level metrics: Span-based precision, recall, and F1 via compute_entity_metrics.
5. Post-training diagnostics: Evaluates on gold SkillSpan validation examples and sanity check prompts.
"""
import os
import sys
import argparse
import random
import numpy as np
from typing import Dict, Any, List

# Ensure skillgap root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.prepare_dataset import align_labels_with_tokens, preprocess_example
from model.evaluation import compute_entity_metrics, DIAGNOSTIC_SUITE, evaluate_diagnostic_suite

ID2LABEL = {0: "O", 1: "B-SKILL", 2: "I-SKILL"}
LABEL2ID = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def prepare_hf_dataset(
    dataset_name: str = "jjzha/skillspan",
    target_field: str = "tags_knowledge",
    tokenizer: Any = None,
    max_length: int = 256,
    label_all_subtokens: bool = False,
):
    """
    Load jjzha/skillspan and align word-level tags with tokenizer subtokens.
    """
    from datasets import load_dataset

    print(f"Loading '{dataset_name}' from Hugging Face Hub...")
    raw_datasets = load_dataset(dataset_name)

    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )

        all_labels = []
        for i, tags in enumerate(examples[target_field]):
            tokens = examples["tokens"][i]
            # Convert tags to canonical BIO labels (B-SKILL, I-SKILL, O)
            bio_tags = preprocess_example(tokens, tags, entity_name="SKILL")["labels"]
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned = align_labels_with_tokens(
                labels=bio_tags,
                word_ids=word_ids,
                label2id=LABEL2ID,
                label_all_subtokens=label_all_subtokens,
            )
            all_labels.append(aligned)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    processed = raw_datasets.map(
        tokenize_and_align,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc=f"Aligning {target_field} labels",
    )
    return processed


def create_compute_metrics_fn():
    """Build compute_metrics callback for HuggingFace Trainer using span evaluation."""
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)

        batch_pred_tags: List[List[str]] = []
        batch_gold_tags: List[List[str]] = []

        for p_seq, l_seq in zip(preds, labels):
            cur_preds: List[str] = []
            cur_golds: List[str] = []
            for p, l in zip(p_seq, l_seq):
                if l != -100:
                    cur_preds.append(ID2LABEL.get(p, "O"))
                    cur_golds.append(ID2LABEL.get(l, "O"))
            batch_pred_tags.append(cur_preds)
            batch_gold_tags.append(cur_golds)

        return compute_entity_metrics(batch_pred_tags, batch_gold_tags)

    return compute_metrics


def run_experiment(args: argparse.Namespace):
    """Orchestrate dataset loading, training, evaluation, and artifact saving."""
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        TrainingArguments,
        Trainer,
    )

    set_seed(args.seed)
    print("=" * 60)
    print("SkillGap Stage 12: Technical Knowledge Extraction Experiment")
    print(f"Target Field:      {args.target_field}")
    print(f"Base Model:        {args.model_name}")
    print(f"Learning Rate:     {args.learning_rate}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Max Seq Length:    {args.max_length}")
    print(f"Output Directory:  {args.output_dir}")
    print(f"Random Seed:       {args.seed}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_datasets = prepare_hf_dataset(
        dataset_name=args.dataset_name,
        target_field=args.target_field,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_all_subtokens=args.label_all_subtokens,
    )

    if args.dry_run:
        print("\n" + "=" * 60)
        print("[DRY-RUN] Executing verification checks (non-destructive)...")
        print("=" * 60)

        # 1. Verify dataset structure & slicing
        print("1. Checking dataset splits...")
        print(f"   Train samples available:      {len(tokenized_datasets['train'])}")
        print(f"   Validation samples available: {len(tokenized_datasets['validation'])}")
        print(f"   Test samples available:       {len(tokenized_datasets['test'])}")

        # 2. Verify subtoken alignment on first train sample
        print("2. Verifying subtoken alignment and special token masking...")
        sample = tokenized_datasets["train"][0]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        assert len(input_ids) == len(labels), f"Length mismatch: {len(input_ids)} ids vs {len(labels)} labels"
        assert labels[0] == -100, "First token ([CLS]) must be masked with -100"
        print(f"   Sample input_ids length: {len(input_ids)}, labels length: {len(labels)} (Verified)")

        # 3. Verify model initialization & forward pass
        print("3. Verifying model instantiation and forward pass...")
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(ID2LABEL),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model.eval()

        pad_token_id = tokenizer.pad_token_id or 0
        s0_ids = tokenized_datasets["train"][0]["input_ids"][:32]
        s1_ids = tokenized_datasets["train"][1]["input_ids"][:32]
        s0_mask = tokenized_datasets["train"][0]["attention_mask"][:32]
        s1_mask = tokenized_datasets["train"][1]["attention_mask"][:32]
        s0_lbl = tokenized_datasets["train"][0]["labels"][:32]
        s1_lbl = tokenized_datasets["train"][1]["labels"][:32]

        batch_ids = torch.tensor([s0_ids, s1_ids])
        batch_mask = torch.tensor([s0_mask, s1_mask])
        batch_labels = torch.tensor([s0_lbl, s1_lbl])

        with torch.no_grad():
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_labels)

        assert outputs.loss is not None, "Model forward pass failed to return loss"
        assert outputs.logits.shape == (2, 32, len(ID2LABEL)), f"Unexpected logits shape: {outputs.logits.shape}"
        print(f"   Dummy forward loss: {outputs.loss.item():.4f}, logits shape: {list(outputs.logits.shape)} (Verified)")

        # 4. Verify span evaluation callback
        print("4. Verifying span evaluation callback...")
        compute_metrics = create_compute_metrics_fn()
        dummy_preds = (outputs.logits.detach().cpu().numpy(), batch_labels.detach().cpu().numpy())
        metrics_res = compute_metrics(dummy_preds)
        print(f"   Sample metrics computed: {metrics_res} (Verified)")

        print("\n" + "=" * 60)
        print("[DRY-RUN COMPLETE] Pipeline verified successfully!")
        print("SAFETY CONFIRMED:")
        print(" - No model training was executed.")
        print(" - Held-out test split was NOT touched or evaluated.")
        print(" - Existing checkpoints were NOT modified.")
        print("=" * 60)
        return {
            "dry_run": True,
            "status": "verified",
            "dummy_loss": float(outputs.loss.item()),
            "metrics_keys": list(metrics_res.keys()),
        }

    # Full training pipeline (only executed when --dry_run is NOT set)
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]
    epochs = args.epochs

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    import inspect
    ta_sig = inspect.signature(TrainingArguments.__init__)
    eval_arg = "eval_strategy" if "eval_strategy" in ta_sig.parameters else "evaluation_strategy"

    ta_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": epochs,
        "weight_decay": args.weight_decay,
        eval_arg: "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": args.seed,
        "logging_steps": 20,
        "report_to": "none",
    }
    training_args = TrainingArguments(**ta_kwargs)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer_sig = inspect.signature(Trainer.__init__)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "compute_metrics": create_compute_metrics_fn(),
    }
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("\nStarting model training...")
    trainer.train()

    print("\nEvaluating on validation set (Model Selection Split)...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation F1:        {val_metrics.get('eval_f1', 0.0):.4f}")
    print(f"Validation Precision: {val_metrics.get('eval_precision', 0.0):.4f}")
    print(f"Validation Recall:    {val_metrics.get('eval_recall', 0.0):.4f}")

    # Save best checkpoint and tokenizer
    best_checkpoint_dir = os.path.join(args.output_dir, "best_model")
    print(f"\nSaving best checkpoint to '{best_checkpoint_dir}'...")
    trainer.save_model(best_checkpoint_dir)
    tokenizer.save_pretrained(best_checkpoint_dir)

    print("\nEvaluating on held-out test set (Official Final Evaluation)...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(f"Test F1:              {test_metrics.get('test_f1', 0.0):.4f}")
    print(f"Test Precision:       {test_metrics.get('test_precision', 0.0):.4f}")
    print(f"Test Recall:          {test_metrics.get('test_recall', 0.0):.4f}")

    # Post-training diagnostic suite run
    print("\nRunning diagnostic evaluation suite on best model...")
    from inference.extractor import SkillExtractor
    extractor = SkillExtractor(model_path=best_checkpoint_dir)
    diag_results = evaluate_diagnostic_suite(extractor)

    print(f"Diagnostic Gold Recall:       {diag_results['gold_summary']['overall_recall']:.2%}")
    print(f"Diagnostic Sanity Hit Rate:   {diag_results['sanity_summary']['overall_hit_rate']:.2%}")

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "diagnostic_results": diag_results,
    }


def main():
    parser = argparse.ArgumentParser(description="SkillGap Stage 12: Train DistilBERT on tags_knowledge")
    parser.add_argument("--dataset_name", type=str, default="jjzha/skillspan", help="Hugging Face dataset name")
    parser.add_argument("--target_field", type=str, default="tags_knowledge", help="Annotation field to train on")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model")
    parser.add_argument("--output_dir", type=str, default="checkpoints/knowledge_experiment", help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--label_all_subtokens", action="store_true", help="Label continuation subtokens")
    parser.add_argument("--dry_run", action="store_true", help="Quick verification run on a tiny data slice")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
