"""Audit preference trajectories and optionally fit a transition reward MLP."""

import argparse
import json
from pathlib import Path

import torch

from src.preference_reward import (FEATURE_SCHEMA_VERSION, TERMINAL_REASONS,
                                   fit_preference_reward, load_and_audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-spec", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path,
                        help="Omit to audit/prepare only; no fitting is performed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.audit_output.exists():
        parser.error(f"audit output exists: {args.audit_output}")
    trajectories, pairs, audit = load_and_audit(args.data_spec)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    if not audit["ok"]:
        raise SystemExit("data audit failed; see audit output")
    if args.model_output is None:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return
    if args.model_output.exists():
        parser.error(f"model output exists: {args.model_output}")
    if not audit["fit_ready"]:
        raise SystemExit("fitting refused: independent train/validation/test pairs are required")
    model, normalizer, history, test_loss = fit_preference_reward(
        trajectories, pairs, epochs=args.epochs, learning_rate=args.learning_rate,
        seed=args.seed)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "type": "TransitionRewardMLP", "input_dim": audit["feature_dim"],
            "hidden_dims": [256, 128], "feature_schema": FEATURE_SCHEMA_VERSION,
            "terminal_reasons": list(TERMINAL_REASONS),
        },
        "normalizer_mean": normalizer["mean"],
        "normalizer_scale": normalizer["scale"],
        "audit": audit,
        "training": {"epochs": args.epochs, "learning_rate": args.learning_rate,
                     "seed": args.seed, "history": history, "test_loss": test_loss},
    }, args.model_output)
    print(json.dumps({"model_output": str(args.model_output), "test_loss": test_loss}, indent=2))


if __name__ == "__main__":
    main()
