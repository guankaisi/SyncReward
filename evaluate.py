#!/usr/bin/env python3
"""Evaluate agreement with human sync ratings (e.g. on SyncReward-Bench).

Two modes:
  1. --checkpoint CKPT --data bench.jsonl        score every clip with SyncReward, then compute metrics.
  2. --predictions preds.jsonl [--data bench.jsonl]
                                                 evaluate an existing predictions file (any model). Rows are
                                                 joined to --data on (uid, model) when --data is given;
                                                 otherwise each prediction row must carry the target itself.
"""

import argparse
import json
from pathlib import Path

from syncreward.data import read_jsonl
from syncreward.metrics import bootstrap_ci, compute_metrics
from syncreward.utils import parse_args_with_config


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="Clips jsonl with `video_path`, `uid`, `model`, `target`.")
    p.add_argument("--predictions", help="Existing predictions jsonl to evaluate (skips scoring).")
    p.add_argument("--checkpoint", help="Stage-2 checkpoint dir (or training_state.pt).")
    p.add_argument("--encoder_path", default="facebook/pe-av-large")
    p.add_argument("--output", default=None, help="Where to write predictions jsonl (scoring mode).")
    p.add_argument("--metrics_output", default=None, help="Optional json file for the metrics.")
    p.add_argument("--target_key", default="target")
    p.add_argument("--pred_key", default="reward")
    p.add_argument("--num_segments", type=int, default=18)
    p.add_argument("--segment_duration", type=float, default=0.5)
    p.add_argument("--segment_stride", type=float, default=0.25)
    p.add_argument("--crop", default="center", choices=["center", "random", "random_independent"],
                   help="`center` (default) is deterministic; `random_independent` reproduces the legacy "
                        "independent audio/video window sampling.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--encode_batch_size", type=int, default=18)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for 95%% CIs (0 = off).")
    p.add_argument("--seed", type=int, default=20260722)
    return p


def score_clips(args):
    import torch
    from torch.utils.data import DataLoader

    from syncreward.data import ClipDataset, ShardSampler, collate
    from syncreward.model import build_reward_model, load_processor, predict, processor_sample_rate
    from syncreward.utils import cleanup_distributed, is_main, setup_distributed

    rank, world, device = setup_distributed(backend="gloo")
    processor = load_processor(args.encoder_path)
    model = build_reward_model(args.encoder_path, checkpoint=args.checkpoint, num_segments=args.num_segments)
    model.to(device).eval()
    dataset = ClipDataset.from_file(args.data, processor_sample_rate(processor))
    loader = DataLoader(dataset, sampler=ShardSampler(dataset, rank, world), batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=collate)
    torch.manual_seed(args.seed)
    preds = predict(model, processor, loader, device, args, crop=args.crop)
    rows = []
    if is_main():
        for i, row in enumerate(dataset.rows):
            if i in preds:
                rows.append({**row, args.pred_key: preds[i]})
            else:
                print(f"no prediction for {row['video_path']} (missing or undecodable)", flush=True)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            print(f"wrote {len(rows)} predictions to {args.output}", flush=True)
    cleanup_distributed()
    return rows if is_main() else None


def join_targets(pred_rows, data_path, target_key, pred_key):
    if not data_path:
        return [(float(r[target_key]), float(r[pred_key])) for r in pred_rows
                if r.get(target_key) is not None and r.get(pred_key) is not None]
    preds = {(r["uid"], r["model"]): float(r[pred_key]) for r in pred_rows if r.get(pred_key) is not None}
    pairs, missing = [], 0
    for r in read_jsonl(data_path):
        if r.get(target_key) is None:
            continue
        key = (r["uid"], r["model"])
        if key in preds:
            pairs.append((float(r[target_key]), preds[key]))
        else:
            missing += 1
    if missing:
        print(f"warning: {missing} target rows have no prediction")
    return pairs


def main():
    args = parse_args_with_config(build_parser())
    if args.predictions:
        pairs = join_targets(read_jsonl(args.predictions), args.data, args.target_key, args.pred_key)
    elif args.checkpoint and args.data:
        rows = score_clips(args)
        if rows is None:
            return
        pairs = join_targets(rows, None, args.target_key, args.pred_key)
    else:
        raise SystemExit("give --predictions, or --checkpoint together with --data")
    if not pairs:
        print(f"no rows with both `{args.target_key}` and `{args.pred_key}`; nothing to evaluate")
        return

    target, pred = zip(*pairs)
    metrics = compute_metrics(target, pred)
    if args.bootstrap > 0:
        metrics["ci95"] = bootstrap_ci(target, pred, num_samples=args.bootstrap, seed=args.seed)
    print(f"n={metrics['n']}  pairs(gap>=0.75)={metrics['pairwise_pairs']}")
    for name in ("spearman", "pearson", "kendall", "pairwise_accuracy", "mae"):
        ci = metrics.get("ci95", {}).get(name)
        extra = f"  [95% CI {ci['low']:.4f}, {ci['high']:.4f}]" if ci else ""
        print(f"{name:>18s}: {metrics[name]:.4f}{extra}")
    if args.metrics_output:
        Path(args.metrics_output).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
