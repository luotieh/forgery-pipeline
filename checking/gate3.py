"""闸门 3：多 σ 增量 + 跨生成器掉点。"""
from __future__ import annotations
from pathlib import Path
from checking import data, gate1
from checking.metrics import NearestCentroid, balanced_accuracy, group_split


def run(probe_dir, run_dir, extractor, max_n=None) -> dict:
    probe_dir = Path(probe_dir)
    g1 = gate1.run(probe_dir, extractor, max_n=max_n)
    delta = round(g1["metrics"]["multi_sigma_acc"] - g1["metrics"]["single_sigma_acc"], 4)
    samples = data.load(probe_dir / "gate2_operator.jsonl")
    if max_n:
        samples = samples[:max_n]
    X, kept = data.profiles(extractor, probe_dir, samples)
    ops = [s.operator for s in kept]
    seen = [i for i, s in enumerate(kept) if s.split == "train"]
    held = [i for i, s in enumerate(kept) if s.split == "test_b"]
    n_ops = len(set(ops)) or 1
    random_acc = 1.0 / n_ops
    seen_acc = heldout_acc = 0.0
    if seen and held and len(set(ops[i] for i in seen)) >= 2:
        Xseen = X[seen]; yseen = [ops[i] for i in seen]
        gseen = [kept[i].real_image_path or kept[i].image_id for i in seen]
        tr, te = group_split(gseen, test_frac=0.4, seed=0)
        if tr and te and len(set(yseen[i] for i in tr)) >= 2:
            seen_acc = balanced_accuracy(
                [yseen[i] for i in te],
                NearestCentroid().fit(Xseen[tr], [yseen[i] for i in tr]).predict(Xseen[te]))
        heldout_acc = balanced_accuracy(
            [ops[i] for i in held], NearestCentroid().fit(Xseen, yseen).predict(X[held]))
    verdict = "PASS" if delta > 0 and heldout_acc > random_acc else "PARTIAL"
    return {"gate": 3,
            "metrics": {"multi_sigma_delta": delta, "seen_acc": round(seen_acc, 4),
                        "heldout_acc": round(heldout_acc, 4),
                        "cross_generator_drop": round(seen_acc - heldout_acc, 4),
                        "random_acc": round(random_acc, 4)},
            "verdict": verdict,
            "note": "跨生成器崩得厉害→第二篇泛化动机；部分掉点可接受"}
