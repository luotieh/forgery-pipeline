"""闸门执行入口：跑 gate0-3 + gate4_eval，打印 VERDICT，写 report.json。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from checking.extractor import get_extractor
from checking import gate0, gate1, gate2, gate3, gate4_eval

_CAVEAT = ("extractor=multisigma 是 CPU 代理信号：在 mock 数据上的 VERDICT 仅验证分析代码通路，"
           "非科学结论（甚至可能假阳性）。真实判定需 extractor=real（SD2）+ 真实扩散生成数据 + GPU。")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="checking.run_gates", description="闸门执行测试")
    ap.add_argument("--run", default="data/run")
    ap.add_argument("--probe", default="data/probe")
    ap.add_argument("--extractor", default="multisigma")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--out", default="data/checking_report.json")
    args = ap.parse_args(argv)
    ext = get_extractor(args.extractor)
    plot = str(Path(args.out).with_name("gate2_pca.png"))
    gates = {
        "gate0": gate0.run(args.run, ext, max_n=args.max or 200),
        "gate1": gate1.run(args.probe, ext, max_n=args.max),
        "gate2": gate2.run(args.probe, ext, max_n=args.max, plot_path=plot),
        "gate3": gate3.run(args.probe, args.run, ext, max_n=args.max),
        "gate4": gate4_eval.run(args.run, ext, max_n=args.max),
    }
    for k, r in gates.items():
        print(f"[{k}] VERDICT={r['verdict']}  {json.dumps(r['metrics'], ensure_ascii=False)}")
    print("CAVEAT:", _CAVEAT)
    report = {"extractor": args.extractor, "caveat": _CAVEAT, "gates": gates}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report ->", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
