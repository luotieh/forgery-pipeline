"""命令行入口（argparse）。默认 mock backend，开箱即跑。"""
from __future__ import annotations
import argparse
import dataclasses
import json
import sys
import webbrowser
from pathlib import Path
from forgery_pipeline import manifest
from forgery_pipeline.config import load_config
from forgery_pipeline.pipeline import run_pipeline


def _cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.out:
        cfg = dataclasses.replace(cfg, out_dir=args.out)
    st = run_pipeline(cfg)
    print(json.dumps(st, ensure_ascii=False, indent=2))
    return 0


def _cmd_stats(args) -> int:
    samples = manifest.read_jsonl(args.path)
    print(json.dumps(manifest.stats(samples), ensure_ascii=False, indent=2))
    return 0


def _cmd_validate(args) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"manifest 不存在: {path}", file=sys.stderr)
        return 2
    try:
        samples = manifest.read_jsonl(path)  # 逐行 schema 校验
    except Exception as e:  # noqa: BLE001
        print(f"manifest 校验失败: {e}", file=sys.stderr)
        return 1
    from forgery_pipeline.validate import check_all
    holdout_generators = None
    testc_holdout = None
    split_config = Path(args.split_config)
    if split_config.exists():
        import yaml
        rules = yaml.safe_load(split_config.read_text(encoding="utf-8")) or {}
        holdout_generators = set(rules.get("holdout_generators", [])) or None
        testc_holdout = rules.get("testc_holdout") or None

    # 诚实输出：V1–V7 恒执行；V8/V10/V11/V12 仅 profile=="run" 执行（裁决B），V9/V10 另需
    # split-config 提供对应 holdout 清单——三者任一未满足，该检查就没有真的跑过，成功行
    # 不得笼统宣称"V1–V10 通过"（见 forgery_pipeline.validate 模块 docstring 裁决B）。
    # W2T6：V11/V12（PATCH 9 Wave 2 nuisance 记录/面积分桶下限）与 V8/V10 同一门控条件
    # （check_all 内部已按 profile 各自门控，此处只是让 cli 的 executed/skipped 记账追上
    # check_all 实际的执行范围——此前 executed 硬编码 V1–V10，profile=="run" 时 V11/V12
    # 明明已执行却从未出现在成功行，是记账缺口而非执行缺口）。
    skipped: list[str] = []
    if args.profile != "run":
        skipped += ["V8", "V10"]
        print("注意：V8/V10 未执行（profile=auto，用 --profile run 启用）")
        skipped += ["V11", "V12"]
        print("注意：V11/V12 未执行（profile=auto，用 --profile run 启用）")
    missing_split_cfg = []
    if holdout_generators is None:
        missing_split_cfg.append("V9")
    if testc_holdout is None and "V10" not in skipped:
        missing_split_cfg.append("V10")
    if missing_split_cfg:
        skipped += missing_split_cfg
        print(f"注意：{'/'.join(missing_split_cfg)} 跳过（--split-config 未提供 holdout 清单）")

    errs = check_all(samples, profile=args.profile,
                     holdout_generators=holdout_generators, testc_holdout=testc_holdout)
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        return 1
    executed = [c for c in (f"V{i}" for i in range(1, 13)) if c not in skipped]
    print(f"OK: {len(samples)} 条样本通过 schema + {', '.join(executed)} 校验")
    return 0


def _cmd_viewer(args) -> int:
    from forgery_pipeline.viewer import build_viewer
    run = Path(args.run)
    if not (run / "manifest.jsonl").exists():
        print(f"run 目录缺少 manifest.jsonl: {run}", file=sys.stderr)
        return 2
    out = build_viewer(run, out_html=args.out, max_samples=args.max)
    print(f"已生成 {out.resolve()}")
    if args.open:
        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception:
            pass
    return 0


def _cmd_probe(args) -> int:
    import yaml
    from forgery_pipeline.config import load_generators
    from forgery_pipeline.builders.probe import run_probe
    data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    _, inps, imgs = load_generators(data["generators_config"])
    out = args.out or data.get("out_dir", "data/probe")
    n_base = args.n_base if args.n_base is not None else int(data.get("n_base", 40))
    st = run_probe(
        out, n_base=n_base,
        strengths=data.get("strengths", [round(0.1 * i, 1) for i in range(1, 10)]),
        operators=data.get("operators",
                           ["img2img", "inpaint", "outpaint",
                            "object_replacement", "background_editing"]),
        img2img_specs=imgs, inpainter_specs=inps,
        holdout_generators=set(data.get("holdout_generators", [])),
        backend=data.get("backend", "mock"), seed=int(data.get("seed", 0)),
        cfg_grid=data.get("cfg_grid"), steps_grid=data.get("steps_grid"),
        compositing_pairs=int(data.get("compositing_pairs", 0)),
    )
    print(json.dumps(st, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forgery-pipeline",
                                     description="伪造检测数据集生成 pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="运行完整 pipeline")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", default=None, help="覆盖 out_dir")
    p_run.set_defaults(func=_cmd_run)

    p_stats = sub.add_parser("stats", help="统计 manifest")
    p_stats.add_argument("--path", required=True)
    p_stats.set_defaults(func=_cmd_stats)

    p_val = sub.add_parser("validate-manifest", help="逐行校验 manifest")
    p_val.add_argument("--path", required=True)
    p_val.add_argument("--profile", default="auto", choices=["auto", "run"])
    p_val.add_argument("--split-config", default="configs/split.yaml",
                       help="V9/V10 holdout 清单来源（存在才读；缺省文件不存在则两者跳过）")
    p_val.set_defaults(func=_cmd_validate)

    p_view = sub.add_parser("viewer", help="生成数据集可视化 viewer.html")
    p_view.add_argument("--run", required=True, help="run 目录（含 manifest.jsonl）")
    p_view.add_argument("--out", default=None, help="输出 html 路径，默认 <run>/viewer.html")
    p_view.add_argument("--max", type=int, default=None, help="最多渲染样本数")
    p_view.add_argument("--open", action="store_true", help="生成后尝试用浏览器打开")
    p_view.set_defaults(func=_cmd_viewer)

    p_probe = sub.add_parser("probe", help="生成 Gate 1/2 受控 probe 子集")
    p_probe.add_argument("--config", required=True)
    p_probe.add_argument("--out", default=None, help="输出目录，默认取 config 的 out_dir")
    p_probe.add_argument("--n-base", type=int, default=None, dest="n_base",
                         help="底图数，默认取 config 的 n_base")
    p_probe.set_defaults(func=_cmd_probe)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
