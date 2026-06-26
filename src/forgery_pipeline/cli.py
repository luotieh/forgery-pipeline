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
    print(f"OK: {len(samples)} 条样本全部通过校验")
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
    from forgery_pipeline.builders.probe import run_probe
    cfg = load_config(args.config)
    st = run_probe(
        args.out, n_base=args.n_base,
        strengths=[round(0.1 * i, 1) for i in range(1, 10)],
        operators=["img2img", "inpaint", "outpaint",
                   "object_replacement", "background_editing"],
        img2img_specs=cfg.img2img, inpainter_specs=cfg.inpainters,
        backend=cfg.backend, seed=cfg.seed,
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
    p_val.set_defaults(func=_cmd_validate)

    p_view = sub.add_parser("viewer", help="生成数据集可视化 viewer.html")
    p_view.add_argument("--run", required=True, help="run 目录（含 manifest.jsonl）")
    p_view.add_argument("--out", default=None, help="输出 html 路径，默认 <run>/viewer.html")
    p_view.add_argument("--max", type=int, default=None, help="最多渲染样本数")
    p_view.add_argument("--open", action="store_true", help="生成后尝试用浏览器打开")
    p_view.set_defaults(func=_cmd_viewer)

    p_probe = sub.add_parser("probe", help="生成 Gate 1/2 受控 probe 子集")
    p_probe.add_argument("--config", required=True)
    p_probe.add_argument("--out", default="data/probe")
    p_probe.add_argument("--n-base", type=int, default=40, dest="n_base")
    p_probe.set_defaults(func=_cmd_probe)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
