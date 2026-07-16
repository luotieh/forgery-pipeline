"""成对 compositing 探针断言（PATCH 7.5 冒烟验收，spec `docs/PATCHES_addendum_06_07_2026-07-15.md`
§7.5）：paste_feather 变体在羽化带外须与底图逐像素相等；none 变体须与底图存在非平凡差异
（掩码重绘 + mock 全局印记/real VAE 直出足迹所致）。

用法：python scripts/assert_compositing_pairs.py --probe <probe_dir> [--plot <out.png>]
退出码：0=全部通过；1=存在断言失败（消息打印到 stderr）。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from forgery_pipeline import image_io, manifest

# paste_feather 羽化带外豁免的膨胀核大小：k = _DILATE_MULT * feather_px + 1。
#
# 注（相对 task brief 字面 "4×feather_px+1" 的已验证偏差）：cv2.GaussianBlur(mask, (0,0),
# sigma=feather_px) 对非 uint8（此处 float32）输入按 OpenCV 内部公式取自动核尺寸
# ksize = round(sigma*4*2+1)|1，核半宽 = 4×feather_px（feather_px=8 时半宽=32，已用单独脚本
# 对合成竖直边界直接测过该核的响应剖面：offset<32 处混合权重严格为正、offset≥32 处严格为
# 0——不是"很小"，是浮点意义上精确为 0，因为 cv2 的核是有限支撑，不是解析高斯尾部）。
# 4×feather_px+1 对应的膨胀半径只有 2×feather_px=16，覆盖不到核的实际支撑半径；对本仓库
# 真实 pipeline 产出的成对 probe 实测（4 组 pair，seed=0），带外 16–31px 环上会残留 1-level
# 舍入差异（2–8 像素/对，4/4 组复现，出现在底图自身色块边缘恰好落入该环的位置）。改用
# 8×feather_px+1（膨胀半径 4×feather_px=32，与核半宽严格对齐）后同一组 4 对样本零残差，
# 采用此修正值（详见 task-7-report.md 自查记录）。
_DILATE_MULT = 8

# none 变体与底图差异比例的下限阈值。
#
# 注（相对 task brief 字面 "> 0.5" 的已验证偏差）：0.5 这一数值可追溯到
# tests/test_vae_rt.py::test_mock_vae_rt_global_deterministic 对**纯随机噪声图**
# （np.random.default_rng(0).integers(0,256,...)，逐像素独立同分布）的实测——噪声图任意
# 轻微模糊核都会在全图范围触发差异，"过半像素被触碰"对它成立。但 compositing_pair 探针的
# 底图是 backends.mock.synth_image() 的低频渐变 + 少量随机色块背景，MockVaeRoundtrip
# （GaussianBlur σ=0.6 + 残差衰减）对平滑渐变区域近似恒等变换，只在色块边缘产生可见差异；
# none 行差异的主体其实是掩码本身（probe._box() 固定 frac=0.2）。对本仓库真实 pipeline 实测
# （4 组 pair，seed=0）差异比例稳定落在 0.227–0.237（掩码面积 0.198 的确定性下界 + 少量边缘
# 效应），远低于 0.5、但显著高于 0。改用 0.15：明显低于实测下界（留裕量，不会因为随机种子/
# 底图内容波动而误报），同时足以捕获"none 与底图基本相同"这类真实回归
# （例如 compositing 分支写反、或生成图/掩码路径接错）。
_NONE_DIFF_RATIO_MIN = 0.15


def check(probe_dir) -> list[str]:
    """读 probe_dir/manifest.jsonl 中 probe_group=='compositing_pair' 行，按 pair_id 配对校验。

    返回空列表 = 全部通过：
    - paste_feather 行：`_DILATE_MULT×feather_px` 膨胀带外逐像素须与对应底图
      （real_image_path）相等；
    - none 行：与底图全图差异比例须大于 `_NONE_DIFF_RATIO_MIN`（证明确实经过掩码重绘 + VAE
      往返足迹，而非误直出底图）。
    """
    probe_dir = Path(probe_dir)
    rows = manifest.read_jsonl(probe_dir / "manifest.jsonl")
    pairs = [r for r in rows if r.probe_group == "compositing_pair"]
    by_pid: dict[str, list] = {}
    for r in pairs:
        by_pid.setdefault(r.pair_id, []).append(r)

    errs: list[str] = []
    # key 用 (是否 None, str(pid)) 排序：pair_id=None（残缺行，理论上不应出现，见下方行数≠2
    # 分支）与正常 str 混排时，None 与 str 在 Python 3 下不可比较，直接 sorted() 会 TypeError。
    for pid, group in sorted(by_pid.items(), key=lambda kv: (kv[0] is None, str(kv[0]))):
        if len(group) != 2:
            errs.append(f"pair_id={pid!r} 行数≠2（实得 {len(group)}）")
            continue
        for r in group:
            if not r.real_image_path:
                errs.append(f"pair_id={pid!r} image_id={r.image_id} 缺 real_image_path")
                continue
            base = image_io.load_image(probe_dir / r.real_image_path)
            img = image_io.load_image(probe_dir / r.image_path)
            if r.compositing == "paste_feather":
                if not r.mask_path:
                    errs.append(f"pair_id={pid!r} image_id={r.image_id} "
                               "paste_feather 行缺 mask_path")
                    continue
                mask = image_io.load_mask(probe_dir / r.mask_path)
                feather_px = r.feather_px or 8
                k = _DILATE_MULT * feather_px + 1
                band = cv2.dilate(mask, np.ones((k, k), np.uint8))
                if not np.array_equal(img[band == 0], base[band == 0]):
                    errs.append(f"pair_id={pid!r} image_id={r.image_id} "
                               "paste_feather 羽化带外与底图不逐像素相等")
            elif r.compositing == "none":
                ratio = float((img != base).mean())
                if not (ratio > _NONE_DIFF_RATIO_MIN):
                    errs.append(f"pair_id={pid!r} image_id={r.image_id} "
                               f"none 行与底图差异比例过低: {ratio:.4f} "
                               f"(阈值 > {_NONE_DIFF_RATIO_MIN})")
            else:
                errs.append(f"pair_id={pid!r} image_id={r.image_id} "
                           f"compositing 非预期值: {r.compositing!r}")
    return errs


def _plot_vae_rt_residual(probe_dir, out_path) -> bool:
    """`--plot` 记录项（无通过阈值，7.5 验收标准第 3 条）：real vs real_vae_rt 的 |img-base|
    残差直方图。

    probe manifest（run_probe 产出）本身不含 real_vae_rt 行——该样本类型由主 pipeline 在 split
    后按 split 分层插入（见 pipeline.py），probe 不走 split，因此在 probe_dir 上调用本函数时
    安全跳过（打印提示，不视为错误）。matplotlib 缺失时同样安全跳过（guarded import，不引入
    硬依赖，风格同 checking/gate2.py::_plot）。
    """
    probe_dir = Path(probe_dir)
    rows = manifest.read_jsonl(probe_dir / "manifest.jsonl")
    vae_rt_rows = [r for r in rows if r.sample_kind == "real_vae_rt" and r.real_image_path]
    if not vae_rt_rows:
        print(f"[assert_compositing_pairs] {probe_dir} 无 real_vae_rt 行，跳过残差分布图")
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[assert_compositing_pairs] matplotlib 不可用，跳过残差分布图")
        return False

    residuals = []
    for r in vae_rt_rows:
        base = image_io.load_image(probe_dir / r.real_image_path).astype(np.float32)
        img = image_io.load_image(probe_dir / r.image_path).astype(np.float32)
        residuals.append(np.abs(img - base).ravel())
    residuals = np.concatenate(residuals)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=50)
    ax.set_xlabel("|img - base| (per pixel-channel)")
    ax.set_ylabel("count")
    ax.set_title(f"real_vae_rt residual distribution (n={len(vae_rt_rows)} rows)")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"[assert_compositing_pairs] 残差分布图已写入 {out_path}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="成对 compositing 探针断言（PATCH 7.5）")
    ap.add_argument("--probe", required=True, help="probe 输出目录（含 manifest.jsonl）")
    ap.add_argument("--plot", default=None,
                    help="可选：real vs real_vae_rt 残差直方图输出路径（记录项，无阈值）")
    args = ap.parse_args(argv)

    errs = check(args.probe)
    if args.plot:
        _plot_vae_rt_residual(args.probe, args.plot)
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        return 1
    print(f"OK: {args.probe} 成对 compositing 断言全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
