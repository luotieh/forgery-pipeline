# tests/test_resolution_groups.py —— PATCH 9 Wave 2 Task 4：多分辨率组摄取（TDD 先红）
#
# 9.2c 要求：每个生成器分辨率组（SD1.5@512 / SDXL@1024 等）各自配套同链 real + vae_rt
# 行——否则 PATCH 7 的 V2（split 内 real/fake 非生成链集合相等）在组内是空判据。
# build_d0(resolutions=[...]) 对每张通过 QC/去重的源图，按 resolutions（pipeline 传入前
# 已排序）逐 size 产一行；D2/D3/grid 只消费 resolutions[0]（基准组）底图行，其余分辨率组
# 只出 real（+vae_rt）行，fake 侧覆盖改由 grid 按 policies.resolution_groups 对 img2img
# spec 名分组路由（见 test_pipeline_multi_resolution_groups_e2e_v2_holds）。
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from forgery_pipeline import image_io, manifest
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.config import GeneratorSpec, PipelineConfig, StageScales, load_config
from forgery_pipeline.pipeline import run_pipeline
from forgery_pipeline.split.grouping import origin_key
from forgery_pipeline.split.splitter import assign_splits
from forgery_pipeline.validate import check_all


def _res_cover_invariant(rows) -> None:
    """9.2c 不变量本体（裁决4）：对每个含 real 行的 split，fake 的分辨率集合 ⊆ real 的
    分辨率集合——即「含 fake@rs{r} 的 split 必有 real@rs{r} 行」。real 侧为空的 split
    跳过（test_e 退化 carve-out 结构性无 real 行，与 check_v2 的豁免口径一致）。"""
    real_res: dict = defaultdict(set)
    fake_res: dict = defaultdict(set)
    for r in rows:
        res = image_io.chain_resolution(r.io_chain)
        if res is not None and r.split:
            (fake_res if r.is_fake else real_res)[r.split].add(res)
    for split, fres in fake_res.items():
        if real_res[split]:
            assert fres <= real_res[split], (split, fres, real_res[split])


# ---------------------------------------------------------------------------
# build_d0(resolutions=...)：每源图按 size 产一行
# ---------------------------------------------------------------------------

def test_build_d0_resolutions_produces_per_size_rows_with_shared_base_id(tmp_path):
    rows = build_d0(tmp_path, n=3, seed=0, resolutions=[64, 128])
    assert len(rows) == 6

    # 按摄取顺序两两成对（每源图先产 resolutions[0]=64 行，再产 128 行）
    pairs = [(rows[i], rows[i + 1]) for i in range(0, len(rows), 2)]
    assert len(pairs) == 3
    for r64, r128 in pairs:
        assert r64.io_chain == "decode>rs64>png"
        assert r128.io_chain == "decode>rs128>png"
        # 裁决：base_id 组键取首分辨率行（resolutions[0]=64）的 image_id
        assert r64.base_id == r64.image_id
        assert r128.base_id == r64.image_id
        # 尺寸正确（载图断言 shape）
        img64 = image_io.load_image(Path(tmp_path) / r64.image_path)
        img128 = image_io.load_image(Path(tmp_path) / r128.image_path)
        assert img64.shape == (64, 64, 3)
        assert img128.shape == (128, 128, 3)
        assert r64.is_fake == 0 and r128.is_fake == 0
        assert r64.sample_kind == "real" and r128.sample_kind == "real"

    # image_id 全不同（源图 content hash + 分辨率后缀区分）
    assert len({r.image_id for r in rows}) == 6


def test_build_d0_resolutions_none_matches_head_behavior(tmp_path):
    """resolutions=None（含未传参的默认值）与 HEAD 行为逐字段一致：回归锚。"""
    implicit = build_d0(tmp_path / "implicit", n=3, seed=0)
    explicit_none = build_d0(tmp_path / "explicit_none", n=3, seed=0, resolutions=None)
    assert len(implicit) == len(explicit_none) == 3
    for a, b in zip(implicit, explicit_none):
        assert a.image_id == b.image_id == a.base_id      # 无 -r 后缀语义差异、自指 base_id
        assert a.io_chain == b.io_chain
        assert a.sample_kind == b.sample_kind == "real"
        assert Path(a.image_path).name == Path(b.image_path).name


def test_build_d0_resolutions_qc_dedup_runs_once_per_source_not_per_size(tmp_path):
    """QC/去重只对源图做一次：n=5 时用 resolutions=[64,128] 与 resolutions=None 应接受
    同样数量的源图（即 resolutions=[64,128] 产 2*5=10 行，而非因重复 QC/去重导致源图
    接受数量漂移）。"""
    single = build_d0(tmp_path / "single", n=5, seed=0)
    multi = build_d0(tmp_path / "multi", n=5, seed=0, resolutions=[64, 128])
    assert len(single) == 5
    assert len(multi) == 10


def test_build_d0_resolutions_origin_key_matches_base_id_group(tmp_path):
    """回归锚（实现期间发现并修正的必要一步，见 d0_real.build_d0 docstring）：
    `split/grouping.origin_key()` 只做一跳解析（real_image_path 或 image_path 的
    stem），不直接认 base_id 字段。非基准分辨率行必须把 real_image_path 回填到基准行，
    否则它会以自己的路径独立成组，与同 base_id 的基准行被 assign_splits 各自独立
    哈希——V8（同 base_id 须同 split）几乎必红（已用最小复现验证：不回填时 12 行、
    6 组几乎全部组内跨 split）。这条测试直接钉住 origin_key 与 base_id 分组一致，比
    对 assign_splits 输出做端到端断言更贴近根因。"""
    rows = build_d0(tmp_path, n=6, seed=0, resolutions=[64, 128])
    by_base_id: dict[str, set] = {}
    by_origin_key: dict[str, set] = {}
    for r in rows:
        by_base_id.setdefault(r.base_id, set()).add(r.image_id)
        by_origin_key.setdefault(origin_key(r), set()).add(r.image_id)
    assert by_base_id == by_origin_key

    # 端到端复核：assign_splits 之后同 base_id 组必须同 split（V8 的数据前提，
    # 同 tests/test_base_id.py 的断言风格）。
    assign_splits(rows, holdout_generators=[], holdout_manipulation=[], seed=0)
    by_base = {}
    for r in rows:
        by_base.setdefault(r.base_id, set()).add(r.split)
    assert all(len(s) == 1 for s in by_base.values()), by_base


# ---------------------------------------------------------------------------
# load_config：resolution_groups 字符串键 coercion（W2T1 遗留回归测试）
# ---------------------------------------------------------------------------

def test_load_config_resolution_groups_string_keys_coerced_to_int(tmp_path):
    """yaml 里 resolution_groups 的键若被解析成字符串（如显式加引号的 "512"），
    load_config 须 coerce 成 int——W2T1 review 记录的遗留缺口，本任务补齐回归测试。"""
    yaml_text = (
        f'out_dir: "{tmp_path / "run"}"\n'
        "seed: 0\n"
        "backend: mock\n"
        "stages: {d0: true}\n"
        "scales: {d0: 2}\n"
        "generators_config: configs/generators.yaml\n"
        "resolution_groups:\n"
        '  "512": [stable-diffusion-img2img]\n'
        '  "1024": [sdxl-img2img]\n'
    )
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    # 前提：yaml 确实把键解析成了字符串（否则这条测试没有测到 coercion 分支本身）
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert set(raw["resolution_groups"].keys()) == {"512", "1024"}
    assert all(isinstance(k, str) for k in raw["resolution_groups"])

    cfg = load_config(str(p))
    assert cfg.resolution_groups == {512: ["stable-diffusion-img2img"],
                                     1024: ["sdxl-img2img"]}
    assert all(isinstance(k, int) for k in cfg.resolution_groups)


# ---------------------------------------------------------------------------
# pipeline e2e：两个分辨率组各自配套 real+fake，9.2c 不变量 + 组成规则（裁决4 重写）
# ---------------------------------------------------------------------------

def _run_multires_e2e(out_dir, seed: int, d0: int) -> list:
    """多分辨率 e2e 主体（裁决4 参数化：sweep 脚本与正式测试跑同一份断言代码）。
    断言全部为机制的确定性性质，任意 seed/规模成立。"""
    cfg = PipelineConfig(
        out_dir=str(Path(out_dir) / "run"), seed=seed, backend="mock",
        stages={"d0": True, "d1": False, "d2": True, "d3": True, "d4": True,
                "grid": True, "postprocess": True, "split": True},
        scales=StageScales(d0=d0, d1_per_generator=0, d2=10, d3=10, d4=3),
        inpainters=[GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")],
        img2img=[GeneratorSpec("a", "diffusion", "img2img"),
                 GeneratorSpec("b", "diffusion-sdxl", "img2img")],
        resolution_groups={64: ["a"], 128: ["b"]},
        grid_per_op=d0,           # ≥ len(grid_pool)：覆盖二分后 grid 池的全部底图
        vae_rt_frac=0.25,
        split_config="configs/split.yaml",   # 生产配置（含 base_resolution_only_splits）
    )
    st = run_pipeline(cfg)                    # 无 RuntimeError（泄漏机制绿）
    assert st["total"] > 0
    rows = manifest.read_jsonl(Path(cfg.out_dir) / "manifest.jsonl")

    # D0 多分辨率摄取：real 侧双链存在
    real_chains = {r.io_chain for r in rows if r.is_fake == 0 and r.sample_kind == "real"}
    assert {"decode>rs64>png", "decode>rs128>png"} <= real_chains

    # 9.2c 不变量本体：含 fake@rs{r} 的 split 必有 real@rs{r}
    _res_cover_invariant(rows)

    # io_chain_by_fake_split：存在 split 其 real 侧同时含 rs64/rs128 双链
    by_split = manifest.stats(rows)["io_chain_by_fake_split"]
    assert any("rs64>png" in chains and chains["rs64>png"]["real"] > 0
              and "rs128>png" in chains and chains["rs128>png"]["real"] > 0
              for chains in by_split.values()), by_split

    # grid 按组路由确定性产 rs128 fake（"b" 非 holdout → 每个 grid 池底图各一行）
    b_rows = [r for r in rows if r.generator_name == "b"]
    assert b_rows
    assert {image_io.chain_resolution(r.io_chain) for r in b_rows} == {128}

    # base_id 组（含跨分辨率兄弟行）split 全一致（V8 的数据前提，端到端复核）
    by_base: dict = {}
    for r in rows:
        if r.postprocess_of:
            continue
        by_base.setdefault(r.base_id, set()).add(r.split)
    assert all(len(s) == 1 for s in by_base.values()), by_base

    # test_c 组成规则：过滤后只含基准分辨率（Test-C 测算子泛化，分辨率非其轴）
    tc = [r for r in rows if r.split == "test_c"]
    assert tc, "前提：生产 holdout_manipulation 确实把 D2 组路由进了 test_c"
    assert all(image_io.chain_resolution(r.io_chain) == 64 for r in tc), (
        [(r.image_id, r.io_chain) for r in tc])

    # 过滤不拆散 postprocess 母子行：留存行的 postprocess_of 必须仍指向留存行
    ids_present = {r.image_id for r in rows}
    assert all(r.postprocess_of in ids_present for r in rows if r.postprocess_of)
    return rows


def test_pipeline_multi_resolution_groups_e2e_v2_holds(tmp_path):
    """9.2c 机制作用域断言（裁决4 重写，任意 seed/规模必过）：生产 configs/split.yaml +
    resolution_groups={64,128} 下——D0 双分辨率 real 行齐备；9.2c 不变量本体（任何含
    real 的 split 内 fake 分辨率集合 ⊆ real 分辨率集合）；grid 按组路由确定性产 rs128
    fake；V8 组语义（同 base_id 同 split）成立；test_c 组成规则过滤后只含基准分辨率、
    postprocess 母子不拆散。

    不再断言全局 check_all()==[]（裁决4）：那会把无关 split 的小 n 组合噪声（如 val
    恰好只抽到无 rs128 fake 的组）一并背上——mock 规模下是 seed/规模掷币，且放大规模
    修不了；机制正确性由上述确定性性质直接覆盖。参数化主体 _run_multires_e2e 供裁决4
    sweep 复用（seed∈{0..5}×d0∈{16,20,28} 全过，见 report「裁决执行 4」）。"""
    _run_multires_e2e(tmp_path, seed=0, d0=20)


def test_pipeline_without_base_resolution_only_splits_keeps_sibling_rows(tmp_path):
    """回归锚：split 配置缺 base_resolution_only_splits 键（旧式配置；显式空列表走同一
    falsy 分支）→ 不过滤，test_c 保留非基准分辨率的兄弟 real 行——证明组成规则过滤严格
    config 门控，键缺省时多分辨率行为与裁决落地前一致。"""
    legacy_split = tmp_path / "split_legacy.yaml"
    legacy_split.write_text(yaml.dump({
        "holdout_generators": [],
        "holdout_manipulation": ["text_editing", "object_replacement"],
        "holdout_domains": [],
    }), encoding="utf-8")
    cfg = PipelineConfig(
        out_dir=str(tmp_path / "run"), seed=0, backend="mock",
        stages={"d0": True, "d1": False, "d2": True, "d3": False, "d4": False,
                "grid": False, "postprocess": False, "split": True},
        scales=StageScales(d0=20, d1_per_generator=0, d2=10, d3=0, d4=0),
        inpainters=[GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")],
        resolution_groups={64: ["a"], 128: ["b"]},
        vae_rt_frac=0.0,
        split_config=str(legacy_split),
    )
    run_pipeline(cfg)
    rows = manifest.read_jsonl(Path(cfg.out_dir) / "manifest.jsonl")
    tc = [r for r in rows if r.split == "test_c"]
    assert tc, "前提：holdout_manipulation 确实把 D2 组路由进了 test_c"
    res_in_tc = {image_io.chain_resolution(r.io_chain) for r in tc}
    assert 128 in res_in_tc, res_in_tc   # 兄弟分辨率行仍在——未被过滤（旧行为锚）


# ---------------------------------------------------------------------------
# test_b 分辨率覆盖设计约束（裁决执行 2）：正负双回归
# 约束：holdout 生成器须覆盖每个参与 test_b 的分辨率组；违反时 V2 红是期望的响亮失败
# （test_b 刻意不进 base_resolution_only_splits——跨生成器轴需要各分辨率组的 holdout
# fake，过滤会把 1024 侧整个删掉，见 configs/split.yaml 注释）。
# ---------------------------------------------------------------------------

def _check_testb_coverage_positive(out_dir, seed: int, d0: int) -> dict:
    """测试 A 主体（裁决4 参数化，任意 seed/规模必过）。返回两条 hold 路径的 firing
    统计供 sweep 汇报非空泛率。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = out_dir / "split_b3.yaml"
    sp.write_text(yaml.dump({
        "holdout_generators": ["kandinsky-inpaint", "b"],
        "holdout_manipulation": [], "holdout_domains": [],
    }), encoding="utf-8")
    cfg = PipelineConfig(
        out_dir=str(out_dir / "run"), seed=seed, backend="mock",
        stages={"d0": True, "d1": False, "d2": True, "d3": False, "d4": False,
                "grid": True, "postprocess": False, "split": True},
        scales=StageScales(d0=d0, d1_per_generator=0, d2=10, d3=0, d4=0),
        inpainters=[GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint"),
                    GeneratorSpec("kandinsky-inpaint", "kandinsky", "inpaint")],
        img2img=[GeneratorSpec("a", "diffusion", "img2img"),
                 GeneratorSpec("b", "diffusion-sdxl", "img2img"),
                 GeneratorSpec("b2", "diffusion-sdxl", "img2img")],
        resolution_groups={64: ["a"], 128: ["b", "b2"]},
        grid_per_op=d0,
        vae_rt_frac=0.25,
        split_config=str(sp),
    )
    run_pipeline(cfg)   # (i) 无 RuntimeError（泄漏机制绿；裁决3 前此行即 RuntimeError）
    rows = manifest.read_jsonl(Path(cfg.out_dir) / "manifest.jsonl")
    tb = [r for r in rows if r.split == "test_b"]
    b_rows = [r for r in rows if r.generator_name == "b"]
    kd_rows = [r for r in rows if r.generator_name == "kandinsky-inpaint"]

    # (iii) 池分离机制：test_b 内 fake 生成器名 ⊆ holdout 名集合；train 池名绝不入
    # test_b；holdout 名的行（存在则）只落 test_b
    assert {r.generator_name for r in tb if r.is_fake == 1} <= {"kandinsky-inpaint", "b"}
    assert all(r.split != "test_b" for r in rows
               if r.generator_name in {"a", "b2", "stable-diffusion-inpaint"})
    assert {r.split for r in b_rows} <= {"test_b"}
    assert {r.split for r in kd_rows} <= {"test_b"}

    # 通用不变量：含 fake@rs{r} 的 split 必有 real@rs{r}
    _res_cover_invariant(rows)

    # (ii) test_b 双侧链集合 == {rs64,rs128}：grid hold 池被哈希抽中（b 行存在）时成立
    # ——b@128 + 同底图 kandinsky outpaint@64 供 fake 侧，real 侧为兄弟行对。hold 池按
    # stable_hash(okey)%5==0 抽 ~20% 底图，小规模个别 seed 可能抽空（此时本段空泛跳过，
    # 其余断言不依赖抽中与否；sweep 汇报 firing 率，正式测试的固定参数已锁两路径均触发）。
    if b_rows:
        assert {image_io.chain_resolution(r.io_chain) for r in b_rows} == {128}
        chains_tb = manifest.stats(rows)["io_chain_by_fake_split"]["test_b"]
        real_set = {c for c, v in chains_tb.items() if v["real"] > 0}
        fake_set = {c for c, v in chains_tb.items() if v["fake"] > 0}
        assert real_set == fake_set == {"rs64>png", "rs128>png"}
    if kd_rows:
        assert {image_io.chain_resolution(r.io_chain) for r in kd_rows} == {64}

    # 镜像半边：非 holdout 的 128 组 spec "b2" 供 train 侧 rs128 fake（train 池 ~80%+
    # 底图，恒非空）
    b2_rows = [r for r in rows if r.generator_name == "b2"]
    assert b2_rows
    assert {image_io.chain_resolution(r.io_chain) for r in b2_rows} == {128}

    # (iv) 定向校验：check_all 输出不含 V8/V9/V10 前缀消息（split 完整性/holdout 防泄漏
    # 机制；允许其他 split 的 V2/V4 小 n 组合噪声存在——它们与被测机制无关）
    errs = check_all(rows, profile="run", holdout_generators={"kandinsky-inpaint", "b"})
    bad = [e for e in errs if e.startswith(("V8:", "V9:", "V10:"))]
    assert not bad, bad
    return {"grid_hold_fired": bool(b_rows), "d2_hold_fired": bool(kd_rows)}


def test_testb_holdout_covers_every_resolution_group_v2_green(tmp_path):
    """正向（真字面 B3 形态；裁决4 机制作用域断言，任意 seed/规模必过）：holdout =
    {grid 的 128 组 img2img spec "b", D2 的基准组 inpainter kandinsky-inpaint}，train =
    {a@64、b2@128、stable-diffusion-inpaint}。裁决3 池分离前该形态直接 RuntimeError
    （check_leakage 规则4，见 report「裁决执行 2/3」）。断言四条确定性性质：(i) run 无
    RuntimeError；(ii) grid hold 池抽中时 test_b 双侧非生成链集合 == {rs64,rs128}
    （D2-holdout 供 rs64、grid-holdout 供 rs128）；(iii) test_b 内 fake 生成器名 ⊆
    holdout 集合、train 池名绝不入 test_b（池分离机制）；(iv) check_all 无 V8/V9/V10
    消息——不再断言全局 ==[]（裁决4：其他 split 的 V2/V4 是小 n 组合噪声，与被测机制
    无关）。"b2" 是约束镜像半边：每个分辨率组须同时有 holdout 与非 holdout 成员
    （B3 config 推论，见 report「裁决执行 3」）。"""
    stats = _check_testb_coverage_positive(tmp_path, seed=0, d0=20)
    # 固定参数下两条 hold 路径均确定性触发（保证正式测试非空泛；任意参数的通过性由
    # 裁决4 sweep 证明，见 report）
    assert stats == {"grid_hold_fired": True, "d2_hold_fired": True}


def _check_testb_missing_group_red(out_dir, seed: int, d0: int) -> dict:
    """测试 B 主体（裁决4 参数化，任意 seed/规模必过）。返回守门信号 firing 统计。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = out_dir / "split_reviewer.yaml"
    sp.write_text(yaml.dump({
        "holdout_generators": ["kandinsky-inpaint"],
        "holdout_manipulation": [], "holdout_domains": [],
    }), encoding="utf-8")
    cfg = PipelineConfig(
        out_dir=str(out_dir / "run"), seed=seed, backend="mock",
        stages={"d0": True, "d1": False, "d2": True, "d3": False, "d4": False,
                "grid": True, "postprocess": False, "split": True},
        scales=StageScales(d0=d0, d1_per_generator=0, d2=10, d3=0, d4=0),
        inpainters=[GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint"),
                    GeneratorSpec("kandinsky-inpaint", "kandinsky", "inpaint")],
        img2img=[GeneratorSpec("a", "diffusion", "img2img"),
                 GeneratorSpec("b", "diffusion-sdxl", "img2img")],   # 均非 holdout
        resolution_groups={64: ["a"], 128: ["b"]},
        grid_per_op=d0,
        vae_rt_frac=0.25,   # 中性取值：V4 噪声不再被断言（裁决4 删隔离断言），不敏感
        split_config=str(sp),
    )
    run_pipeline(cfg)   # 不 crash：泄漏规则不涉及（kandinsky 只在 test_b，无同名 train 行）
    rows = manifest.read_jsonl(Path(cfg.out_dir) / "manifest.jsonl")

    # 非对称空池分支（裁决4 第 5 点轻断言）：holdout 侧只有 inpainter、无 holdout
    # img2img → grid hold 池底图 i2i 池为空，产 0 条 img2img + 恰 1 条 outpaint
    # （kandinsky）。kd_outp 即 grid hold 池 outpaint 行（D2 行的 operator 不会是
    # outpaint），其底图上不得有任何 img2img 行。
    kd_outp = [r for r in rows if r.operator == "outpaint"
               and r.generator_name == "kandinsky-inpaint"]
    i2i_bases = {r.base_id for r in rows if r.operator == "img2img"}
    assert all(r.base_id not in i2i_bases for r in kd_outp), "hold 池底图不得有 img2img 行"
    assert all(c == 1 for c in Counter(r.base_id for r in kd_outp).values())

    tb = [r for r in rows if r.split == "test_b"]
    if tb:   # hold 池被哈希抽中（~20%/底图；小规模个别 seed 可能抽空，sweep 汇报率）
        # 约束违反的形状：real 侧兄弟行两分辨率、fake 侧只有基准组
        assert {image_io.chain_resolution(r.io_chain)
                for r in tb if r.is_fake == 0} == {64, 128}
        assert {image_io.chain_resolution(r.io_chain)
                for r in tb if r.is_fake == 1} == {64}
        errs = check_all(rows, profile="run", holdout_generators={"kandinsky-inpaint"})
        assert any(("V2" in e) and ("test_b" in e) for e in errs), (
            f"设计约束违反必须触发含 V2 与 test_b 的失败消息，实得: {errs}")
    return {"fired": bool(tb)}


def test_testb_holdout_missing_resolution_group_v2_red_is_expected(tmp_path):
    """负向（文档化守门行为，复刻审查者场景；裁决4 机制作用域断言）：holdout 只覆盖
    基准组（kandinsky@64 走 D2 与 grid hold 池 outpaint），128 组无任何 holdout →
    test_b 的 real 侧有 {rs64,rs128} 兄弟行、fake 侧只有 rs64 → **V2 红是对的**：
    「holdout 生成器须覆盖每个参与 test_b 的分辨率组」是设计约束，违反时 V2 如实咬住
    （期望的响亮失败；修复方向是调整组配置或显式把 test_b 加入
    base_resolution_only_splits，见 configs/split.yaml 注释）。只断言该消息存在——
    不再断言错误隔离（裁决4：val/test_a 的 V2/V4 可因小 n 组合噪声独立出现，与被测
    约束无关）。顺带锁非对称空池分支：hold 侧只有 inpainter 时 hold 池底图 0 条
    img2img + 恰 1 条 outpaint。"""
    stats = _check_testb_missing_group_red(tmp_path, seed=0, d0=20)
    # 固定参数下守门信号确定性发出（保证正式测试非空泛；任意参数由 sweep 证明）
    assert stats == {"fired": True}
