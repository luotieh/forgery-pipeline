# tests/test_resolution_groups.py —— PATCH 9 Wave 2 Task 4：多分辨率组摄取（TDD 先红）
#
# 9.2c 要求：每个生成器分辨率组（SD1.5@512 / SDXL@1024 等）各自配套同链 real + vae_rt
# 行——否则 PATCH 7 的 V2（split 内 real/fake 非生成链集合相等）在组内是空判据。
# build_d0(resolutions=[...]) 对每张通过 QC/去重的源图，按 resolutions（pipeline 传入前
# 已排序）逐 size 产一行；D2/D3/grid 只消费 resolutions[0]（基准组）底图行，其余分辨率组
# 只出 real（+vae_rt）行，fake 侧覆盖改由 grid 按 policies.resolution_groups 对 img2img
# spec 名分组路由（见 test_pipeline_multi_resolution_groups_e2e_v2_holds）。
from pathlib import Path

import yaml

from forgery_pipeline import image_io, manifest
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.config import GeneratorSpec, PipelineConfig, StageScales, load_config
from forgery_pipeline.pipeline import run_pipeline
from forgery_pipeline.split.grouping import origin_key
from forgery_pipeline.split.splitter import assign_splits
from forgery_pipeline.validate import check_all


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
# pipeline e2e：两个分辨率组各自配套 real+vae_rt+fake，V1-V10 全绿
# ---------------------------------------------------------------------------

def test_pipeline_multi_resolution_groups_e2e_v2_holds(tmp_path):
    """9.2c 核心断言（裁决执行后恢复生产口径，见 task-4-report「裁决执行」）：
    resolution_groups={64:["a"],128:["b"]} + 生产 configs/split.yaml（holdout_manipulation
    含 object_replacement/text_editing、base_resolution_only_splits: [test_c]）下，两个
    分辨率组各自都有 real 行（D0）与 fake 行（grid 按 spec 名路由），V2（split 内
    real/fake 非生成链集合相等）在两组之间同时成立；test_c 经组成规则过滤（pipeline.py，
    Test-C 测算子泛化、分辨率非其轴）后只含基准分辨率行，check_all 于 profile="run" +
    testc_holdout 生产口径全绿。"""
    cfg = PipelineConfig(
        out_dir=str(tmp_path / "run"), seed=0, backend="mock",
        stages={"d0": True, "d1": False, "d2": True, "d3": True, "d4": True,
                "grid": True, "postprocess": True, "split": True},
        scales=StageScales(d0=20, d1_per_generator=0, d2=10, d3=10, d4=3),
        inpainters=[GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")],
        img2img=[GeneratorSpec("a", "diffusion", "img2img"),
                 GeneratorSpec("b", "diffusion-sdxl", "img2img")],
        resolution_groups={64: ["a"], 128: ["b"]},
        grid_per_op=10,           # 覆盖全部 d3_bases（PATCH 6 不变式下 grid 的可用底图池）
        vae_rt_frac=0.25,
        split_config="configs/split.yaml",   # 生产配置（含 base_resolution_only_splits）
    )
    st = run_pipeline(cfg)
    assert st["total"] > 0
    rows = manifest.read_jsonl(Path(cfg.out_dir) / "manifest.jsonl")

    # real 行含两条 nongen 链（rs64 与 rs128）
    real_chains = {r.io_chain for r in rows if r.is_fake == 0 and r.sample_kind == "real"}
    assert "decode>rs64>png" in real_chains
    assert "decode>rs128>png" in real_chains

    # vae_rt 行两分辨率都出现
    vae_chains = {r.io_chain for r in rows if r.sample_kind == "real_vae_rt"}
    assert any(c and c.startswith("decode>rs64>") for c in vae_chains), vae_chains
    assert any(c and c.startswith("decode>rs128>") for c in vae_chains), vae_chains

    # io_chain_by_fake_split：至少一个 split 内 real 侧同时含 rs64/rs128 两条链
    by_split = manifest.stats(rows)["io_chain_by_fake_split"]
    assert any("rs64>png" in chains and chains["rs64>png"]["real"] > 0
              and "rs128>png" in chains and chains["rs128>png"]["real"] > 0
              for chains in by_split.values()), by_split
    # fake 侧同样两条链都要出现（grid 按分辨率组路由 img2img 的直接效果）——否则该 split
    # 内 real 集合 {rs64,rs128} ≠ fake 集合 {rs64}，V2 会红（9.2c 设计张力的关键处）。
    assert any("rs64>png" in chains and chains["rs64>png"]["fake"] > 0
              and "rs128>png" in chains and chains["rs128>png"]["fake"] > 0
              for chains in by_split.values()), by_split

    # base_id 组（含跨分辨率兄弟行）split 全一致（V8 的数据前提，端到端复核）
    by_base = {}
    for r in rows:
        if r.postprocess_of:
            continue
        by_base.setdefault(r.base_id, set()).add(r.split)
    assert all(len(s) == 1 for s in by_base.values()), by_base

    # test_c 组成规则：过滤后全部基准分辨率（Test-C 测算子泛化，分辨率非其轴）
    tc = [r for r in rows if r.split == "test_c"]
    assert tc, "前提：生产 holdout_manipulation 确实把 D2 组路由进了 test_c"
    assert all(image_io.chain_resolution(r.io_chain) == 64 for r in tc), (
        [(r.image_id, r.io_chain) for r in tc])

    # 过滤不拆散 postprocess 母子行：留存行的 postprocess_of 必须仍指向留存行
    ids_present = {r.image_id for r in rows}
    assert all(r.postprocess_of in ids_present for r in rows if r.postprocess_of)

    errs = check_all(rows, profile="run", testc_holdout="object_replacement")
    assert errs == [], f"check_all 非空: {errs[:10]}"


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

def test_testb_holdout_covers_every_resolution_group_v2_green(tmp_path):
    """正向（B3 形态）：两个分辨率组各配一个 holdout 生成器——64 组 kandinsky-inpaint 走
    D2 @基准、128 组 "b" 走 grid 按组路由——test_b 两侧非生成链集合相等，check_all(run)
    全绿；并按裁决第 5 点核对 grid 的 holdout img2img 行确实落 test_b（_split_for 标注 +
    assign_splits 组路由同向）。

    形态注记（如实记录，见 report「裁决执行 2」）：inpainter 池全 holdout（仅 kandinsky）
    是当前代码下唯一可构造的形态——grid 对每个底图无条件套用全部 img2img specs（无
    holdout 池分离，W2T3 报告已记录的 deferred 设计缺口），holdout img2img "b" 会把全部
    grid 底图组拖进 test_b，任何同组共存的非 holdout 生成器名（outpaint 池 = D2 的
    pool_train，两者结构性同集合）只要同时出现在 train 就触发 check_leakage 规则4。已实
    测复现：inpainters 含非 holdout 的 stable-diffusion-inpaint 时 run_pipeline 直接
    RuntimeError「cross-generator 生成器出现在 train: ['stable-diffusion-inpaint']」。
    B3 真实配置（sd15 非 holdout + sdxl holdout img2img 并存）启用前须先裁决 grid 的
    img2img holdout 池分离——本测试锁定的是约束满足时 test_b 的双侧链集合语义。"""
    sp = tmp_path / "split_b3.yaml"
    sp.write_text(yaml.dump({
        "holdout_generators": ["kandinsky-inpaint", "b"],
        "holdout_manipulation": [], "holdout_domains": [],
    }), encoding="utf-8")
    cfg = PipelineConfig(
        out_dir=str(tmp_path / "run"), seed=0, backend="mock",
        stages={"d0": True, "d1": False, "d2": True, "d3": True, "d4": False,
                "grid": True, "postprocess": True, "split": True},
        scales=StageScales(d0=20, d1_per_generator=0, d2=6, d3=6, d4=0),
        inpainters=[GeneratorSpec("kandinsky-inpaint", "kandinsky", "inpaint")],
        img2img=[GeneratorSpec("a", "diffusion", "img2img"),
                 GeneratorSpec("b", "diffusion-sdxl", "img2img")],
        resolution_groups={64: ["a"], 128: ["b"]},
        grid_per_op=10,     # 覆盖全部 d3_bases：D3 的 manual-web-edit 随组整体进 test_b，
                            # 不在 train 留同名行（否则 check_leakage 规则4）
        vae_rt_frac=0.0,
        split_config=str(sp),
    )
    run_pipeline(cfg)
    rows = manifest.read_jsonl(Path(cfg.out_dir) / "manifest.jsonl")

    # 裁决第 5 点：holdout img2img "b" 的行全部落 test_b，且经按组路由生成在 128
    b_rows = [r for r in rows if r.generator_name == "b"]
    assert b_rows
    assert {r.split for r in b_rows} == {"test_b"}
    assert {image_io.chain_resolution(r.io_chain) for r in b_rows} == {128}

    # test_b 两侧非生成链集合相等，且恰为两个分辨率组
    chains_tb = manifest.stats(rows)["io_chain_by_fake_split"]["test_b"]
    real_set = {c for c, v in chains_tb.items() if v["real"] > 0}
    fake_set = {c for c, v in chains_tb.items() if v["fake"] > 0}
    assert real_set == fake_set == {"rs64>png", "rs128>png"}

    errs = check_all(rows, profile="run", holdout_generators={"kandinsky-inpaint", "b"})
    assert errs == [], f"check_all 非空: {errs[:10]}"


def test_testb_holdout_missing_resolution_group_v2_red_is_expected(tmp_path):
    """负向（文档化守门行为，复刻审查者场景）：holdout 只覆盖基准组（kandinsky@64 走
    D2），128 组无任何 holdout → test_b 的 real 侧有 {rs64,rs128} 兄弟行、fake 侧只有
    rs64 → **V2 红是对的**：「holdout 生成器须覆盖每个参与 test_b 的分辨率组」是设计
    约束，违反时 V2 如实咬住（期望的响亮失败，而非需要修复的 bug；修复方向是调整组
    配置或显式把 test_b 加入 base_resolution_only_splits，见 configs/split.yaml 注释）。
    本测试断言这声失败可靠发出——若未来有人改动让它静默通过，等于拆掉守门。"""
    sp = tmp_path / "split_reviewer.yaml"
    sp.write_text(yaml.dump({
        "holdout_generators": ["kandinsky-inpaint"],
        "holdout_manipulation": [], "holdout_domains": [],
    }), encoding="utf-8")
    cfg = PipelineConfig(
        out_dir=str(tmp_path / "run"), seed=0, backend="mock",
        stages={"d0": True, "d1": False, "d2": True, "d3": False, "d4": False,
                "grid": True, "postprocess": False, "split": True},
        scales=StageScales(d0=20, d1_per_generator=0, d2=10, d3=0, d4=0),
        inpainters=[GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint"),
                    GeneratorSpec("kandinsky-inpaint", "kandinsky", "inpaint")],
        img2img=[GeneratorSpec("a", "diffusion", "img2img"),
                 GeneratorSpec("b", "diffusion-sdxl", "img2img")],   # 均非 holdout
        resolution_groups={64: ["a"], 128: ["b"]},
        grid_per_op=10,
        vae_rt_frac=0.15,   # 让 V4 绿（0.25 实测 8/22=0.36 出带），把红隔离到 V2/test_b
        split_config=str(sp),
    )
    run_pipeline(cfg)   # 不 crash：泄漏规则不涉及（kandinsky 只在 test_b，无同名 train 行）
    rows = manifest.read_jsonl(Path(cfg.out_dir) / "manifest.jsonl")

    tb = [r for r in rows if r.split == "test_b"]
    assert {image_io.chain_resolution(r.io_chain) for r in tb if r.is_fake == 0} == {64, 128}
    assert {image_io.chain_resolution(r.io_chain) for r in tb if r.is_fake == 1} == {64}

    errs = check_all(rows, profile="run", holdout_generators={"kandinsky-inpaint"})
    assert any(("V2" in e) and ("test_b" in e) for e in errs), (
        f"设计约束违反必须触发含 V2 与 test_b 的失败消息，实得: {errs}")
    # 隔离性：V2 的红只在 test_b（其余 split 该约束未被违反；不对非 V2 检查器做排他断言，
    # 避免未来 V11/V12 等新增校验器误伤本测试）
    assert all("test_b" in e for e in errs if e.startswith("V2:")), errs
