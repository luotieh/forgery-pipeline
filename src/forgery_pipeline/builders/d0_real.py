"""D0 真实图像池：摄取 → 清洗 → 去重（报告 §4）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline import image_io, ids
from forgery_pipeline.backends import registry
from forgery_pipeline.dedup import PHashDeduper
from forgery_pipeline.qc.image_qc import check_image
from forgery_pipeline.schema import Sample, TaskType


def build_d0(out_dir, n: int, backend: str = "mock", seed: int = 0,
             resolutions: list[int] | None = None) -> list[Sample]:
    """resolutions=None（现状）：每张通过 QC/去重的源图产 1 行，逐字节沿用 HEAD 行为
    （regression anchor）。

    resolutions=[size, ...]（PATCH 9 Wave2 9.2c 多分辨率组摄取；pipeline 传入前已排序）：
    QC/去重只对源图做一次（不逐分辨率重复），之后对每个 size 各产一行——每组分辨率各自
    配套同链 real 行，V2（split 内 real/fake 非生成链集合相等）才在每个分辨率组内都有
    真实判据（否则某分辨率组的 real 行落单，is_fake 在该组内变得可从非生成链预测）。
    每行 image_id 用「源图 content hash + 分辨率后缀」区分（确定性，非源图本身内容);
    resolutions[0]（最小的一组，即"基准组"）那一行的 image_id 兼作全组 base_id——D2/D3/
    grid 只消费基准组底图行（见 pipeline.py），其余分辨率组只出 real（+vae_rt）行，
    fake 侧覆盖改由 grid 按分辨率组路由负责（见 builders/grid_ops.py）。

    非基准分辨率行的 `real_image_path` 显式回填为基准行的 image_path（不同于「其余同
    现状」的字面读法，是实现期间发现并修正的必要一步——见下方说明）：`split/grouping.
    origin_key()` 只做「一跳」解析（`real_image_path or image_path` 取 stem），本身不认
    `base_id` 字段。若非基准分辨率行不设 `real_image_path`（真·现状字面读法），它会以
    **自己的** image_path 形成一个独立 origin-group，与基准行（同 base_id，但不同
    origin_key）被 `assign_splits` 各自独立哈希——V8（同 base_id 须同 split）几乎必红
    （已实测复现：`build_d0(resolutions=[64,128])` 直接接 `assign_splits` 后，多数分辨率
    兄弟行组各自落到不同 split）。回填后 origin_key 与 base_id 组重新重合（同 D2/D3/
    grid 现有行一样，一跳可达基准行），修的是 split/grouping.py 未变但仍需成立的既有不
    变式，不是新增语义；vae_rt 插入行（pipeline.py）经既有 `s.model_copy(deep=True)` 直接
    继承源行已定的 `.split`（非重新过 origin_key 哈希），故不需要改动——已实测核对两分辨
    率组的 vae_rt 行均与其组一致，无需在 pipeline.py 另作改动。
    """
    out_dir = Path(out_dir)
    src = registry.get_image_source(backend, seed=seed)
    dedup = PHashDeduper()
    samples: list[Sample] = []
    accepted = 0
    for img, meta in src.iter_images(n * 3):  # 过采样以容忍 QC 丢弃
        if accepted >= n:
            break
        ok, _ = check_image(img)
        if not ok or not dedup.add(img):
            continue
        accepted += 1

        if not resolutions:
            iid = ids.make_image_id("real", ids.content_hash(img))
            rel = f"D0_real_pristine/{iid}.png"
            image_io.save_canonical(img, out_dir / rel)
            samples.append(Sample(
                image_id=iid, image_path=rel, is_fake=0,
                task_type=TaskType.real_pristine,
                source_dataset=meta.get("source_dataset"),
                license=meta.get("license"),
                sample_kind="real",
                base_id=iid,
                io_chain=image_io.chain("decode", f"rs{img.shape[0]}", "png"),
            ))
            continue

        content_hex = ids.content_hash(img).hex()
        group_base_id = None
        group_base_path = None
        for size in resolutions:
            resized = image_io.resize_square(img, size)
            iid = ids.make_image_id("real", f"{content_hex}-r{size}")
            rel = f"D0_real_pristine/{iid}.png"
            if group_base_id is None:      # resolutions[0]（首分辨率行）的 iid 作全组 base_id
                group_base_id = iid
                group_base_path = rel
            image_io.save_canonical(resized, out_dir / rel)
            samples.append(Sample(
                image_id=iid, image_path=rel, is_fake=0,
                # 非基准分辨率行回指基准行，使 origin_key 与 base_id 组重合（见上方
                # docstring）；基准行自身保持 real_image_path=None（现状：自指）。
                real_image_path=(group_base_path if rel != group_base_path else None),
                task_type=TaskType.real_pristine,
                source_dataset=meta.get("source_dataset"),
                license=meta.get("license"),
                sample_kind="real",
                base_id=group_base_id,
                io_chain=image_io.chain("decode", f"rs{size}", "png"),
            ))
    return samples
