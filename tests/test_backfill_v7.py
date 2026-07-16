import json
from forgery_pipeline import manifest
from forgery_pipeline.schema import Sample, TaskType
from scripts.backfill_manifest_v7 import backfill

def _legacy_rows(tmp_path):
    rows = [Sample(image_id="r0", image_path="a.jpg", is_fake=0,
                   task_type=TaskType.real_pristine),
            Sample(image_id="f0", image_path="b.jpg", is_fake=1, operator="inpaint",
                   mask_path="m.png", task_type=TaskType.localization,
                   manipulation_level1="partial_manipulated")]
    p = tmp_path / "old.jsonl"; manifest.write_jsonl(p, rows); return p

def test_new_fields_roundtrip(tmp_path):
    s = Sample(image_id="x", image_path="x.png", is_fake=1,
               task_type=TaskType.localization, io_chain="decode>rs256>edit:m>png",
               sample_kind="edited", compositing="paste_feather", feather_px=8,
               probe_group="compositing_pair", pair_id="p0",
               manipulation_level1="partial_manipulated", mask_path="m.png")
    p = tmp_path / "m.jsonl"; manifest.write_jsonl(p, [s])
    r = manifest.read_jsonl(p)[0]
    assert (r.io_chain, r.sample_kind, r.compositing, r.feather_px,
            r.probe_group, r.pair_id) == ("decode>rs256>edit:m>png", "edited",
                                          "paste_feather", 8, "compositing_pair", "p0")

def test_backfill_fills_legacy(tmp_path):
    p = _legacy_rows(tmp_path); out = tmp_path / "new.jsonl"
    assert backfill(p, out) == 2
    rows = manifest.read_jsonl(out)
    assert [r.sample_kind for r in rows] == ["real", "edited"]     # 按 is_fake 推断
    assert all(r.io_chain == "legacy" for r in rows)
    assert rows[1].compositing == "none" and rows[0].compositing is None  # 仅 masked 编辑行
