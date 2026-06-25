from forgery_pipeline.schema import Sample, TaskType
from forgery_pipeline import manifest


def _real(i):
    return Sample(image_id=f"real_{i}", image_path=f"D0/real_{i}.jpg",
                  is_fake=0, task_type=TaskType.real_pristine,
                  generator_family=None)


def _fake(i):
    return Sample(image_id=f"gen_{i}", image_path=f"D1/gen_{i}.png", is_fake=1,
                  task_type=TaskType.whole_image_detection,
                  manipulation_level1="whole_generated",
                  manipulation_level2="diffusion", generator_family="diffusion")


def test_write_read_roundtrip(tmp_path):
    p = tmp_path / "d0.jsonl"
    n = manifest.write_jsonl(p, [_real(0), _real(1)])
    assert n == 2
    got = manifest.read_jsonl(p)
    assert [s.image_id for s in got] == ["real_0", "real_1"]


def test_append_and_merge(tmp_path):
    p0, p1 = tmp_path / "d0.jsonl", tmp_path / "d1.jsonl"
    manifest.write_jsonl(p0, [_real(0)])
    manifest.write_jsonl(p1, [_fake(0)])
    out = tmp_path / "manifest.jsonl"
    n = manifest.merge([p0, p1], out)
    assert n == 2
    s = manifest.stats(manifest.read_jsonl(out))
    assert s["total"] == 2 and s["fake"] == 1 and s["real"] == 1


def test_stats_counts():
    s = manifest.stats([_real(0), _fake(0), _fake(1)])
    assert s["by_task_type"]["whole_image_detection"] == 2
    assert s["by_generator_family"]["diffusion"] == 2
