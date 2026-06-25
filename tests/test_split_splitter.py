from forgery_pipeline.schema import Sample, TaskType, Postprocess
from forgery_pipeline.split.splitter import assign_splits, SPLITS
from forgery_pipeline.split.leakage import check_leakage


def _real(i, ds="COCO"):
    return Sample(image_id=f"real_{i}", image_path=f"D0/real_{i}.jpg",
                  is_fake=0, task_type=TaskType.real_pristine, source_dataset=ds)


def _fake(i, real_i, gen="sd", manip="object_replacement", ds="COCO", pp=None):
    return Sample(image_id=f"f_{i}", image_path=f"D2/f_{i}.jpg",
                  real_image_path=f"D0/real_{real_i}.jpg",
                  mask_path=f"D2/m/f_{i}.png", is_fake=1,
                  task_type=TaskType.localization,
                  manipulation_level1="partial_manipulated",
                  manipulation_level2="AIGC-editing", manipulation_level3=manip,
                  generator_name=gen, source_dataset=ds, seed=i,
                  postprocess=pp or Postprocess())


def test_holdout_routing():
    s_b = _fake(1, 1, gen="ideogram")
    s_c = _fake(2, 2, manip="text_editing")
    s_d = _real(3, ds="Places")
    assign_splits([s_b, s_c, s_d], holdout_generators=["ideogram"],
                  holdout_manipulation=["text_editing"], holdout_domains=["Places"])
    assert s_b.split == "test_b"
    assert s_c.split == "test_c"
    assert s_d.split == "test_d"


def test_degraded_testa_becomes_teste():
    samples = [_fake(i, i, pp=Postprocess(jpeg_quality=70)) for i in range(40)]
    assign_splits(samples, holdout_generators=[], holdout_manipulation=[],
                  holdout_domains=[])
    assert any(s.split == "test_e" for s in samples)
    assert all(s.split in SPLITS for s in samples)


def test_no_leakage_after_split():
    samples = [_real(i) for i in range(20)] + [_fake(i, i) for i in range(20, 40)]
    assign_splits(samples, holdout_generators=["ideogram"],
                  holdout_manipulation=["text_editing"], holdout_domains=["Places"])
    assert check_leakage(samples) == []
