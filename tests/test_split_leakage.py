from forgery_pipeline.schema import Sample, TaskType
from forgery_pipeline.split.leakage import check_leakage


def _fake(iid, real, split, gen="sd", prompt="p", seed=1):
    return Sample(image_id=iid, image_path=f"D2/{iid}.jpg", real_image_path=real,
                  mask_path=f"D2/m/{iid}.png", is_fake=1,
                  task_type=TaskType.localization,
                  manipulation_level1="partial_manipulated",
                  manipulation_level2="AIGC-editing",
                  manipulation_level3="object_replacement",
                  generator_name=gen, prompt=prompt, seed=seed, split=split)


def test_clean_split_has_no_leak():
    a = _fake("a", "D0/real_1.jpg", "train", seed=1)
    b = _fake("b", "D0/real_2.jpg", "test_a", seed=2)
    assert check_leakage([a, b]) == []


def test_same_origin_train_and_test_flagged():
    a = _fake("a", "D0/real_1.jpg", "train", seed=1)
    b = _fake("b", "D0/real_1.jpg", "test_a", seed=2)  # 同原图跨 train/test
    errs = check_leakage([a, b])
    assert any("原图" in e for e in errs)


def test_cross_generator_generator_in_train_flagged():
    a = _fake("a", "D0/r1.jpg", "train", gen="ideogram", seed=1)
    b = _fake("b", "D0/r2.jpg", "test_b", gen="ideogram", seed=2)
    errs = check_leakage([a, b])
    assert any("生成器" in e for e in errs)
