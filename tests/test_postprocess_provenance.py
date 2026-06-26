import numpy as np
from forgery_pipeline.schema import Sample, TaskType
from forgery_pipeline.pipeline import apply_postprocess
from forgery_pipeline import image_io


def test_degradation_keeps_original_and_links(tmp_path):
    img = (np.random.default_rng(0).integers(0, 256, (64, 64, 3))).astype(np.uint8)
    rel = "D2/x.jpg"; image_io.save_image(img, tmp_path / rel)
    orig = Sample(image_id="x", image_path=rel, real_image_path="D0/r.jpg",
                  mask_path="D2/m/x.png", is_fake=1, task_type=TaskType.localization,
                  manipulation_level1="partial_manipulated",
                  manipulation_level2="AIGC-editing",
                  manipulation_level3="object_replacement")
    deg = apply_postprocess(tmp_path, [orig], prob=1.0, seed=1)
    assert (tmp_path / rel).exists()                      # 原图仍在
    assert len(deg) == 1 and deg[0].postprocess_of == "x" # 退化样本回链
    assert deg[0].image_path != rel                       # 退化版独立成文件
