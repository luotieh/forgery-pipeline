from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.d2_local import build_d2
from forgery_pipeline.builders.d4_explain import build_d4


def test_build_d4_explanations(tmp_path):
    base = build_d0(tmp_path, n=6, seed=0)
    inps = [GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")]
    d2 = build_d2(tmp_path, base, n=4, inpainters=inps, seed=0)
    d4 = build_d4(tmp_path, d2, n=3)
    assert len(d4) == 3
    for s in d4:
        assert s.task_type.value == "explainable"
        assert s.explanation is not None
        assert s.explanation.forensic_conclusion
        assert s.mask_path is not None
