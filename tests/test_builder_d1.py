from pathlib import Path
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.d1_whole import build_d1


def test_build_d1_multi_generator(tmp_path):
    gens = [GeneratorSpec("stable-diffusion-xl", "diffusion", "txt2img"),
            GeneratorSpec("stylegan2", "GAN", "unconditional")]
    samples = build_d1(tmp_path, gens, per_generator=2, seed=0)
    assert len(samples) == 4
    fams = {s.generator_family for s in samples}
    assert fams == {"diffusion", "GAN"}
    for s in samples:
        assert s.is_fake == 1
        assert s.manipulation_level1 == "whole_generated"
        assert s.manipulation_level2 in {"diffusion", "GAN", "autoregressive"}
        assert (Path(tmp_path) / s.image_path).exists()
        assert s.prompt and s.seed is not None
