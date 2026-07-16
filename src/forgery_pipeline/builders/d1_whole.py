"""D1 整图 AIGC 生成：强调生成器多样性（报告 §5，Community Forensics）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline import image_io, ids
from forgery_pipeline.backends import registry
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.qc.gen_qc import check_generation
from forgery_pipeline.schema import Sample, TaskType

PROMPTS = [
    "A realistic photo of a dog running on the beach at sunset.",
    "A news-style photo of a crowded street after heavy rain.",
    "A product photography image of a black backpack on a white background.",
    "A portrait of a smiling person in soft natural light.",
    "A landscape photo of mountains under a clear blue sky.",
]
_FAMILY_TO_L2 = {"diffusion": "diffusion", "GAN": "GAN",
                 "autoregressive": "autoregressive"}


def build_d1(out_dir, generators: list[GeneratorSpec], per_generator: int,
             backend: str = "mock", seed: int = 0) -> list[Sample]:
    out_dir = Path(out_dir)
    samples: list[Sample] = []
    for gi, gen in enumerate(generators):
        g = registry.get_whole_generator(backend, gen.name, gen.family)
        for j in range(per_generator):
            s = seed + gi * 1000 + j
            prompt = PROMPTS[(gi + j) % len(PROMPTS)]
            img, meta = g.generate(prompt, {"seed": s})
            ok, _, bucket = check_generation(img, prompt)
            if not ok:
                continue
            iid = ids.make_image_id("whole_gen", f"{gen.name}-{s}-{prompt}")
            rel = f"D1_whole_generated/{iid}.png"
            image_io.save_canonical(img, out_dir / rel)
            samples.append(Sample(
                image_id=iid, image_path=rel, is_fake=1,
                task_type=TaskType.whole_image_detection,
                manipulation_level1="whole_generated",
                manipulation_level2=_FAMILY_TO_L2.get(gen.family, "diffusion"),
                manipulation_level4=gen.name,
                generator_name=gen.name, generator_family=gen.family,
                prompt=prompt, seed=meta["seed"], sampler=meta["sampler"],
                steps=meta["steps"], cfg_scale=meta["cfg_scale"],
                quality_bucket=bucket,
                sample_kind="edited",
                io_chain=image_io.chain(f"gen:{gen.name}", f"rs{img.shape[0]}", "png"),
            ))
    return samples
