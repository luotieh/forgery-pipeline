import pytest
from forgery_pipeline.backends import registry, mock


def test_mock_resolves():
    assert isinstance(registry.get_image_source("mock"), mock.MockImageSource)
    assert isinstance(registry.get_segmenter("mock"), mock.MockSegmenter)
    g = registry.get_whole_generator("mock", "sdxl", "diffusion")
    assert g.name == "sdxl"


def test_real_backend_raises_with_hint():
    with pytest.raises(NotImplementedError) as ei:
        registry.get_whole_generator("real:diffusers", "sdxl", "diffusion")
    assert "pip install" in str(ei.value)
