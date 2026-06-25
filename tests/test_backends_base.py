import pytest
from forgery_pipeline.backends import base


def test_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        base.ImageSource()


def test_subclass_must_implement():
    class Bad(base.Segmenter):
        pass
    with pytest.raises(TypeError):
        Bad()

    class Good(base.Segmenter):
        def propose_masks(self, image, k):
            return []
    assert Good().propose_masks(None, 0) == []
