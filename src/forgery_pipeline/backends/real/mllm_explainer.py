"""真实 MLLM 解释适配器骨架（OpenAI/Anthropic）。需 `pip install .[mllm]` 与 API key。"""
from __future__ import annotations
from forgery_pipeline.backends import base
from forgery_pipeline.schema import Explanation


class MLLMExplainer(base.Explainer):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "未安装 openai：`pip install .[mllm]`。") from e
        raise NotImplementedError("参考骨架：调用 MLLM 并实现 explain()。")

    def explain(self, image, mask, context) -> Explanation:
        raise NotImplementedError
