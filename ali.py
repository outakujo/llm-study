from dashscope import Generation
from http import HTTPStatus
from typing import Iterator

key = ""


def stream(
    prompt: str,
    model: str = "qwen-turbo",
) -> Iterator[str]:
    resps = Generation.call(model=model, prompt=prompt, stream=True, api_key=key)
    for r in resps:
        if r.status_code == HTTPStatus.OK:
            text = r.output.text
            yield text
        else:
            yield f"失败: {r.message}"
