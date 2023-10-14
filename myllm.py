from langchain.llms.base import LLM
from typing import Optional, List, Any, Iterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.output import GenerationChunk
from langchain.schema import LLMResult, Generation
import time
from langchain.callbacks.base import BaseCallbackHandler
from queue import Queue
from enum import Enum


class MyLLM(LLM):
    model: str = "my-llm"
    en_stream = False

    def __init__(self, stream=False):
        super().__init__()
        self.en_stream = stream

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        # todo your code
        text = f"stream: {prompt}"
        for i in range(10):
            text = f"{text},{i}"
            time.sleep(0.5)
            chunk = GenerationChunk(
                text=text,
            )
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(token=text, chunk=chunk)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        # todo your code
        return f"call:{prompt}"

    @property
    def _llm_type(self) -> str:
        return self.model

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            if self.en_stream:
                generation: Optional[GenerationChunk] = None
                for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                    generation = chunk
                generations.append(generation)
            else:
                text = self._call(prompt, stop, run_manager, **kwargs)
                generations.append(Generation(text=text))
        return LLMResult(generations=[generations])


class PrintCall(BaseCallbackHandler):
    q = None
    en_print = True

    class Flag(Enum):
        End = 1

    def __init__(self, print=True):
        super().__init__()
        self.en_print = print
        self.q = Queue()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)
        if self.en_print:
            print("\r", token, end="", flush=True)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.q.put(self.Flag.End)
        if self.en_print:
            print("\n")

    def next(self):
        t = self.q.get()
        return t
