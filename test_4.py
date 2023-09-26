# 任意のAPIを使用してllmクラスを作成して、llama-indexで使用する検証

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests

from llm import LLAMA_INDEX

class CustomLLM(LLM):
    max_tokens: int
    
    
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        import json
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post('http://localhost:8000/v1/completions', headers=headers, json={'prompt': prompt, 'max_tokens': self.max_tokens, 'temperature': 0.3, 'top_p': 0.8})
        return json.loads(response.text)['choices'][0]['text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {'max_tokens': self.max_tokens}


llm = CustomLLM(max_tokens=256)

llama_index = LLAMA_INDEX(llm=llm)

# llama_index.save_storage_context(INPUT_TEXT)
response = llama_index.query("結束バンドは江戸時代に結成された？")

# prompt = '''
# あなたは優秀で誠実な日本人のアシスタントです。

# USER: AWS Glue Databrewについて教えてください。
# ASSISTANT:
# '''

# print(llm(prompt))