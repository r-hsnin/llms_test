# llama-cpp-pythonでローカルモデルを動作するテスト
# 通常のpredictとllama-indexのクエリを検証

from llm import LLM, LLAMA_INDEX

# llm = LLM()
# llm.predict(prompt="結束バンドは江戸時代に結成された？")


llm = LLM().llm
llama_index = LLAMA_INDEX(llm,)

llama_index.save_storage_context('bocchi.txt')
print(llama_index.query("結束バンドは江戸時代に結成された？"))