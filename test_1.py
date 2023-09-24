from llm import LLM, LLAMA_INDEX

llm = LLM()
llama_index = LLAMA_INDEX(llm.llm)

print(llama_index.query("結束バンドは江戸時代に結成された？"))