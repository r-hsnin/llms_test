# llama-cpp-pythonのhttpサーバー機能を使用するテスト
# httpサーバー機能のAPIをllmクラスで取得して、llama-indexで使用する

from llm import LLM, LLAMA_INDEX, get_openai_llm

# インデックスの読み込みに必要なパラメータ
# INPUT_TEXT = 'bocchi.txt'

llm = get_openai_llm(
    api_base='http://localhost:8000/v1',
    api_key='',
    model='text-davinci-003',
    model_name='completions'
)

llama_index = LLAMA_INDEX(llm=llm)

# llama_index.save_storage_context(INPUT_TEXT)
response = llama_index.query("結束バンドは江戸時代に結成された？")



