import openai

openai.api_base = 'http://localhost:8000/v1'

# プロンプトの設定
prompt = '''
あなたは優秀で誠実な日本人のアシスタントです。

USER: AWS Glue Databrewについて教えてください。
ASSISTANT:
'''

# APIリクエストの設定
response = openai.Completion.create(
    model="completions",
    prompt=prompt,
    max_tokens=500,
    temperature=0.7,
    top_p=0.9,
    stream=True
)

# responseの内容をストリーミングで出力する
for chunk in response:
    content=chunk["choices"][0].get("text")
    if content is not None:
      print(content, end='')
