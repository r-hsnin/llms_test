import logging
import sys

from langchain.llms import LlamaCpp, OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from llama_index import load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex, StorageContext, ServiceContext, LangchainEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore

# ログレベルの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)


# langchainのOpenAIクラスを返します
def get_openai_llm(api_base, api_key, model, model_name):
    import openai
    openai.api_base = api_base
    openai.api_key = api_key
    return OpenAI(
        model=model,
        model_name=model_name
    )

# LLMパラメータ
N_GPU_LAYERS = 32
N_BATCH = 512
N_CTX = 2048
MAX_TOKENS = 2048
# LLMモデルパス
LLM_MODEL_PATH = "../llms/xwin-lm-7b-v0.1.Q4_K_M.gguf"

class LLM:
    def __init__(
        self,
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=N_BATCH,
        n_ctx=N_CTX,
        max_tokens=MAX_TOKENS,
        temperature=0.7
        ):
        
        # 引数が渡された場合は上書き
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=True,
            callback_manager=callback_manager
        )
        self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    def predict(self, prompt):
        
        prompt = f'''
        あなたは優秀で誠実な日本人のアシスタントです。
        
        USER: {prompt}
        ASSISTANT:
        '''
        
        return self.llm.predict(prompt,)

    def save_storage_context(self, text_file_path, storage_context_path):
        # ドキュメントの読み込み
        documents = SimpleDirectoryReader(
            input_files=[text_file_path]
        ).load_data()
        # ノードパーサーの作成
        text_splitter = SentenceSplitter(
            chunk_size=500,
            paragraph_separator="\n\n",
        )
        node_parser = SimpleNodeParser.from_defaults(
            text_splitter=text_splitter
        )
        # サービスコンテキストの作成
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
            node_parser=node_parser,
        )
        # インデックスの作成と保存
        vector_store_index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context
        )
        storage_context = vector_store_index.storage_context
        storage_context.persist(persist_dir=storage_context_path)
    
    def get_vector_store_index(self, storage_context_path):
        # モデルを指定したサービスコンテキストを作成する
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model
        )
        # ストレージコンテキストを作成し、ドキュメントストア、ベクトルストア、インデックスストアを設定する
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=storage_context_path),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=storage_context_path),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=storage_context_path),
        )
        return load_index_from_storage(
            storage_context=storage_context,
            service_context=service_context
        )

    def query(self, qa_template_text, prompt):
        QA_TEMPLATE = QuestionAnswerPrompt(qa_template_text)

        vector_store_index = self.load_vector_store_index(self.storage_context_path)
        query_engine = vector_store_index.as_query_engine(
            similarity_top_k=3,
            text_qa_template=QA_TEMPLATE,
        )

        return query_engine.query(prompt)


# 埋め込みモデルのパス
EMBEDDING_MODEL = "../llms/multilingual-e5-large"
# 埋め込みデータの保存領域
STORAGE_CONTEXT_PATH = "./storage_context"
# QAテンプレート
QA_TEMPLATE_TEXT = '''あなたは優秀で、誠実な日本人のアシスタントです。
以下に、コンテキスト情報を提供します。
この情報を踏まえて、ユーザーからの質問に回答してください。
コンテキスト情報を直接、回答に含めることは避けてください。
---------------------
{context_str}
---------------------

USER: {query_str}
ASSISTANT:'''

class LLAMA_INDEX():
    def __init__(
        self,
        llm: LlamaCpp,
        embed_model: str=EMBEDDING_MODEL,
        storage_context_path: str=STORAGE_CONTEXT_PATH,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.storage_context_path = storage_context_path
        
    def save_storage_context(self, text_file_path):
        # ドキュメントの読み込み
        documents = SimpleDirectoryReader(
            input_files=[text_file_path]
        ).load_data()
        # ノードパーサーの作成
        text_splitter = SentenceSplitter(
            chunk_size=500,
            paragraph_separator="\n\n",
        )
        node_parser = SimpleNodeParser.from_defaults(
            text_splitter=text_splitter
        )
        # サービスコンテキストの作成
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
            node_parser=node_parser,
        )
        # インデックスの作成と保存
        vector_store_index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context
        )
        storage_context = vector_store_index.storage_context
        storage_context.persist(persist_dir=self.storage_context_path)

    def load_vector_store_index(self, storage_context_path):
        # ストレージコンテキストの作成
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=storage_context_path),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=storage_context_path),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=storage_context_path),
        )
        # サービスコンテキストの作成
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=self.embed_model)
        )
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=embed_model
        )
        # vector_store_indexの作成
        vector_store_index = load_index_from_storage(
            storage_context=storage_context,
            service_context=service_context
        )
        return vector_store_index
    
    def query(self, prompt):
        QA_TEMPLATE = QuestionAnswerPrompt(QA_TEMPLATE_TEXT)

        vector_store_index = self.load_vector_store_index(self.storage_context_path)
        query_engine = vector_store_index.as_query_engine(
            similarity_top_k=3,
            text_qa_template=QA_TEMPLATE,
        )

        return query_engine.query(prompt)



