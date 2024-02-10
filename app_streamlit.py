"""
ref:
https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/chat_with_documents.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.fake import FakeMessagesListChatModel
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import AIMessage
from langchain.vectorstores.chroma import Chroma

import src.constants as const
from src.utils import MaxPoolingEmbeddings, PathHelper, get_logger

# logger
logger = get_logger(__name__)

# load env variables
dotenv_path = PathHelper.root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)
encoding_model_name = const.ENCODING_MODEL_NAME

st.set_page_config(page_title="Ask Me Parenting", page_icon="")
st.title("👪 Ask me Parenting Questions!")

# create a streamlit toggle buttom for switching between testing and production
if st.sidebar.toggle(
    "Testing Mode",
    value=True,
    help="Toggle between calling the OpenAI API or testing locally",
):
    st.write("In Testing Mode!")
    # testing
    msg = AIMessage(content="docs,clean_code")
    llm = FakeMessagesListChatModel(responses=[msg])
else:
    st.write("In Production Mode!")
    # Setup LLM and QA chain
    llm = ChatOpenAI(
        model_name=const.CHAT_GPT_MODEL_NAME,
        temperature=0,
        streaming=True,
    )


@st.cache_resource(ttl=24 * 3600)
def configure_retriever():
    embeddings = MaxPoolingEmbeddings(
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_name=encoding_model_name,
    )
    vectordb = Chroma(
        persist_directory=str(PathHelper.db_dir / const.CHROMA_DB),
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": const.N_DOCS}
    )

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            # source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {doc.page_content}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer",
)

# use PromptTemplate to generate prompts
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    verbose=True,
    # condense_question_prompt=PromptTemplate.from_template(const.PROMPT),
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message(const.INIT_MESSAGE)

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(
            user_query, callbacks=[retrieval_handler, stream_handler]
        )
