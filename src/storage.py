import os

from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

try:
    import constants as const
    from utils import PathHelper, get_logger, timeit
except Exception as e:
    print(e)
    raise ("Please run this script from the root directory of the project")

# logger
logger = get_logger(__name__)

# load env variables
dotenv_path = PathHelper.root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# model name for creating embeddings
model_name = const.ENCODING_MODEL_NAME


# get text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# creating embeddings for chunks of text
def get_vectorstore(text_chunks, model_name):
    logger.info("creating vectorstore")
    logger.info(f"model_name: {model_name}")

    # get embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # create vectorstore
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore = Chroma.from_texts(
        text_chunks,
        embeddings,
        persist_directory=str(PathHelper.db_dir / const.CHROMA_DB),
    )
    return vectorstore


@timeit
def main():
    # load data
    videos = os.listdir(PathHelper.text_dir)
    m_total_docs = len(videos)

    text_chunks = []
    m_success_transcript = 0
    m_failed_docs = 0
    for v in videos:
        try:
            with open(PathHelper.text_dir / v, encoding="utf8") as f:
                transcript = f.readlines()
                transcript = eval(transcript[0])

            # remove non-utf8 characters
            transcript = transcript.encode("utf-8", "ignore").decode("utf-8")

            if transcript:
                text_chunks_i = get_text_chunks(transcript)
                text_chunks.extend(text_chunks_i)
                m_success_transcript += 1
        except Exception as e:
            logger.error(e)
            m_failed_docs += 1

    logger.info(f"total docs: {m_total_docs}")
    logger.info(f"success transcript: {m_success_transcript}")
    logger.info(f"failed docs: {m_failed_docs}")

    # create vector store
    vectorstore = get_vectorstore(text_chunks, model_name=model_name)

    # Creating a database for similarity search and
    # retrieving the top 5 most similar vectors from the vector index.
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": const.N_DOCS}
    )
    logger.info(f"retriever created: {retriever}")


if __name__ == "__main__":
    # calculate how many seconds to run
    main()
