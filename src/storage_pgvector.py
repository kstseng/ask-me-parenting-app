import argparse
import os

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores.pgvector import PGVector
from tqdm import tqdm

try:
    import constants as const
    from utils import PathHelper, get_connection_string, get_logger, timeit
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


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    if not string:
        return 0
    # Returns the number of tokens in a text string
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


@timeit
def main(args):
    # init embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=100, length_function=len
    )

    # load data to dataframe
    videos = os.listdir(PathHelper.text_dir)
    video_ids = [v.split(".")[0] for v in videos]
    df_videos = pd.DataFrame({"video_id": video_ids})
    new_list = []

    # Create a new list by splitting up text into token sizes of around 512 tokens
    for _, row in df_videos.iterrows():
        video_id = row["video_id"]
        try:
            with open(PathHelper.text_dir / f"{video_id}.txt", encoding="utf8") as f:
                transcript = f.readlines()
                text = eval(transcript[0])
            token_len = num_tokens_from_string(text)
            if token_len <= 512:
                new_list.append([video_id, text])
            else:
                # split text into chunks using text splitter
                split_text = text_splitter.split_text(text)
                for j in range(len(split_text)):
                    new_list.append([video_id, split_text[j]])
        except Exception as e:
            logger.error(e)

    df_new = pd.DataFrame(new_list, columns=["video_id", "content"])

    # create docs
    loader = DataFrameLoader(df_new, page_content_column="content")
    docs = loader.load()

    if args.db == "pgvector":
        db = PGVector(
            collection_name=const.COLLECTION_NAME,
            connection_string=get_connection_string(),
            embedding_function=embeddings,
        )

        # add documents
        for doc in tqdm(docs, total=len(docs)):
            db.add_documents([doc])
        logger.info("done loading from docs")

    elif args.db == "chroma":
        page_contents = [doc.page_content for doc in docs]
        db = Chroma.from_texts(
            page_contents,
            embeddings,
            persist_directory=str(PathHelper.db_dir / const.CHROMA_DB),
        )
    else:
        raise ValueError(f"db: {args.db} not supported")

    return db


if __name__ == "__main__":
    # calculate how many seconds to run
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        type=str,
        default="pgvector",
    )

    args = parser.parse_args()
    main(args)
