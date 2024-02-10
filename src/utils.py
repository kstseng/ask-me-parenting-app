import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

load_dotenv()

LOG_LEVEL = "info"

# formatter and level options
FORMATTER = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARN,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
}


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVELS[LOG_LEVEL])
    logger.addHandler(get_console_handler())
    logger.propagate = False
    return logger


class PathHelper:
    def __init__(self) -> None:
        pass

    root_dir = Path(__file__).parent.parent.absolute()
    src_dir = root_dir / "src"
    data_dir = root_dir / "data"
    entities_dir = data_dir / "entities"
    audio_dir = data_dir / "audio"
    text_dir = data_dir / "text"
    db_dir = root_dir / "db"

    # create
    data_dir.mkdir(exist_ok=True, parents=True)
    entities_dir.mkdir(exist_ok=True, parents=True)
    audio_dir.mkdir(exist_ok=True, parents=True)
    text_dir.mkdir(exist_ok=True, parents=True)
    db_dir.mkdir(exist_ok=True, parents=True)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


class MaxPoolingEmbeddings(HuggingFaceInferenceAPIEmbeddings):
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])[0]
        average_pooling = np.mean(embeddings[0], axis=0)

        return average_pooling.tolist()


def get_connection_string():
    conn_str = PGVector.connection_string_from_db_params(
        driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
        user=os.environ.get("PGVECTOR_USER", "user"),
        password=os.environ.get("PGVECTOR_PASSWORD", "pwd"),
    )

    return conn_str
