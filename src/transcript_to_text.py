import argparse
import json
import os

from deepmultilingualpunctuation import PunctuationModel
from dotenv import load_dotenv

try:
    from utils import PathHelper, get_logger
except Exception as e:
    print(e)
    raise ("Please run this script from the root directory of the project")

# load env variables
dotenv_path = PathHelper.root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# logger
logger = get_logger(__name__)

# init model
logger.info("init punctuation model")
punct_model = PunctuationModel()


# process functions
def preprocess_transcript(transcript):
    """
    add punctuation to transcript and merge into one doc
    """
    # split into list of list with length n
    n = 5
    transcript_text = [transcript[i : i + n] for i in range(0, len(transcript), n)]

    # restore punctuation
    transcript_text_restore = []
    for transcript_text_i in transcript_text:
        result_punct = punct_model.restore_punctuation(" ".join(transcript_text_i))
        transcript_text_restore.append(result_punct)

    # merge transcripts
    transcript = "\n".join(transcript_text_restore)

    return transcript


def main(args):
    # select files
    entities = [i for i in os.listdir(PathHelper.entities_dir) if i.endswith(".json")]

    logger.info(f"# files: {len(entities)}")

    # text to text
    m_transcripts_processed = 0
    for jf in entities:
        fname = jf.split(".")[0]
        try:
            with open(PathHelper.entities_dir / jf, "r") as f:
                ent_i = json.load(f)

            # if file exist, then skip
            if (PathHelper.text_dir / f"{fname}.txt").exists():
                logger.info(f"file exist: {fname}")
                continue

            # if having transcript from entities, then save text
            if ent_i.get("transcript"):
                transcript_text = [t["text"] for t in ent_i["transcript"]]

                # preprocess transcript
                transcript = preprocess_transcript(transcript_text)

                # save transcript with encoding
                with open(
                    PathHelper.text_dir / f"{fname}.txt", "w", encoding="utf8"
                ) as f:
                    json.dump(transcript, f)

                m_transcripts_processed += 1

        except Exception as e:
            logger.error(e)
            continue

    # logger
    logger.info(f"extract transcript from {m_transcripts_processed} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
