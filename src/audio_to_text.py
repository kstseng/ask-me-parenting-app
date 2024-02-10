import argparse
import json
import os

from deepmultilingualpunctuation import PunctuationModel
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pydub import AudioSegment
from tqdm import tqdm

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


def main(args):
    channel_name = args.channel_name
    logger.info(f"channel_name: {channel_name}")

    # init model
    punct_model = PunctuationModel()

    # select files
    fnames = [i for i in os.listdir(PathHelper.audio_dir) if i.endswith(".mp3")]
    fnames_has_text = [i for i in os.listdir(PathHelper.text_dir) if i.endswith(".txt")]
    fnames_wo_text = list(set(fnames) - set(fnames_has_text))
    logger.info(f"# files has text (all): {len(fnames_has_text)}")
    logger.info(f"# files without text (all): {len(fnames_wo_text)}")

    # select a subset of files
    json_files = os.listdir(PathHelper.entities_dir)
    entities_selected = []
    for jf in json_files:
        fname = jf.split(".")[0]
        try:
            with open(PathHelper.entities_dir / jf, "r") as f:
                ent_i = json.load(f)

            # if having transcript from entities, then save text
            if ent_i.get("transcript"):
                transcript_text = [t["text"] for t in ent_i["transcript"]]

                # split into list of list with length 50
                transcript_text = [
                    transcript_text[i : i + 50]
                    for i in range(0, len(transcript_text), 50)
                ]

                # restore punctuation
                transcript_text_restore = []
                for transcript_text_i in transcript_text:
                    result_punct = punct_model.restore_punctuation(
                        " ".join(transcript_text_i)
                    )
                    transcript_text_restore.append(result_punct)

                # merge transcripts
                transcript = "\n".join(transcript_text_restore)

                # save transcript with encoding
                with open(
                    PathHelper.text_dir / f"{fname}.txt", "w", encoding="utf8"
                ) as f:
                    json.dump(transcript, f)

            # if it's selected channel, then add to entities_selected
            if channel_name:
                if ent_i.get("channel_name") == channel_name:
                    entities_selected.append(f"{ent_i['video_id']}.mp3")
            else:
                entities_selected.append(f"{ent_i['video_id']}.mp3")

        except Exception as e:
            logger.error(e)
            continue

    # logger
    fnames_selected = set(fnames_wo_text).intersection(set(entities_selected))
    logger.info(f"# files w/o text: {len(fnames_selected)}")

    if args.limit > 0:
        logger.info("limiting the size of the subset of files to be processed")
        fnames_selected = list(fnames_selected)[: args.limit]

    logger.info(f"# files selected: {len(fnames_selected)}")

    # audio to text
    # use faster whisper
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    total_sec = 0
    for fname_ext in tqdm(fnames_selected):
        fname = fname_ext.split(".")[0]
        audio_file = AudioSegment.from_mp3(PathHelper.audio_dir / fname_ext)
        # logger.info(f"fname: {fname}")

        # get total seconds
        total_sec += audio_file.duration_seconds

        # estimated cost
        estimated_cost = total_sec / 60 * 0.006 * 32
        logger.info("estimated cost in TWD: ${:.2f}".format(estimated_cost))

        # audio to text and merge
        # # split into chunks of 100 seconds
        # chunk_size = 100 * 1000  # 100 secs
        # chunks = [
        #     audio_file[i : i + chunk_size]
        #     for i in range(0, len(audio_file), chunk_size)
        # ]
        # # use openai api to transcribe
        # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # transcripts = []
        # for idx, chunk in tqdm(enumerate(chunks), total=len(chunks)):
        #     fname_i = f"{PathHelper.audio_dir}/temp_{fname}_{idx}.mp3"
        #     with chunk.export(fname_i, format="mp3") as f:
        #         try:
        #             result = client.audio.transcriptions.create(
        #                 model="whisper-1", response_format="vtt", file=f
        #             )
        #             result_punct = punct_model.restore_punctuation(result.text)
        #             transcripts.append(result_punct)
        #         except Exception as e:
        #             logger.error(e)
        #             continue
        #         finally:
        #             os.remove(fname_i)
        seg_i = []
        segments, info = model.transcribe(
            str(PathHelper.audio_dir / fname_ext),
            beam_size=5,
            initial_prompt="以下是普通話的句子。",
        )
        logger.info(
            "Detected language '%s' with probability %f"
            % (info.language, info.language_probability)
        )

        if info.language not in ("en", "zh"):
            with open(PathHelper.text_dir / f"{fname}.txt", "w") as f:
                json.dump("", f)
            continue

        for segment in segments:
            # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            seg_i.append([segment.start, segment.end, segment.text])

        # merge transcripts
        transcript_text = [
            [s[2] for s in seg_i[i : i + 10]] for i in range(0, len(seg_i), 10)
        ]
        transcript_text_restore = []
        for transcript_text_i in transcript_text:
            result_punct = punct_model.restore_punctuation(" ".join(transcript_text_i))
            transcript_text_restore.append(result_punct)

        transcript_processed = "".join(transcript_text_restore)

        # save transcript
        with open(PathHelper.text_dir / f"{fname}.txt", "w") as f:
            json.dump(transcript_processed, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel-name",
        type=str,
        help="Channel Name without @",
        # default="SunnyHuangIBCLC"
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="size of the subset of files to be processed",
        default=-1,
    )

    args = parser.parse_args()
    main(args)
