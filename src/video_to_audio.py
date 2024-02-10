import json
import os

from moviepy.editor import AudioFileClip
from pytube import YouTube
from tqdm import tqdm

try:
    import constants as const
    from utils import PathHelper, get_logger
except Exception as e:
    print(e)
    raise ("Please run this script from the root directory of the project")

logger = get_logger(__name__)


def download_and_convert_to_mp3(video_url, output_path):
    # get video id
    video_id = video_url.split("=")[-1]

    # skip if file exist
    if os.path.exists(f"{output_path}/{video_id}.mp3"):
        return

    # Download video from YouTube
    yt = YouTube(video_url)
    video = yt.streams.get_highest_resolution()
    video.download(output_path, filename="temp.mp4")

    # Convert to mp3
    video_clip = AudioFileClip(f"{output_path}/temp.mp4")
    video_clip.write_audiofile(f"{output_path}/{video_id}.mp3")

    # Remove the downloaded video file
    os.remove(f"{output_path}/temp.mp4")


def main():
    entites = [i for i in os.listdir(PathHelper.entities_dir) if i.endswith(".json")]

    m_docs = len(entites)
    m_docs_wo_transcript = 0
    m_docs_failed = 0

    for e in tqdm(entites, total=len(entites)):
        try:
            with open(PathHelper.entities_dir / e, "r") as f:
                data = json.load(f)

            # download the audio if no transcript
            if not data.get(const.TRNASCRIPT):
                m_docs_wo_transcript += 1
                download_and_convert_to_mp3(data[const.VIDEO_URL], PathHelper.audio_dir)
        except Exception as e:
            logger.error(e)
            m_docs_failed += 0
            continue

    logger.info(f"total docs: {m_docs}")
    logger.info(f"docs without transcript: {m_docs_wo_transcript}")
    logger.info(f"docs failed: {m_docs_failed}")


if __name__ == "__main__":
    main()
