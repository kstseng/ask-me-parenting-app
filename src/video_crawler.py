import argparse
import json
import time

# import mysql.connector
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from youtube_transcript_api import YouTubeTranscriptApi

try:
    import constants as const
    from utils import PathHelper, get_logger
except Exception as e:
    print(e)
    raise ("Please run this script from the root directory of the project")

# logger
logger = get_logger(__name__)


def main(args):
    channel_name = args.channel_name
    logger.info(f"channel_name: {channel_name}")

    # init driver
    driver = webdriver.Chrome()

    url = f"https://www.youtube.com/@{channel_name}"
    driver.get(url + "/videos")

    # scroll
    ht = driver.execute_script("return document.documentElement.scrollHeight;")
    while True:
        prev_ht = driver.execute_script("return document.documentElement.scrollHeight;")
        driver.execute_script(
            "window.scrollTo(0, document.documentElement.scrollHeight);"
        )
        time.sleep(2)
        ht = driver.execute_script("return document.documentElement.scrollHeight;")
        if prev_ht == ht:
            break

    # save
    # https://stackoverflow.com/questions/74578175/getting-video-links-from-youtube-channel-in-python-selenium
    videos = []
    try:
        for e in WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div#details"))
        ):
            # attr
            title = e.find_element(By.CSS_SELECTOR, "a#video-title-link").get_attribute(
                "title"
            )
            vurl = e.find_element(By.CSS_SELECTOR, "a#video-title-link").get_attribute(
                "href"
            )

            # append
            videos.append(
                {
                    const.VIDEO_URL: vurl,
                    const.TITLE: title,
                }
            )
    except Exception as e:
        e
        pass

    logger.info(f"# videos from {channel_name}: {len(videos)}")

    # get transcripts
    for video_i in videos:
        video_id = video_i[const.VIDEO_URL].split("=")[-1]
        video_i[const.VIDEO_ID] = video_id
        video_i[const.CHANNEL_NAME] = channel_name
        logger.info(f"video id: {video_id}")

        entity_fname = PathHelper.entities_dir / f"{video_i[const.VIDEO_ID]}.json"

        # check if file exist
        if entity_fname.exists():
            logger.info(f"file exist: {entity_fname}")
            # jump to next video
            continue

        try:
            transcript_i = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["zh-TW"]
            )
            video_i[const.TRNASCRIPT] = transcript_i
        except Exception as e:
            e
            video_i[const.TRNASCRIPT] = []
        finally:
            # save obj as json to local
            with open(
                PathHelper.entities_dir / f"{video_i[const.VIDEO_ID]}.json", "w"
            ) as f:
                json.dump(video_i, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel-name", type=str, help="Channel Name without @", default="DrTNHuang"
    )

    args = parser.parse_args()
    main(args)
