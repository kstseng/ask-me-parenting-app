# ask-me-parenting-line-bot


1. get video entites
```
channe_name=SunnyHuangIBCLC
channe_name=DrTNHuang
channe_name=imleaderkid
python3 video_crawler.py --channel-name=$channe_name
```

2. extract transcript and process, then save to text
```
python3 transcript_to_text.py
```

3. video to audio if no transcript
```
python3 video_to_audio.py --channel-name=$channe_name
```

4. use api to get transcript for videos without transcript
```
python3 audio_to_text.py --channel-name=$channe_name --limit=10
```

5. save to local as persistent storage
```
python3 storage.py
```

6. run server (for streamlit)
```
sh run_server.sh
```