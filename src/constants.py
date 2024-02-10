# constants
VIDEO_ID = "video_id"
VIDEO_URL = "video_url"
TITLE = "title"
TRNASCRIPT = "transcript"
CHANNEL_NAME = "channel_name"

# model name
# MODEL_NAME = "shibing624/text2vec-base-chinese"
ENCODING_MODEL_NAME = "GanymedeNil/text2vec-large-chinese"
CHAT_GPT_MODEL_NAME = "gpt-3.5-turbo"

# db
DB = "db"
CHROMA_DB = "chroma"

# m docs
N_DOCS = 5
N_SOURCE_DOCS = 2

# message
INIT_MESSAGE = "我可以怎麼幫你呢？"
TEST_RESPONSES = ["測試一", "測試二", "測試三", "測試四", "測試五"]

# condense_question_prompt
PROMPT = (
    "Given the following conversation and a follow up question, "
    + "rephrase the follow up question to be a standalone question,"
    + "in Traditional Chinese and DO NOT in Simplified Chinese."
)

# lang
DEFAULT_LANG = "zh-TW"

# s3
S3_BUCKET_NAME = "ask-me-parenting"

# collection
COLLECTION_NAME = "video_chunks"
