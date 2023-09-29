import os
from dotenv import load_dotenv

load_dotenv()

KEY_PATH = os.getenv("SPEECH_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SAMPLE_RATE_HERTZ = 16000
LANGUAGE_CODE = "en-US"
