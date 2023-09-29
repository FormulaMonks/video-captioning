from abc import abstractmethod
import io
from google.cloud import texttospeech
from google.oauth2 import service_account
from elevenlabs import set_api_key, Voice, VoiceSettings, generate, play as play_eleven
import logging
import streamlit as st
from constants import LANGUAGE_CODE, ELEVEN_LABS_API_KEY
import tempfile
import os

logger = logging.getLogger(__name__)

class TextToSpeech:
    @abstractmethod
    def synthesize(self, text: str):
        pass

class GoogleCloudTTS(TextToSpeech):
    def __init__(self, credentials):
        self.client = texttospeech.TextToSpeechClient(credentials=credentials)

    def synthesize(self, text_response):
        try:
            input_text = texttospeech.SynthesisInput(text=text_response)
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=LANGUAGE_CODE, name="en-US-Wavenet-F", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            response = self.client.synthesize_speech(input=input_text, voice=voice_params, audio_config=audio_config)

            # Save the audio to a temporary file
            temp_audio_path = os.path.join(tempfile.gettempdir(), "google_tts.mp3")
            with open(temp_audio_path, "wb") as audio_file:
                audio_file.write(response.audio_content)

            # Play the audio using Streamlit's audio widget
            st.audio(temp_audio_path, format="audio/mp3")
        except Exception as e:
            logger.error(f"Error in GoogleCloudTTS: {e}")
            raise

class ElevenLabsTTS(TextToSpeech):
    def __init__(self):
        set_api_key(ELEVEN_LABS_API_KEY)

    def synthesize(self, text: str):
        try:
            audio = generate(
                text=text,
                voice=Voice(
                    voice_id='KavW1Pkc0hhhh7ge60Uk',
                    settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
                )
            )

            # Save the audio to a temporary file
            temp_audio_path = os.path.join(tempfile.gettempdir(), "eleven_labs_tts.mp3")
            audio.export(temp_audio_path, format="mp3")

            # Play the audio using Streamlit's audio widget
            st.audio(temp_audio_path, format="audio/mp3")
        except Exception as e:
            logger.error(f"Error in ElevenLabsTTS: {e}")
            raise

def main():
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("Text-to-Speech App")

    # Create instances of TTS engines
    google_tts = GoogleCloudTTS(credentials)  # Initialize with your Google Cloud credentials
    eleven_labs_tts = ElevenLabsTTS()  # Initialize with your Eleven Labs API key

    # Get user input text
    text_input = st.text_area("Enter the text you want to convert to speech:")

    if st.button("Generate Google TTS"):
        google_tts.synthesize(text_input)

    if st.button("Generate Eleven Labs TTS"):
        eleven_labs_tts.synthesize(text_input)

if __name__ == "__main__":
    main()
