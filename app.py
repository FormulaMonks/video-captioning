import streamlit as st
import av
import os
from pytube import YouTube
from transformers import pipeline
import openai
from PIL import Image
import torch
from google.oauth2 import service_account
from text_to_speech import GoogleCloudTTS
from constants import KEY_PATH, OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

class VideoProcessor:
    def __init__(self):
        print("Loading Model...")
        self.model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

class VideoExtractor:
    @staticmethod
    def extract_frames(video_file):
        container = av.open(video_file)
        total_frames = container.streams.video[0].frames

        num_frames_to_capture = min(10, total_frames)
        print(f"{num_frames_to_capture=}")

        # Calculate the step size to capture frames at equal intervals
        step_size = total_frames // num_frames_to_capture  # Capture a minimum of 10 frames or all frames if less

        frames_list = []

        for i, frame in enumerate(container.decode(video=0)):
            if i % step_size == 0:
                frames_list.append(frame.to_ndarray(format="rgb24"))

        return frames_list, num_frames_to_capture

class VideoDownloader:
    @staticmethod
    def download_youtube_video(youtube_link):
        try:
            yt = YouTube(youtube_link)
            
            # Get video details
            video_title = yt.title
            video_thumbnail_url = yt.thumbnail_url
            
            # Download the video
            stream = yt.streams.filter(file_extension="mp4", progressive=True).first()
            video_file = stream.download()
            
            return {
                "video_file": video_file,
                "video_title": video_title,
                "video_thumbnail_url": video_thumbnail_url
            }
        except Exception as e:
            st.error(f"Error downloading YouTube video: {e}")
            return None

class CaptionGenerator:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.prompt = """
        Generate a coherent video description by merging individual captions that belong to the same video. 
        Take the following captions, ordered chronologically, and create a cohesive and readable description. 
        You may add transition phrases or sentences for better coherence. 
        The goal is to produce a clear and engaging description of the video content without too much repetition.

        """

    def generate_caption(self, image):
        return self.video_processor.model(image)[0]["generated_text"]

    def generate_captions(self, frames_list, seg_len):
        self.prompt += "\nCaptions:"
        with st.status("Analyzing frames..."):
            for idx, frame in enumerate(frames_list):
                pil_image = Image.fromarray(frame)
                caption = self.generate_caption(pil_image)
                st.write(f"\n{idx}/{seg_len}: {caption}")
                self.prompt += f"\n{caption}"

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"{self.prompt}"},
            ],
        )

        st.write("Generated Description:")
        if not completion:
            return caption
        return completion.choices[0].message["content"]

def main():
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    t2s = GoogleCloudTTS(credentials)

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("VideoVista AI")

    # Option to choose between uploading a video file or pasting a YouTube link
    option = st.radio("Choose an option:", ("Paste a YouTube link", "Upload a video file"))

    caption_generator = CaptionGenerator()
    video_file = None

    if option == "Upload a video file":
        video_file = st.file_uploader("Upload a video file", type=["mp4"])
    elif option == "Paste a YouTube link":
        youtube_link = st.text_input("Paste a YouTube link")
        if youtube_link:
            st.write("Downloading video from YouTube...")
            video_file, video_title, video_thumbnail_url = VideoDownloader.download_youtube_video(youtube_link).values()
            caption_generator.prompt += f"\nYouTube Title: {video_title}\nYouTube Thumbnail Description: {caption_generator.generate_caption(video_thumbnail_url)}"
            print(caption_generator.prompt)

    if video_file:
        st.video(video_file)
        frames, seg_len = VideoExtractor.extract_frames(video_file)
        if frames:
            caption = caption_generator.generate_captions(frames, seg_len)
            t2s.synthesize(caption)
            st.write(caption)


if __name__ == "__main__":
    main()
