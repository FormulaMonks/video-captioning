import streamlit as st
import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from pytube import YouTube
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained(
    "Neleac/timesformer-gpt2-video-captioning"
).to(device)


def main():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("Video Captioning with TimesFormer-GPT2")

    # Option to choose between uploading a video file or pasting a YouTube link
    option = st.radio(
        "Choose an option:", ("Paste a YouTube link", "Upload a video file")
    )

    if option == "Upload a video file":
        video_file = st.file_uploader("Upload a video file", type=["mp4"])
        if video_file is not None:
            st.video(video_file)
            frames, seg_len = extract_frames(video_file)
            if frames:
                generate_captions(frames, seg_len)
    elif option == "Paste a YouTube link":
        youtube_link = st.text_input("Paste a YouTube link")
        if youtube_link:
            st.write("Downloading video from YouTube...")
            video_file_path = download_youtube_video(youtube_link)
            if video_file_path:
                st.video(video_file_path)
                frames, seg_len = extract_frames(video_file_path)
                if frames:
                    generate_captions(frames, seg_len)


# Add a function to download a YouTube video
def download_youtube_video(youtube_link):
    try:
        yt = YouTube(youtube_link)
        stream = yt.streams.filter(file_extension="mp4", progressive=True).first()
        video_file_path = stream.download()
        return video_file_path
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None

# Add a function to extract frames from a video
def extract_frames(video_file):
    container = av.open(video_file)
    total_frames = container.streams.video[0].frames
    frame_rate = container.streams.video[0].average_rate
    chunk_duration = 1
    frames_per_chunk = chunk_duration * frame_rate

    # Calculate the number of chunks
    num_chunks = total_frames // frames_per_chunk
    # st.write(f"total_frames: {total_frames}")
    # st.write(f"num_chunks: {num_chunks}")
    # st.write(f"frame_rate: {frame_rate}")
    # st.write(f"frames_per_chunk: {frames_per_chunk}")
    # st.write(f"chunk_duration: {chunk_duration}")


    frames_list = []

    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * frames_per_chunk
        end_frame = start_frame + frames_per_chunk

        frames = []

        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i >= start_frame and i < end_frame:
                frames.append(frame.to_ndarray(format="rgb24"))
            elif i >= end_frame:
                break

        frames_list.append(frames)

    return frames_list, total_frames

def resize_frames(frames, target_num_frames):
    # Calculate the number of frames in the input
    num_frames = len(frames)
    
    # Ensure that the target number of frames is not greater than the number of frames
    target_num_frames = min(target_num_frames, num_frames)
    
    # Calculate the step size for frame selection
    step_size = num_frames / target_num_frames
    
    # Initialize the resized frames list
    resized_frames = []
    
    for i in range(target_num_frames):
        # Calculate the index of the frame to select
        index = int(i * step_size)
        
        # Select and append the frame
        resized_frames.append(frames[index])
    
    return resized_frames

# Add a function to generate captions
def generate_captions(frames_list, seg_len):
    st.write("Generating description...")

    # st.write(len(frames_list))

    gen_kwargs = {
        "min_length": 10,
        "max_length": 20,
        "num_beams": 8,
    }

    prompt = """
Generate a coherent video description by merging individual captions. 
Take the following captions, ordered chronologically, and create a cohesive and readable description. 
You may add transition phrases or sentences for better coherence. 
The goal is to produce a clear and engaging description of the video content.

Captions:
"""

    i = 0
    target_num_frames = model.config.encoder.num_frames
    for chunk_frames in frames_list:
        resized_chunk_frames = resize_frames(chunk_frames, target_num_frames)
        pixel_values = image_processor(resized_chunk_frames, return_tensors="pt").pixel_values.to(device)
        tokens = model.generate(pixel_values, **gen_kwargs)
        caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        prompt += f"\n{caption}"
        # st.write(f"{i}: {caption}")
        i += 1

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
    )

    st.write("Generated Caption:")
    if not completion:
        st.write(caption)
    else:
        st.write(completion.choices[0].message["content"])


if __name__ == "__main__":
    main()
