import streamlit as st
import numpy as np
import requests
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import ssl  # Add this line to import the 'ssl' module
import os
from urllib import request
import re
import imageio
from IPython import display as embed


# Utilities to fetch videos from UCF101 dataset
# (Please replace these with your actual implementations)
UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
_VIDEO_LIST = None
_CACHE_DIR = "/path/to/cache/directory"
unverified_context = ssl._create_unverified_context()

def list_ucf_videos():
    index = request.urlopen(UCF_ROOT, context=unverified_context).read().decode("utf-8")
    videos = re.findall("(v_[\w_]+\.avi)", index)
    return sorted(set(videos))

def fetch_ucf_video(video):
    cache_path = os.path.join(_CACHE_DIR, video)
    if not os.path.exists(cache_path):
        urlpath = request.urljoin(UCF_ROOT, video)
        print("Fetching %s => %s" % (urlpath, cache_path))
        data = request.urlopen(urlpath, context=unverified_context).read()
        open(cache_path, "wb").write(data)
    return cache_path

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0

def to_gif(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, duration=40)
    return embed.embed_file('./animation.gif')


# Get the kinetics-400 action labels from the GitHub repository.
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
    labels = [line.decode("utf-8").strip() for line in obj.readlines()]
print("Found %d labels." % len(labels))

# Load the pre-trained I3D Kinetics-400 model from TensorFlow Hub
model_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
i3d = hub.load(model_url).signatures['default']

# Helper function to preprocess the input video
# Helper function to preprocess the input video
def preprocess_video(uploaded_file):
    # Save the uploaded video to a temporary file
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load the video using cv2.VideoCapture
    sample_video = load_video(temp_video_path)

    # Add a batch axis to the sample video.
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

    # Remove the temporary video file
    os.remove(temp_video_path)

    return model_input


# Helper function to make predictions
def predict(sample_video):
    logits = i3d(sample_video)['default'][0]
    probabilities = tf.nn.softmax(logits)

    # Get the top 5 actions
    top_actions = [
        (labels[i], probabilities[i].numpy() * 100)
        for i in np.argsort(probabilities)[::-1][:5]
    ]

    return top_actions

# Streamlit app
st.title("Pose Estimation & Action Recognition ")
uploaded_file = st.file_uploader("Choose a video...", type="mp4")

if uploaded_file is not None:
    # Display the uploaded video
    st.video(uploaded_file)

    # Preprocess the video
    processed_video = preprocess_video(uploaded_file)

    # Make predictions
    predictions = predict(processed_video)

    # Display predictions
    st.header("Top 5 Actions:")
    for action, probability in predictions:
        st.write(f"{action}: {probability:.2f}%")
