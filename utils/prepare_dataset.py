import requests
import cv2
import os
import numpy as np

def preprocess_video(sample, target_height=64, target_width=64, max_frames=80):
    """
    Downloads a video from the given sample, preprocesses it, and extracts metadata.
    The preprocessing involves cropping the video to the target dimensions and reducing
    the frame count to a maximum of `max_frames`.

    Args:
        sample (dict): A dictionary containing video metadata with keys:
                       - 'contentUrl': URL of the video.
        target_height (int): The target height for cropping the video.
        target_width (int): The target width for cropping the video.
        max_frames (int): The maximum number of frames to retain.

    Returns:
        dict: A dictionary with preprocessed video metadata:
              - height: Target video height.
              - width: Target video width.
              - channels: Number of color channels (assumes RGB).
              - frame_count: Actual number of frames after preprocessing.
              - duration: Approximate duration based on original FPS.
              - processed_frames: A list of processed video frames as NumPy arrays.
    """
    video_url = sample.get('contentUrl')
    if not video_url:
        raise ValueError("Sample must contain a 'contentUrl' key with the video URL.")
    
    video_path = "temp_video.mp4"

    # Download the video
    response = requests.get(video_url, stream=True)
    if response.status_code == 200:
        with open(video_path, 'wb') as f:
            f.write(response.content)
    else:
        raise RuntimeError(f"Failed to download the video from {video_url}. Status code: {response.status_code}")

    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        os.remove(video_path)  # Clean up
        raise RuntimeError("Error opening the video file.")

    # Get original frame count and FPS
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 1  # Avoid division by zero
    
    # Initialize processed frames
    processed_frames = []
    frame_idx = 0
    total_frames_to_read = min(original_frame_count, max_frames)

    while cap.isOpened() and frame_idx < total_frames_to_read:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to target dimensions
        height, width, _ = frame.shape
        crop_top = max(0, (height - target_height) // 2)
        crop_left = max(0, (width - target_width) // 2)
        cropped_frame = frame[crop_top:crop_top + target_height, crop_left:crop_left + target_width]

        # Add the processed frame to the list
        processed_frames.append(cropped_frame)
        frame_idx += 1

    cap.release()
    os.remove(video_path)  # Clean up downloaded video

    return {
        "height": target_height,
        "width": target_width,
        "channels": 3,
        "frame_count": len(processed_frames),
        "duration": len(processed_frames) / fps,
        "processed_frames": processed_frames
    }

def preprocess_video_tensor(my_tensor):
    my_tensor = my_tensor.float()  # Cast to float32
    my_tensor = my_tensor / 255.0  # Normalize if the input is in range [0, 255]

    return my_tensor