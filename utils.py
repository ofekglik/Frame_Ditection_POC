import os
import cv2

def ensure_dir(path):
    """Ensure that a directory exists, and if not, create it."""
    os.makedirs(path, exist_ok=True)

def save_frame(frame, path, filename):
    """Save a frame to a specified directory."""
    file_path = os.path.join(path, filename)
    cv2.imwrite(file_path, frame)
    return file_path
