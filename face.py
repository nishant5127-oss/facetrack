# Import operating system module to check for local model file existence, and urllib to download if missing
import os, urllib.request

# Import OpenCV for image color conversion and video decoding, and NumPy for vector array manipulation
import cv2, numpy as np

# Import Google MediaPipe framework for running ML vision tasks
import mediapipe as mp

# Import MediaPipe Python task API bindings
from mediapipe.tasks import python as mp_python

# Import MediaPipe Vision tasks (specifically FaceLandmarker)
from mediapipe.tasks.python import vision

# Local filename where the pre-trained MediaPipe face landmark model will be stored
MODEL_PATH = "face_landmarker.task"

# Google Cloud Storage URL to download the float16 quantized FaceLandmarker model (~3.76 MB)
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Global singleton cache variable for the FaceLandmarker instance to prevent reloading on each call
_landmarker = None

def _get_landmarker():
    """Lazily loads and returns the singleton MediaPipe FaceLandmarker instance."""
    global _landmarker
    # Check if the model has not been initialized yet in memory
    if _landmarker is None:
        # If the model binary file does not exist locally on disk, download it from Google's CDN
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)  # ~4 MB, first run only
        
        # Configure FaceLandmarker options: specify model file path and restrict detection to 1 primary face
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH), num_faces=1)
        
        # Instantiate the FaceLandmarker model from the configured options
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    
    # Return the cached landmarker instance
    return _landmarker


def decode(data: bytes):
    """Decodes raw image file bytes (JPEG/PNG) into an OpenCV BGR numpy array."""
    # Convert byte buffer to 1D uint8 NumPy array, then decode into BGR image matrix
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def embed(img_bgr):
    """
    Extracts 468 3D facial landmarks from a BGR image, normalizes them for scale
    and translation, and returns a 1404-dimensional flattened float vector.
    """
    # Convert OpenCV BGR channel ordering to RGB, ensuring array memory is contiguous
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    # Run MediaPipe face detection and landmark inference on the RGB image tensor
    res = _get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    
    # If no face was detected in the frame, return None
    if not res.face_landmarks:
        return None
    
    # Slice the first 468 standard facial mesh landmarks (ignoring dense iris/eyeball points)
    lm = res.face_landmarks[0][:468]
    
    # Extract [x, y, z] coordinates from each landmark into a (468, 3) float32 array
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    
    # TRANSLATION NORMALIZATION: Subtract Landmark #1 (nose tip) so the nose tip becomes origin (0, 0, 0)
    pts -= pts[1]                                        # nose tip as origin
    
    # SCALE NORMALIZATION: Compute Euclidean distance between Landmark #33 and #263 (outer eye corners)
    # and divide all coordinates by this inter-ocular distance (unit scale)
    pts /= (np.linalg.norm(pts[33] - pts[263]) or 1.0)   # outer eye-corner distance as unit
    
    # Flatten (468, 3) matrix into a 1D vector of 1404 values (468 * 3 = 1404)
    return pts.flatten()


def embed_bytes(data: bytes):
    """Helper function: takes raw image bytes, decodes them, and extracts the 1404-D face embedding."""
    # Decode raw byte string into an OpenCV image
    img = decode(data)
    # If decoding failed (corrupted file), return None; otherwise compute embedding
    return None if img is None else embed(img)


def sample_frames(video_path: str, every_sec: float = 1.0, max_frames: int = 30):
    """
    Extracts frames from a video file sampled once every `every_sec` seconds,
    up to a maximum of `max_frames` to prevent high resource usage.
    """
    # Open the video stream using OpenCV VideoCapture
    cap = cv2.VideoCapture(video_path)
    
    # Retrieve the frames-per-second (FPS) rate of the video (fallback to 25 if undetected)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    # Calculate step interval (number of frames per sample interval), frame counter i, and output list
    step, i, out = max(int(fps * every_sec), 1), 0, []
    
    # Loop to read frames until max_frames is reached
    while len(out) < max_frames:
        # Read next video frame
        ok, frame = cap.read()
        # If end of video file reached or read error, break loop
        if not ok: break
        # If current frame index is on the step boundary, append frame to output list
        if i % step == 0: out.append(frame)
        # Increment frame counter
        i += 1
    
    # Release the video capture object and free memory
    cap.release()
    
    # Return list of extracted BGR image frames
    return out
