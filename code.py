from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2

webrtc_streamer(key="example", video_transformer_factory=None)

