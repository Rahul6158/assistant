from streamlit_webrtc import webrtc_streamer

webrtc_streamer(key="example", video_constraints={"facing_mode": "user"})
