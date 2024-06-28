import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, ViltForQuestionAnswering
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
import av

# Ensure torch is imported and version is printed
print("Torch version:", torch.__version__)

# Load VQA model, processor, and tokenizer from Hugging Face
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
tokenizer = AutoTokenizer.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Streamlit app
st.title("Visual Question Answering")

st.write("Upload an image or capture a live image and ask a question about it.")

# Class to process the video stream
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.image = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = Image.fromarray(img)
        self.image = image
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit WebRTC streamer
ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)

# Display captured image from webcam
if ctx.video_transformer and ctx.video_transformer.image:
    st.image(ctx.video_transformer.image, caption='Captured Image', use_column_width=True)

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Question input
question = st.text_input("Ask a question about the image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
elif ctx.video_transformer and ctx.video_transformer.image:
    image = ctx.video_transformer.image

if image is not None and question:
    # Prepare inputs for the VQA model with padding
    inputs = processor(images=image, text=question, return_tensors="pt", padding="max_length", truncation=True)

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_answer = torch.argmax(logits, dim=1)

    # Convert predicted answer to text
    answer = tokenizer.decode(predicted_answer, skip_special_tokens=True)

    st.write("Answer:", answer)
