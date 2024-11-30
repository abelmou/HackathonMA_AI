import streamlit as st
import numpy as np
import cv2
import pytesseract
from PIL import Image
import tempfile
import os
import uuid  
import fitz
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import openai

# Azure Storage credentials
connection_string = "**"
container_name = "store"

# Initialize the BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# OpenAI API key
openai.api_key = '**'  

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, processed = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    return processed

# Function to extract text using pytesseract
def extract_text(image_path):
    image = preprocess_image(image_path)
    text = pytesseract.image_to_string(image, lang="ara+eng+fra")
    return text

# Function to extract images from a PDF
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_number in range(len(doc)):
        pix = doc[page_number].get_pixmap(dpi=300)
        image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        images.append(image_array)
    return images

# Function to correct extracted text using OpenAI
def correct_text(input_text):
    """
    Correct and structure OCR-extracted healthcare text using OpenAI GPT model.
    """
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" for more advanced models
            prompt=f"Correct and format the following healthcare OCR text with proper structure:\n\n{input_text}",
            max_tokens=500,
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        # Accessing the corrected text from the response
        corrected_text = response['choices'][0]['text']
        return corrected_text
    except Exception as e:
        return f"Error in text correction: {e}"

# Apply custom styles
st.markdown(
    """
    <style>
        body { background-color: #f8faff; }
        .main { background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); }
        h1, h2, h3 { color: #0056b3; }
        h1 { text-align: center; font-size: 2.5rem; }
        h2 { text-align: center; font-size: 1.8rem; }
        h3 { font-size: 1.3rem; }
        p { text-align: center; color: #6c757d; font-size: 1rem; }
        textarea { font-family: "Courier New", monospace; font-size: 1rem; color: #333; }
        div[data-testid="stFileUploader"] > label { font-size: 1.2rem; color: #0056b3; }
        .stButton > button { background-color: #0056b3; color: white; font-weight: bold; font-size: 1rem; border-radius: 5px; }
        .stButton > button:hover { background-color: #003d80; }
        .success { color: #28a745; font-weight: bold; }
        .error { color: #dc3545; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display Logo and App Title
st.image("images.jpg", width=150, caption="Se7ty Healthcare App", use_column_width=False)
st.markdown("<h1>Se7ty Healthcare App</h1>", unsafe_allow_html=True)
st.markdown("<p>Easily process and save your ordonnance (prescription).</p>", unsafe_allow_html=True)

# File uploader for images or PDFs
uploaded_file = st.file_uploader("Upload an Image or PDF", type=["jpg", "jpeg", "png", "pdf"])
use_camera = st.checkbox("Take a Photo with Your Camera")

image_to_process = None

if uploaded_file:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            pdf_images = extract_images_from_pdf(temp_pdf.name)
        if pdf_images:
            image_to_process = pdf_images[0]
            st.image(image_to_process, caption="First Page of Uploaded PDF", use_column_width=True)
        else:
            st.error("No images found in the PDF.")
    else:
        image_to_process = np.array(Image.open(uploaded_file))
        st.image(image_to_process, caption="Uploaded Ordonnance", use_column_width=True)

elif use_camera:
    picture = st.camera_input("Capture your Ordonnance")
    if picture:
        image_to_process = np.array(Image.open(picture))
        st.image(image_to_process, caption="Captured Ordonnance", use_column_width=True)

if image_to_process is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp_image_path = temp.name
        cv2.imwrite(temp_image_path, cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR))

    preprocessed_image = preprocess_image(temp_image_path)
    st.image(preprocessed_image, caption="Preprocessed Ordonnance", channels="GRAY")

    # Extract text
    st.markdown("<h3>Recognized Text:</h3>", unsafe_allow_html=True)
    recognized_text = extract_text(temp_image_path)
    st.text_area("Extracted Ordonnance Text", recognized_text, height=200)

    # Correct the extracted text
    st.markdown("<h3>Corrected Text:</h3>", unsafe_allow_html=True)
    corrected_text = correct_text(recognized_text)
    st.text_area("Corrected Text", corrected_text, height=200)

    # Save corrected text to a file
    random_file_name = f"corrected_ordonnance_{uuid.uuid4().hex}.txt"
    with open(random_file_name, "w", encoding="utf-8") as file:
        file.write(corrected_text)

    st.markdown(f"<p class='success'>Corrected text saved to `{random_file_name}`.</p>", unsafe_allow_html=True)

    # Upload the corrected text to Azure
    try:
        blob_client = container_client.get_blob_client(random_file_name)
        with open(random_file_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        st.markdown(f"<p class='success'>File `{random_file_name}` uploaded successfully to Azure.</p>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<p class='error'>Error uploading file to Azure. Please try again. Error: {e}</p>", unsafe_allow_html=True)

    os.remove(temp_image_path)
