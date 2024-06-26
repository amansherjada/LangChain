import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure the Google Generative AI client with the API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load the Gemini Pro Vision model
model = genai.GenerativeModel("gemini-pro-vision")

# Function to get a response from the Gemini model
def get_gemini_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text

# Function to process the uploaded image
def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        # Prepare image parts for the API
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")

# Streamlit application configuration
st.set_page_config(page_title="MultiLanguage Invoice Extractor", page_icon="⛏")
st.header("MultiLanguage Invoice Details Extractor ⛏")

# Input prompt for the user
input = st.text_input("Input Prompt: ", key="input")

# File uploader for the invoice image
uploaded_file = st.file_uploader("Choose an Image of invoice", type=["jpg", "jpeg", "png"])

image = ""
if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Button to submit and process the invoice
submit = st.button("Tell me about the Invoice")

# Predefined prompt for the invoice analysis
input_prompt = """
You are an expert in invoice analysis. 
We will upload an image of an invoice, and your task is to 
answer any questions based on the information provided in the uploaded invoice image.
"""

# If the submit button is clicked
if submit:
    # Process the uploaded image
    image_data = input_image_details(uploaded_file)
    # Get a response from the Gemini model
    response = get_gemini_response(input_prompt, image_data, input)
    # Display the response
    st.subheader("The Response is")
    st.write(response)