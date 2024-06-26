# MultiLanguage-Invoice-Details-Extractor

MultiLanguage-Invoice-Details-Extractor is a Streamlit-based application that leverages Google's Generative AI to extract and analyze details from invoices in various languages. 

## Features

- Upload and process invoice images in multiple formats (JPG, JPEG, PNG).
- Analyze and extract information from invoices using Google's Gemini Pro Vision model.
- User-friendly interface with Streamlit for easy interaction.

## Demo

Check out the live demo of the application on Hugging Face Spaces: [MultiLanguage-Invoice-Details-Extractor](https://huggingface.co/spaces/amansherjada/MultiLanguage-Invoice-Details-Extractor)

## Installation

### Prerequisites

- Python 3.7+
- Streamlit
- PIL (Python Imaging Library)
- Google Generative AI Python client
- dotenv

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/amansherjada/MultiLanguage-Invoice-Details-Extractor.git
    cd MultiLanguage-Invoice-Details-Extractor
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the project directory.
    - Add your Google API key in the `.env` file:
        ```plaintext
        GOOGLE_API_KEY=your_google_api_key
        ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to interact with the application.

## Code Explanation

The main functionality is implemented in the `app.py` file:

- **Environment Variables**: Load the Google API key from the `.env` file.
- **Generative AI Configuration**: Configure the Google Generative AI client.
- **Image Processing**: Upload and display invoice images using Streamlit.
- **Invoice Analysis**: Send the uploaded image and input prompt to the Gemini model for analysis and display the response.
