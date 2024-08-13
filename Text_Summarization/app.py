import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import requests
from bs4 import BeautifulSoup

# Define a simple Document class to mimic LangChain's document structure
class SimpleDocument:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}  # Adding a metadata attribute

# Streamlit App
st.set_page_config(page_title="Content Summary Generator", page_icon="📝")
st.title("Generate Summaries from YouTube Videos and Websites")
st.subheader("Summarize URL")

# Get Groq API key and URL
with st.sidebar:
    api_key = st.text_input("Groq API Key", placeholder="Enter here", type="password")
    if st.button("Submit API Key"):
        if api_key.strip():
            st.session_state.api_key = api_key
            st.success("API Key has been set!")
        else:
            st.error("Please enter a valid API Key.")
    st.link_button(label="Create your API key here", url="https://console.groq.com/keys")
    

paste_url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Provide a summary of the following content in 500 words:
content: {text}
"""
prompt = PromptTemplate(
    input_variables=["text"],
    template=prompt_template
)

if st.button("Summarize"):
    # Validate all inputs
    if not api_key.strip() or not paste_url.strip():
        st.error("Please provide both the Groq API key and a valid URL before proceeding.")
    elif not validators.url(paste_url):
        st.error("Please provide a valid URL")
    else:
        try:
            with st.spinner("In progress..."):
                # Initialize ChatGroq only after the API key is provided
                llm = ChatGroq(model="gemma2-9b-it", groq_api_key=api_key)
                
                # Loading the YT or Website data
                if "youtube.com" in paste_url or "youtu.be" in paste_url:
                    loader = YoutubeLoader.from_youtube_url(paste_url, add_video_info=True)
                    docs = loader.load()

                else:
                    response = requests.get(paste_url)
                    soup = BeautifulSoup(response.content, "html.parser")
                    page_text = soup.get_text(separator="\n", strip=True)
                    docs = [SimpleDocument(content=page_text)]

                if docs:
                    # Chain for Summarization
                    chain = load_summarize_chain(
                        llm,
                        chain_type="stuff",
                        prompt=prompt
                    )

                    output_summary = chain.invoke(docs)
                    st.success(output_summary["output_text"])
                    st.toast('Hooray!', icon='🎉')
                else:
                    st.error("Could not extract any content to summarize.")
        except Exception as e:
            st.exception(f"Error occurred: {e}")
st.divider()
st.link_button(label="Project Creator", url="https://github.com/amansherjada")
