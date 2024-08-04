import os 
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt Template

prompt = ChatPromptTemplate(
    [
        ("system","You are a Helpful Assistant. Please respond to the Question Asked"),
        ("user","Question {question}")
    ]
)
# Streamlit
st.title("Langchain Demo with Llama3")
input_text = st.text_input("What Question in your mind?")

# Ollama
llm = Ollama(model = "llama3")
chain = prompt|llm|StrOutputParser()

if input_text:
    st.write(chain.invoke({"question":input_text}))