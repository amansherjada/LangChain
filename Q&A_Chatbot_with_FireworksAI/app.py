import streamlit as st
import fireworks.client
from langchain_fireworks import ChatFireworks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Helpful Assistant. Plaease Response to the User Queries"),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    fireworks.client.api_key = api_key
    llm = ChatFireworks(model=llm, temperature=temperature, max_tokens=max_tokens)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question})
    return answer

# Title of the app
st.title("Q&A Chatbot with Fireworks AI")

# Sidebar for api_key
st.sidebar.title("Parameters")
api_key = st.sidebar.text_input("Enter your Fireworks AI API Key", type="password")

# Dropdown for llm
model_options = {
    "Llama3.1-8b-instruct": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "Mixtral-8x7b-instruct": "accounts/fireworks/models/mixtral-8x7b-instruct"
}
llm = st.sidebar.selectbox("Select an LLM Model", list(model_options.keys()))

# Slider for Parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

max_tokens = st.sidebar.slider("Max Token", min_value=50, max_value=300, value=150)

# Main interface for user input
user_input = st.text_input("Enter Message", placeholder="Message")

if st.button("Submit"):
    if user_input and api_key:
        response = generate_response(user_input, api_key, model_options[llm], temperature, max_tokens)
        st.write(response)
    
    elif user_input:
        st.write("Please enter your Fireworks API key in the side bar")

    else:
        st.write("Please provide the message")
st.divider()
st.markdown('You can create your API key here: [Fireworks AI](https://fireworks.ai/account/api-keys)')