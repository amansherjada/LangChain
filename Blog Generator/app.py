import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

# Function to get response from Llama 3 Model

def getLLMresponse(input_text, no_words, blog_style):

    #Llama 3 model
    llm = Ollama(model = "llama3", temperature = 0.3)

    #Promt Template
    template = """
    Write a blog for {blog_style} job profile for a topic {input_text}
    within {no_words} words
    """
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"],
                            template=template)
    
    # Generate response from Llama Model
    response = llm(prompt.format(blog_style=blog_style, input_text = input_text, no_words = no_words))
    print(response)
    return response

st.set_page_config(page_title= "Generate Blogs",
                    page_icon= '📝',
                    layout= "centered",
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 📝")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("No. of Words")
with col2:
    blog_style = st.selectbox("Writing the Blog for",
    ("Researchers", "Data Scientist", "Common People"))

submit = st.button("Generate")

# Final Output

if submit:
    st.write(getLLMresponse(input_text, no_words, blog_style))