from fastapi import FastAPI
from langserve import add_routes
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model= "gemma2-9b-it", api_key=api_key)

# Create prompt Template
system_prompt = "Translate the following into {language}:"
promt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{text}")
    ]
)
chain = promt | llm | StrOutputParser()

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Simple API server using LangChain Runnable Interfaces"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
