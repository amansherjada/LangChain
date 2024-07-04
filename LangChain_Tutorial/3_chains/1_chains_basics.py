from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_fireworks import ChatFireworks

# Load environment variables from .env
load_dotenv()

# Create a model
model = ChatFireworks(model= "accounts/fireworks/models/llama-v3-70b-instruct")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"topic": "Alex Pereira", "joke_count": 2})

# Output
print(result)