import os
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books","odyssey.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"This File path does not exist {file_path}, Check the path")

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
docs = text_splitter.split_documents(documents)

# Function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")

# 1. OpenAI Embeddings
# Uses OpenAI's embedding models.
# Useful for general-purpose embeddings with high accuracy.
# Note: The cost of using OpenAI embeddings will depend on your OpenAI API usage and pricing plan.
# Pricing: https://openai.com/api/pricing/
# print("\n--- Using OpenAI Embeddings ---")
# openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# create_vector_store(docs, openai_embeddings, "chroma_db_openai")

# 2. Cohere Embeddings
# Uses Cohere's embedding models.
# Suitable for generating embeddings that capture semantic meaning.
# Note: The cost of using Cohere embeddings will depend on your Cohere API usage and pricing plan.
# Pricing: https://cohere.ai/pricing/
print("\n--- Using Cohere Embeddings ---")
cohere_api_key = os.getenv("COHERE_API_KEY")  # Retrieve the API key from environment variables
cohere_embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=cohere_api_key)
create_vector_store(docs, cohere_embeddings, "chroma_db_cohere")

# 3. Hugging Face Transformers
# Uses models from the Hugging Face library.
# Ideal for leveraging a wide variety of models for different tasks.
# Note: Running Hugging Face models locally on your machine incurs no direct cost other than using your computational resources.
# Note: Find other models at https://huggingface.co/models?other=embeddings
print("\n--- Using Hugging Face Transformers ---")
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")

print("Embedding demonstrations for Cohere and Hugging Face completed.")

# Function to query a vector store
def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")


# Define the user's question
query = "Who is Odysseus' wife?"

# Query each vector store
query_vector_store("chroma_db_cohere", query, cohere_embeddings)
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)

print("Querying demonstrations completed.")