import os 
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books Directory: {books_dir}")
print(f"Database Directory: {db_dir}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The file {books_dir} does not exist. Please check the path."
        )
    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    # This line creates a list of all files in the 'books_dir' directory that end with ".txt"
    # It uses a list comprehension to filter only the text files

    # Read the text content from each file and store it with metadata
    documents = []
    # Initialize an empty list to store the processed documents
    for book_file in book_files:
        # Iterate through each text file in the book_files list
        file_path = os.path.join(books_dir, book_file)
        # Create the full file path by joining the directory path and the filename
        loader = TextLoader(file_path, encoding="utf-8")
        # Create a TextLoader object for the current file
        # TextLoader from Langchain for loading text documents   
        book_docs = loader.load()
        # Load the content of the file using the TextLoader
        # This returns a list of document objects 
        for doc in book_docs:
            # Iterate through each document object created from the file
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            # Assign metadata to the document, storing the original filename as the source
            documents.append(doc)
            # Add the processed document (with metadata) to the documents list
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-mpnet-base-v2")

    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
else:
    print("Vector store already exists. No need to initialize.")