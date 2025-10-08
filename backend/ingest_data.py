import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import gdown

# Download .env file from gdrive
file_id = "1ZeqBj8W1smbq_m1LQxdqLT9YP_Jl-A0O"
url = f"https://drive.google.com/uc?id={file_id}"
output = ".env"

gdown.download(url, output, quiet=False)

# Load environment variables from .env file
load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db")

# Ensure API key is set
if not OPEN_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# Load documents
documents = [
    Document(page_content="Meeting notes: Discuss project X deliverables."),
    Document(page_content="To-do list: 1. Finish report 2. Call client 3. Schedule meeting."),
    Document(page_content="Project ideas: Develop a productivity app using AI."),
    Document(page_content="Research topics: Explore advancements in natural language processing."),
    Document(page_content="Personal goals: 1. Exercise regularly 2. Read more books 3. Learn a new skill."),
    Document(page_content="Travel plans: Visit Japan in spring to see the cherry blossoms."),
    Document(page_content="Book summaries: 'Atomic Habits' by James Clear emphasizes the power of small habits."),
    Document(page_content="Health tips: Maintain a balanced diet and stay hydrated."),
    Document(page_content="Learning resources: Coursera, edX, and Khan Academy offer great online courses."),
    Document(page_content="Inspirational quotes: 'The only way to do great work is to love what you do.' - Steve Jobs"),
    Document(page_content='Reminder: Submit the project proposal by Friday.')
]

# Create Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

# Split text for better retrieval
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Store embeddingd in chromaDB
vector_db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_DB_PATH)

print("Documents successfully indexed!")

