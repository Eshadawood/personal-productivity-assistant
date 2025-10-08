import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import gdown

# Download .env file from Google Drive
file_id = "1ZeqBj8W1smbq_m1LQxdqLT9YP_Jl-A0O"
url = f"https://drive.google.com/uc?id={file_id}"
output = ".env"

gdown.download(url, output, quiet=False)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever()

# Use GPT-4 for answer generation
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Load the QA chain
qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

def get_response(query: str):
    # Use the new .invoke() API
    docs = retriever.invoke(query)
    response = qa_chain.invoke({
        "input_documents": docs,  # <-- fixed key (was singular)
        "question": query
    })
    return response
