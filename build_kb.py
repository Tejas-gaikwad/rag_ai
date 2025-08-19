# build_kb.py

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import getpass
import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from google import genai




# Example documents (replace with your own data loading logic)
documents = [
    {"title": "AI Research", "content": "AI is transforming industries."},
    {"title": "Machine Learning", "content": "ML allows computers to learn from data."},
    {"title": "RAG Models", "content": "RAG combines retrieval with generation."}
]

print("Starting pls wait...")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Extract just the text content
texts = [doc["content"] for doc in documents]



client = genai.Client()
print("KEY configuered ...")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)


# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents="Explain how AI works in a few words"
# )
print("embeddings done")


index = faiss.IndexFlatL2((3072)) 
# //embeddings.embed_query("hello world")

print("index also done")


# Create FAISS vector store
# vector_store = FAISS.from_texts(texts, embeddings)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

print("vector store working")

# Save locally
vector_store.save_local("./knowledge_base")

print("✅ Knowledge base created successfully!")
