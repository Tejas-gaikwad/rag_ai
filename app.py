# # import os
# # import streamlit as st
# # from dotenv import load_dotenv

# # # LangChain imports
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # from langchain.prompts import PromptTemplate

# # # 1. Load environment variables
# # load_dotenv()
# # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # # 2. Load embeddings & vector store
# # embeddings = GoogleGenerativeAIEmbeddings(
# #     model="models/embedding-001",  # ✅ use proper embedding model
# #     google_api_key=GOOGLE_API_KEY
# # )

# # vector_store = FAISS.load_local(
# #     "./knowledge_base",
# #     embeddings,
# #     allow_dangerous_deserialization=True
# # )

# # print("Vector store loaded successfully ✅")

# # # 3. Define prompt template
# # prompt = PromptTemplate(
# #     input_variables=["context", "query"],
# #     template="Based on the following context:\n{context}\nAnswer the query: {query}"
# # )

# # # 4. Initialize Gemini LLM
# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-2.0-flash",
# #     google_api_key=GOOGLE_API_KEY,
# #     temperature=0.3
# # )

# # # ✅ New way: chain prompt and llm
# # generator = prompt | llm

# # # 5. Retrieval function
# # def retrieve_documents(query, top_k=3):
# #     docs = vector_store.similarity_search(query, k=top_k)
# #     return "\n".join([doc.page_content for doc in docs])

# # # 6. Response generator
# # def generate_response(query):
# #     print(f"query ---   \n{query}")   # ✅ fixed f-string
# #     context = retrieve_documents(query)
# #     return generator.invoke({"context": context, "query": query}).content

# # # 7. Streamlit UI
# # st.title("RAG-Powered Chatbot")
# # query = st.text_input("Ask me anything:")

# # if query:
# #     response = generate_response(query)
# #     st.write("**Response:**", response)



# import os
# from dotenv import load_dotenv

# # LangChain imports
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate

# # 1. Load environment variables
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# if not GOOGLE_API_KEY:
#     raise ValueError("⚠️ GOOGLE_API_KEY not found in environment variables.")

# # 2. Load embeddings & vector store
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=GOOGLE_API_KEY
# )

# vector_store = FAISS.load_local(
#     "./knowledge_base",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# print("✅ Vector store loaded successfully")

# # 3. Define prompt template
# prompt = PromptTemplate(
#     input_variables=["context", "query"],
#     template="Based on the following context:\n{context}\nAnswer the query: {query}"
# )

# # 4. Initialize Gemini LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0.3
# )

# # ✅ New chain syntax
# generator = prompt | llm

# # 5. Retrieval function
# def retrieve_documents(query, top_k=3):
#     docs = vector_store.similarity_search(query, k=top_k)
#     return "\n".join([doc.page_content for doc in docs])

# # 6. Response generator
# def generate_response(query):
#     print(f"\n🔎 Query: {query}")
#     context = retrieve_documents(query)
#     response = generator.invoke({"context": context, "query": query})
#     return response.content

# # 7. Terminal loop
# def main():
#     print("\n🤖 RAG-Powered Chatbot (Terminal Mode)")
#     print("Type 'exit' to quit.\n")
#     while True:
#         query = input("You: ")
#         if query.lower() in ["exit", "quit", "bye"]:
#             print("👋 Goodbye!")
#             break
#         try:
#             response = generate_response(query)
#             print(f"Bot: {response}\n")
#         except Exception as e:
#             print(f"⚠️ Error: {e}\n")

# if __name__ == "__main__":
#     main()


import os
from google import generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-2.5-flash')

documents = [
    "Tejas gaikwad is a boy",
    "Tejas age is 24",
    "Tejas is born in pune but currently he is leaving in mumbai.",
    "Tejas is a softtware engineer, He works in Mirae asset",
    "Tejas love to code and learn new skills",
    "He is currenttly exploring the AI agents field",
    "Tejas love to ride motorcycle and his dream bike is BMW 1250 GSA"
]
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

vector_store = FAISS.from_texts(documents, embeddings)

def get_relevant_context(query, k=2):
    docs = vector_store.similarity_search(query, k=k)
    # print(f"\n🔎 docs: {docs}")
    return " ".join([doc.page_content for doc in docs])


# def get_relevant_context(query, docs):
#     # In a real RAG, this would involve embedding query and docs, then similarity search
#     # For simplicity, we'll just check for keyword presence
#     print(f"\n🔎 docs: {docs}")
#     relevant_chunks = []
#     for doc_id, content in docs.items():
#         if query.lower() in content.lower():
#             relevant_chunks.append(content)
#     return " ".join(relevant_chunks)

def generate_rag_response(query):
    # print(f"\n🔎 query: {query}")
    context = get_relevant_context(query, 2)
    # print(f"\n🔎 context: {context}")
    if context:
        prompt = f"Based on the following context: '{context}', answer the question: {query}"
    else:
        prompt = query # No relevant context found, answer directly

    response = model.generate_content(prompt)
    return response.text

# Example usage
# query = "what is the age of sushant singh rajput"
# print(generate_rag_response(query))

def main():
    print("\n🤖 RAG-Powered Chatbot (Terminal Mode)")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break
        try:
            response = generate_rag_response(query)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")

if __name__ == "__main__":
    main()