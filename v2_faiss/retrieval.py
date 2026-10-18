from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


load_dotenv("../.env")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_QUERY",   
)

vectorstore = FAISS.load_local(
    "faiss_index",                 
    embeddings,
    allow_dangerous_deserialization=True,
)

print("Vectors in index:", vectorstore.index.ntotal)

results = vectorstore.similarity_search("What is the difference between a void agreement and a voidable contract under the Indian Contract Act?", k=3)
for doc in results:
    print(doc.metadata.get("source"), "page", doc.metadata.get("page"))
    print(doc.page_content)
    print("---")