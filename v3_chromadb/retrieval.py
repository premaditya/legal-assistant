
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

load_dotenv("../.env")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_QUERY",
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    collection_name="bnss",
    embedding_function=embeddings,
)

print("Vectors in index:", vectorstore._collection.count())

results = vectorstore.similarity_search(
    "Who is competent to contract under Section 11 of the Indian Contract Act?",
    k=3,
)

for doc in results:
    print(doc.metadata.get("source"), "page", doc.metadata.get("page"))
    print(doc.page_content)
    print("---")
