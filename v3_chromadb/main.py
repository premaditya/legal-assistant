
import os
import time
import random
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Step 1: Load Configuration
load_dotenv("../.env")

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in ../.env")

DATA_DIR = "../data"
INDEX_DIR = "chroma_db"
COLLECTION_NAME = "bnss"

BATCH_SIZE = 10
DELAY_SECONDS = 10
MAX_RETRIES = 6


# Step 2: Load PDFs
pdf_files = sorted(
    os.path.join(DATA_DIR, filename)
    for filename in os.listdir(DATA_DIR)
    if filename.lower().endswith(".pdf")
)

documents = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())


# Step 3: Update Document Metadata
for doc in documents:
    doc.metadata["file_name"] = os.path.basename(
        doc.metadata.get("source", "")
    )


# Step 4: Split Documents into Chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250,
)

chunks = splitter.split_documents(documents)

chunk_ids = [f"chunk_{i:06d}" for i in range(len(chunks))]

print("Total chunks:", len(chunks))
print(
    "Combined chunk characters:",
    sum(len(chunk.page_content) for chunk in chunks),
)


# Step 5: Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
)


# Step 6: Initialize ChromaDB
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=INDEX_DIR,
)

existing_ids = set(vectorstore.get(include=[])["ids"])

if existing_ids:
    print("Existing collection found. Loading it...")
    print("Already embedded:", len(existing_ids))

    stale = existing_ids - set(chunk_ids)
    if stale:
        raise ValueError(
            "Existing collection has more chunks than the current dataset. "
            "Use a new index directory."
        )


# Step 7: Identify Pending Chunks
pending = [
    (cid, chunk)
    for cid, chunk in zip(chunk_ids, chunks)
    if cid not in existing_ids
]


# Step 8: Embed Documents in Batches
for start in range(0, len(pending), BATCH_SIZE):
    batch = pending[start:start + BATCH_SIZE]
    batch_ids = [cid for cid, _ in batch]
    batch_docs = [chunk for _, chunk in batch]

    for attempt in range(MAX_RETRIES):
        try:
            vectorstore.add_documents(
                documents=batch_docs,
                ids=batch_ids,
            )

            completed = len(existing_ids) + start + len(batch)
            print(f"Embedded {completed}/{len(chunks)} chunks")

            if start + len(batch) < len(pending):
                time.sleep(DELAY_SECONDS)

            break

        except Exception as exc:
            message = str(exc).lower()

            retryable = any(
                term in message
                for term in [
                    "429",
                    "resource_exhausted",
                    "rate limit",
                    "quota exceeded",
                    "too many requests",
                    "500",
                    "502",
                    "503",
                    "504",
                ]
            )

            if not retryable or attempt == MAX_RETRIES - 1:
                raise

            wait = min(60, 10 * (2 ** attempt))
            wait += random.uniform(0, 3)

            print(
                f"Temporary API error. "
                f"Retrying in {wait:.1f}s "
                f"({attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)


# Step 9: Confirm Index Creation
print("Chroma index creation completed!")
print("Index saved at:", INDEX_DIR)
