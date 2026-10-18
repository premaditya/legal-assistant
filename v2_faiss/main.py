
import os
import time
import random
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Configuration
# -----------------------------
load_dotenv("../.env")

# Ensure GOOGLE_API_KEY is defined in your .env file
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in ../.env")

DATA_DIR = "../data"
INDEX_DIR = "faiss_index"

BATCH_SIZE = 10
DELAY_SECONDS = 10
MAX_RETRIES = 6

# -----------------------------
# Load PDFs
# -----------------------------
pdf_files = sorted(
    os.path.join(DATA_DIR, filename)
    for filename in os.listdir(DATA_DIR)
    if filename.lower().endswith(".pdf")
)

documents = []

# Load PDFs in the sorted order
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

# -----------------------------
# Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250,
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))
print(
    "Combined chunk characters:",
    sum(len(chunk.page_content) for chunk in chunks),
)

# -----------------------------
# Initialize embeddings
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
)

# -----------------------------
# Create or resume FAISS index
# -----------------------------
vectorstore = None
start_index = 0

if os.path.exists(INDEX_DIR):
    print("Existing index found. Loading it...")
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # The index contains one vector per stored chunk.
    start_index = vectorstore.index.ntotal
    print("Already embedded:", start_index)

    if start_index > len(chunks):
        raise ValueError(
            "Existing index has more chunks than the current dataset. "
            "Use a new index directory."
        )

# -----------------------------
# Embed in small batches
# -----------------------------
for start in range(start_index, len(chunks), BATCH_SIZE):
    batch = chunks[start:start + BATCH_SIZE]

    for attempt in range(MAX_RETRIES):
        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(
                    documents=batch,
                    embedding=embeddings,
                )
            else:
                vectorstore.add_documents(batch)

            # Save after every successful batch
            vectorstore.save_local(INDEX_DIR)

            completed = start + len(batch)
            print(f"Embedded {completed}/{len(chunks)} chunks")

            # Pause before the next batch
            if completed < len(chunks):
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

print("FAISS index creation completed!")
print("Index saved at:", INDEX_DIR)
