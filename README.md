# ⚖️ Legal Assistant — RAG for Indian Laws (V1–V3)

An AI-powered legal question-answering project built using LangChain and Google Gemini. This repository contains three stages of development, starting with a basic LLM chain and progressing to Retrieval-Augmented Generation (RAG) using FAISS and ChromaDB.

The project uses Indian legal documents to answer questions in plain language and retrieve relevant legal information.

**Final Version (V4):** [Legal Assistant V4 — Reranking](https://github.com/premaditya/legal-assistant-v4)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [Project Versions](#project-versions)
- [Documents Used](#documents-used)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)

---

## Problem Statement

Indian laws contain lengthy legal terminology and detailed provisions that can be difficult to understand without a legal background. Finding relevant information across multiple legal documents can also take time.

This project explores how Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) can help users find relevant information from Indian legal documents.

## Proposed Solution

The project is developed in three stages:

1. Build a basic LangChain chain to understand how prompts, LLMs and output parsers work.
2. Implement RAG using FAISS to retrieve relevant passages from legal documents.
3. Use ChromaDB to store document embeddings and metadata for improved document retrieval and source tracking.

The final version, V4, extends this work by adding reranking, a Streamlit interface and LangSmith monitoring.

## Project Versions

| Version | Implementation | Description |
|---|---|---|
| V1 | LangChain | Basic LLM chain using prompts, an LLM and an output parser. |
| V2 | FAISS | Implements RAG using PDF loading, text splitting, embeddings, FAISS retrieval and an LLM. |
| V3 | ChromaDB | Uses ChromaDB for vector storage and maintains document source and page metadata. |

### Development Flow

```text
V1: Prompt → LLM → Output Parser
                  ↓
V2: Documents → Embeddings → FAISS → Retrieval → LLM
                  ↓
V3: Documents → Embeddings → ChromaDB → Retrieval → LLM
                  ↓
V4: Retrieval → Reranking → LLM → Streamlit App
```

## Documents Used

The project uses Indian legal documents, including:

- Indian Contract Act, 1872
- Bharatiya Nyaya Sanhita, 2023
- Bharatiya Nagarik Suraksha Sanhita, 2023
- Bharatiya Sakshya Adhiniyam, 2023
- Consumer Protection Act, 2019
- Digital Personal Data Protection Act, 2023

Place the required PDF files in the `data/` directory. Make sure you have permission to redistribute any documents you upload.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM Framework | LangChain |
| Language Model | Google Gemini |
| Embeddings | Google Gemini Embeddings |
| V2 Vector Store | FAISS |
| V3 Vector Store | ChromaDB |
| Document Processing | LangChain document loaders and text splitters |

## Project Structure

```text
legal-assistant/
│
├── data/
│   └── Legal PDF documents
│
├── v1_langchain/
│   └── main.py
│
├── v2_faiss/
│   ├── faiss_index/             
│   ├── evaluation.py
│   ├── llm.py
│   ├── main.py
│   ├── retrieval.py
│   └── v2_evaluation_result.txt
│
├── v3_chromadb/
│   ├── chroma_db/               
│   ├── evaluation.py
│   ├── llm.py
│   ├── main.py
│   ├── retrieval.py
│   └── v3_evaluation_result.txt
│
├── .env                         
├── .gitignore
└── requirements.txt
```

The `faiss_index/` and `chroma_db/` directories are generated locally and are not included in this repository.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/premaditya/legal-assistant.git
cd legal-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If your root `requirements.txt` does not include all dependencies required by V2 and V3, install the missing packages as needed.

### 4. Configure API keys

Create a `.env` file in the project root and add your Google API key:

```text
GOOGLE_API_KEY=your-google-api-key
```

Use the exact environment variable names required by your code. Never commit your actual API keys to GitHub.

### 5. Add the legal documents

Place the required PDF documents in the `data/` directory, ensuring their filenames and paths match those used in the Python scripts.

## How to Run

Run the commands from the repository root unless your scripts require a different working directory.

### V1 — LangChain Basics

```bash
cd v1_langchain
python main.py
```

### V2 — FAISS RAG

First, build the vector index:

```bash
cd v2_faiss
python main.py
```

Then ask questions or test retrieval:

```bash
python retrieval.py
python llm.py
```

Run the evaluation:

```bash
python evaluation.py
```

### V3 — ChromaDB RAG

First, build the vector database:

```bash
cd v3_chromadb
python main.py
```

Then run retrieval, question answering or evaluation:

```bash
python retrieval.py
python llm.py
python evaluation.py
```

**Note:** The vector databases must be generated before running the retrieval and question-answering scripts. Check each version's code for the expected database path and working directory.

## Evaluation

V2 and V3 were evaluated using a set of 20 questions covering Indian legal documents.

The evaluation scripts record generated answers and performance information in text files.

The saved evaluation results are available in:

- `v2_faiss/v2_evaluation_result.txt`
- `v3_chromadb/v3_evaluation_result.txt`

These files can be used to compare the versions' answer quality, retrieval behavior and response times.

## Limitations

- Answers depend on the quality and coverage of the provided legal documents.
- Retrieval may fail to identify the most relevant passage for some questions.
- LLM-generated answers may be incomplete or incorrect.
- Vector databases must be generated locally before using V2 and V3.

## Final Version

V4 builds on these implementations by adding a reranking stage, a Streamlit web interface and LangSmith monitoring.

**Repository:** [premaditya/legal-assistant-v4](https://github.com/premaditya/legal-assistant-v4)

## Disclaimer

This project provides general legal information for educational and informational purposes only. It does not provide legal advice. Always verify important information against official legal sources and consult a qualified legal professional for advice about a specific situation.
