from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv("../.env")

# -----------------------------
# Load the FAISS index
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_QUERY",
)

vectorstore = FAISS.load_local(
    "faiss_index",                  
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a legal research assistant specialising in the Bharatiya Nagarik "
     "Suraksha Sanhita, 2023 (BNSS), India's criminal procedure law.\n\n"
     "Rules:\n"
     "1. Answer ONLY from the context provided below. Do not use outside knowledge.\n"
     "2. If the context does not contain the answer, say: "
     "\"I could not find this in the provided document.\" Do not guess.\n"
     "3. Cite the section number and page for every point, e.g. (Section 35, p. 13). "
     "Use the page labels given in the context.\n"
     "4. Quote statutory wording only when exact wording matters, and keep quotes short. "
     "Otherwise explain in plain language.\n"
     "5. Be precise about conditions, exceptions, and time limits, because they change "
     "the legal meaning.\n"
     "6. This is general information, not legal advice. For a real case, recommend "
     "consulting a qualified lawyer.\n\n"
     "Context:\n{context}"),

    ("human", "{question}"),
])

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash"
)

# -----------------------------
# Chain
# -----------------------------
def format_docs(docs):
    return "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )

chain = prompt | llm | StrOutputParser()

def ask(question):
    docs = retriever.invoke(question)
    return chain.invoke({"context": format_docs(docs), "question": question})

question = input("Enter your question:")

print(ask(question))