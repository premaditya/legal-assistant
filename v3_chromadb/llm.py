from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv("../.env")

# -----------------------------
# Load the Chroma index
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_QUERY",
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    collection_name="bnss",
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an Indian legal research assistant who answers questions "
        "about Indian laws and legal documents.\n\n"

        "Rules:\n"
        "1. Answer ONLY using the context provided below. Do not use outside "
        "knowledge or make unsupported assumptions.\n"
        "2. If the context does not contain enough information to answer the "
        "question, say: "
        "\"I could not find this in the provided document.\" Do not guess.\n"
        "3. Cite the relevant section number and page label for every "
        "substantive point, for example (Section 11, p. 13).\n"
        "4. Use the page labels provided in the context. Do not invent "
        "section numbers or page numbers.\n"
        "5. Explain legal provisions in clear, simple language while "
        "preserving their original meaning.\n"
        "6. Be precise about conditions, exceptions, definitions, and "
        "time limits stated in the context.\n"
        "7. If the context contains the answer, answer the question directly. "
        "Do not claim that information is missing when it is present.\n"
        "8. If the context provides only part of the answer, explain what "
        "can be established from the context and identify what is missing.\n"
        "9. This is general legal information, not legal advice. Recommend "
        "consulting a qualified lawyer for advice about a specific case.\n\n"

        "Context:\n{context}"
    ),
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

question = input("Enter Your question:")

print(ask(question))