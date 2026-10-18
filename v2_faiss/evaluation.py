
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv("../.env")

LANGSMITH_PROJECT = "v2_FAISS"

# Configure LangSmith tracing for this script
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGSMITH_TRACING"] = "true"

client = Client()


# --------------------------------------------------
# Output File
# --------------------------------------------------

output_file = Path("v2_evaluation_result.txt")

contents = output_file.read_text(encoding="utf-8") if output_file.exists() else ""


# --------------------------------------------------
# Evaluation Questions
# --------------------------------------------------

questions = [
    "What are the essential elements of a valid contract under the Indian Contract Act, 1872?",
    "How does the Indian Contract Act define a proposal and an acceptance?",
    "Can a contract be legally valid if it is made verbally rather than in writing under Indian law?",
    "What is the difference between a void agreement and a voidable contract under the Indian Contract Act?",
    "Who is competent to contract under Section 11 of the Indian Contract Act?",
    "What is free consent, and how does coercion affect the validity of a contract?",
    "What is the difference between fraud and misrepresentation under the Indian Contract Act?",
    "What is the difference between liquidated damages and a penalty under Section 74 of the Indian Contract Act?",
    "What is the difference between a contract of indemnity and a contract of guarantee?",
    "A person borrows money from a friend and promises to repay it in three months but later refuses. What provisions of the Indian Contract Act apply, and what evidence can establish the agreement under the Bharatiya Sakshya Adhiniyam, 2023?",
    "A buyer pays an advance for goods but the seller does not deliver. What remedies are available under the Indian Contract Act and the Consumer Protection Act, 2019?",
    "What is the definition of 'consumer' under the Consumer Protection Act, 2019?",
    "What are the rights of a Data Principal under the Digital Personal Data Protection Act, 2023?",
    "What does the Bharatiya Sakshya Adhiniyam, 2023 say about the admissibility of electronic records?",
    "How does the Bharatiya Nyaya Sanhita, 2023 define cheating, and what is the punishment?",
    "What is the procedure for registering an FIR under the Bharatiya Nagarik Suraksha Sanhita, 2023?",
    "What is the procedure for obtaining a divorce under the Hindu Marriage Act, 1955?",
    "What are the legal requirements for registering a private limited company under the Companies Act, 2013?",
    "What is the punishment for cyber fraud under the Information Technology Act, 2000?",
    "Who won the 2011 Cricket World Cup?",
]


# --------------------------------------------------
# Prompt Template
# --------------------------------------------------

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


# --------------------------------------------------
# LLM and Chain
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash"
)

parser = StrOutputParser()

chain = prompt | llm | parser


# --------------------------------------------------
# Embeddings and FAISS Vector Store
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_QUERY"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)


# --------------------------------------------------
# LangSmith Metrics Retrieval
# --------------------------------------------------

def get_langsmith_metrics(run_name):
    """Retrieve the trace and LLM metrics for one question."""

    try:
        # Allow some time for the trace to become available
        time.sleep(2)

        runs = list(
            client.list_runs(
                project_name=LANGSMITH_PROJECT,
                filter=f'eq(name, "{run_name}")',
                limit=1
            )
        )

        if not runs:
            return {
                "status": "Trace not found",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "estimated_cost": None,
                "llm_latency": None,
            }

        root_run = runs[0]

        trace = client.read_run(
            root_run.id,
            load_child_runs=True
        )

        # Find the first LLM child run
        llm_run = None

        for child in trace.child_runs or []:
            if child.run_type == "llm":
                llm_run = child
                break

        if llm_run is None:
            return {
                "status": "LLM metrics unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "estimated_cost": None,
                "llm_latency": None,
            }

        # Read token usage from LangSmith
        input_tokens = llm_run.prompt_tokens
        output_tokens = llm_run.completion_tokens
        total_tokens = llm_run.total_tokens
        estimated_cost = llm_run.total_cost

        # Calculate LLM latency using LangSmith timestamps
        llm_latency = None

        if llm_run.start_time and llm_run.end_time:
            llm_latency = (
                llm_run.end_time - llm_run.start_time
            ).total_seconds()

        return {
            "status": "Success",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
            "llm_latency": llm_latency,
        }

    except Exception as e:
        print(f"LangSmith retrieval error: {e}")

        return {
            "status": f"Retrieval failed: {e}",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "estimated_cost": None,
            "llm_latency": None,
        }


# --------------------------------------------------
# Run Evaluation
# --------------------------------------------------

count = 0

for index, question in enumerate(questions, start=1):

    if question not in contents:

        if count%5 == 0:

            time.sleep(90)

        print(f"\nProcessing Question {index}/{len(questions)}")

        # Retrieve relevant documents from FAISS
        results = vectorstore.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            doc.page_content for doc in results
        )

        # Unique run name for this question
        run_name = f"v2_FAISS Question {index}"

        # Measure application latency
        start_time = time.perf_counter()

        result = chain.invoke(
            {
                "context": context,
                "question": question,
            },
            config={
                "run_name": run_name
            }
        )

        end_time = time.perf_counter()

        app_latency = end_time - start_time

        # Retrieve LangSmith metrics
        metrics = get_langsmith_metrics(run_name)

        # --------------------------------------------------
        # Save Answer and Metrics
        # --------------------------------------------------

        with output_file.open("a", encoding="utf-8") as file:

            file.write(f"Question {index}: {question}\n\n")

            file.write(f"Answer:\n{result}\n\n")

            file.write("-" * 80 + "\n")
            file.write("LANGSMITH REQUEST METRICS\n")
            file.write("-" * 80 + "\n")

            file.write(f"Project: {LANGSMITH_PROJECT}\n")
            file.write(f"Run Name: {run_name}\n")
            file.write(f"Trace Status: {metrics['status']}\n")
            file.write(f"App Latency: {app_latency:.2f} seconds\n")

            if metrics["input_tokens"] is not None:
                file.write(
                    f"Input Tokens: {metrics['input_tokens']}\n"
                )

            if metrics["output_tokens"] is not None:
                file.write(
                    f"Output Tokens: {metrics['output_tokens']}\n"
                )

            if metrics["total_tokens"] is not None:
                file.write(
                    f"Total Tokens: {metrics['total_tokens']}\n"
                )

            if metrics["estimated_cost"] is not None:
                file.write(
                    "Estimated Cost: "
                    f"${float(metrics['estimated_cost']):.6f}\n"
                )

            if metrics["llm_latency"] is not None:
                file.write(
                    f"LLM Latency: {metrics['llm_latency']:.2f} seconds\n"
                )

            file.write("=" * 80 + "\n\n")

        contents += f"Question {index}: {question}\n"

        print(f"Question {index} completed.")
        print(f"App Latency: {app_latency:.2f} seconds")
        print(f"LangSmith Status: {metrics['status']}")

        count += 1


print("\nEvaluation completed.")
print(f"Results saved to: {output_file.resolve()}")
print(f"LangSmith Project: {LANGSMITH_PROJECT}")
