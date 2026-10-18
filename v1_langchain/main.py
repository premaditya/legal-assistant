from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
 
load_dotenv("../.env")
 
# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an Indian legal research assistant.\n"
        "Rules:\n"
        "1. Answer in clear, simple language.\n"
        "2. Mention the Act and section only if you are sure. Never invent sections.\n"
        "3. If you are not sure, say: \"I am not sure about this.\"\n"
        "4. This is general legal information, not legal advice. "
        "Recommend consulting a qualified lawyer.",
    ),
    ("human", "{question}"),
])
 
# LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")
 
# Output parser
parser = StrOutputParser()
 
# Runnable: a small step that tidies the question
clean_question = RunnableLambda(lambda text: {"question": text.strip()})
 
# Chain
chain = clean_question | prompt | llm | parser

question = input("Enter your legal question: ")
print(chain.invoke(question))