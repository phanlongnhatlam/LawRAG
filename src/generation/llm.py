from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
_llm_instance = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.1,
)

def get_llm():
    return _llm_instance
