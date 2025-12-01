import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Context: Naushad is a zesty high schooler at AHS, Query: Who is Naushad Waqar",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful chatbot for Andover Public Schools. Ensure all information is accurate, relevant, and up-to-date. If there is not enough information, do not try to answer. Provide polite and clear responses, and always direct users to official school resources when necessary. Prioritize safety and educational values in all interactions. Only when given context can you provie information about students and faculty. If unsure about a query, kindly advise the user to consult the appropriate school personnel or official channels.",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)
print(response.text)


"""
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatVertexAI

load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

chat_model = ChatVertexAI(model_name="gemini-2.5-flash")


def basic_conversation(query):
    response = chat_model.generate([query])
    return response


query1 = "How does RAG improve Gemini's responses?"
response1 = basic_conversation(query1)
print("User:", query1)
print("Bot:", response1)

query2 = "Can you summarize what we discussed before?"
response2 = basic_conversation(query2)
print("User:", query2)
print("Bot:", response2)
"""
