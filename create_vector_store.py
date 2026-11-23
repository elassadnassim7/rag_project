import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

vector_store = client.vector_stores.create(
    name = "My_RAG_Vector_Store"
)

print("Vector store created successfully")
print(f"Vector store ID: {vector_store.id}")
