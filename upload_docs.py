
# Set OpenAI key in the environment
from dotenv import load_dotenv
import os

load_dotenv()

from openai import OpenAI

client = OpenAI()


# Define the place where the Docs are and extensions
docs_directory = "Docs"
supported_extensions = [".txt", ".py"]


# Append all file names and paths together
files_path = []

for filename in os.listdir(docs_directory):
    file_path = os.path.join(docs_directory, filename)
    if os.path.isfile(file_path):
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in supported_extensions:
            files_path.append(file_path)

# Upload files to OpenAI
file_ids = []

for file_path in files_path:
    with open(file_path, "rb") as file:
        response = client.files.create(
            file=file,
            purpose="assistants"
        )
        file_ids.append(response.id)


# Add files to the vector store
vector_store_id = os.getenv("VECTOR_STORE_ID")

for file_id in file_ids:
    result = client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_id
    )
