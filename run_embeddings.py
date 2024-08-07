from utils import openai_embeddigs

response = openai_embeddigs.vectordb_upload(
    "./docs_to_process/Manual de uso simuladores alta fidelidad.pdf"
)
print(response)
