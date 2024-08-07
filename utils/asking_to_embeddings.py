import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from termcolor import colored

load_dotenv()

os.environ["OPENAI_API_KEY"] = str(os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
chroma_local = Chroma(
    persist_directory="../vectordb/", embedding_function=OpenAIEmbeddings()
)


def create_prompt_template(texto):
    system_prompt = texto + "\n\n{context}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    print(colored(f"\n\n[+] prompt: {prompt}", "red"))
    return prompt


def get_response(pregunta, llm, chroma_db, prompt_template):
    retriever = chroma_db.as_retriever()
    chain = create_stuff_documents_chain(llm, prompt_template)
    rag = create_retrieval_chain(retriever, chain)

    results = rag.invoke({"input": pregunta})
    print(colored(f"\n\n[+] results {results}\n", "red"))
    return results


texto = """Tú eres un asistente para tareas de respuesta a preguntas.
    Usa los siguientes fragmentos de contexto recuperado para responder 
    la pregunta. Si no sabes la respuesta, di que no 
    sabes. Usa un máximo de tres oraciones y mantén la 
    respuesta concisa."""


def pregunta(query):
    prompt_template = create_prompt_template(texto)
    print(colored(f"\n\n[+] prompt template: {prompt_template}"))
    return get_response(query, llm, chroma_local, prompt_template)["answer"]


# Ejemplo de uso cuando se ejecuta directamente
if __name__ == "__main__":
    pregunta_ejemplo = "como se carga el bebe?"
    respuesta = pregunta(pregunta_ejemplo)
    print(respuesta)
