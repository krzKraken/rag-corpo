import os

import chromadb
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from termcolor import colored

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


persistent_client = chromadb.PersistentClient(path="../vectordb/")


def vectordb_upload(file_path):
    file_path = file_path
    loader = PyPDFLoader(file_path)
    doc = loader.load()
    print(colored(f"\n\n[+]File: {file_path} fue leido exitosamente..\n", "blue"))
    print(colored(f"\n\n[+]{len(doc)}", "blue"))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )  # el text splitter nos va a permitir partir los documentos con el tamaño de tokens que queramos

    splits = text_splitter.split_documents(doc)  # partimos el pdf en mas partes

    vectorstore = Chroma.from_documents(
        documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./vectordb"
    )  # creamos la base de datos de vectores

    retriever = (
        vectorstore.as_retriever()
    )  # unimos todo y damos la posibilidad de hacer consultas

    response = retriever.get_relevant_documents("who to turn on lucina")
    return response
