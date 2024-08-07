import chromadb
from flask import Flask, redirect, render_template, request, url_for
from termcolor import colored

from utils import openai_chat, request_to_embeddings

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chatgpt", methods=["GET", "POST"])
def chatgpt():
    if request.method == "POST":
        question = request.form["question"]
        # logica para procesar la regunta y obtener la respuesta
        answer = openai_chat.simple_question(question)
        print(colored(f"\n\nanswer: {answer}\n\n", "red"))
        return redirect(url_for("chatgpt", question=question, answer=answer))
    question = request.args.get("question")
    answer = request.args.get("answer")
    return render_template("chatgpt.html", question=question, answer=answer)


@app.route("/chat_embeddings", methods=["GET", "POST"])
def chat_embeddings():
    if request.method == "POST":
        question = request.form["question"]
        print(colored(f"{question}", "red"))
        # logica para procesar la regunta y obtener la respuesta
        persistent_client = chromadb.PersistentClient(path="./vectordb/")
        answer = request_to_embeddings.get_response(question)

        print(colored(f"\n\n[+]Answer: {answer}\n\n", "blue"))
        return redirect(url_for("chat_embeddings", question=question, answer=answer))
    question = request.args.get("question")
    answer = request.args.get("answer")
    return render_template("chat_embeddings.html", question=question, answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
