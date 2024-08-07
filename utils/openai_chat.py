import os

from dotenv import load_dotenv
from openai import OpenAI
from termcolor import colored

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def simple_question(question):

    print(colored("\n\n[+]-------standar request------", "blue"))
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
    )
    return completion.choices[0].message.content
