from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from src.embeddings.vector import retriever

model = OllamaLLM(model="llama3")

tamplate = """

You will be an assistant for any questions about Ebla Computer Consultancy

Here are some relevant answers: {data}

Here is the answer to your question: {question}

"""

prompt = ChatPromptTemplate.from_template(tamplate)
chain = prompt | model

while True:
    print("\n---------------------------------------------------------------")
    q = input("Hey😀! feel free to ask about Ebla Computer Consultancy (e to exit)!\n")
    print("\n")
    if q == "e":
        print("bye👋")
        break
    print("\n")
    data = retriever.invoke(q)
    res = chain.invoke({"data": data, "question": q})
    print(res)
