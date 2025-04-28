from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel

from src.embeddings.vector import retriever

BASE_DIR = Path(__file__).parent.parent

router = APIRouter()
templates = Jinja2Templates(directory=BASE_DIR / "templates")
# Initialize LLM and prompt template
model = OllamaLLM(model="llama3")

template = """
You will be an assistant for any questions about Ebla Computer Consultancy

Here are some relevant answers: {data}

Here is the answer to your question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


class AnswerResponse(BaseModel):
    answer: str


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        return templates.TemplateResponse(
            name="index.html", context={"request": request}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(question: str = Form(...)):
    try:
        """Endpoint to ask questions about Ebla Computer Consultancy"""
        if question.lower() == "e" or question.lower() == "bye":
            return AnswerResponse(answer="Goodbye! 👋")

        # Retrieve relevant data and generate response
        data = retriever.invoke(question)
        response = chain.invoke({"data": data, "question": question})

        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
