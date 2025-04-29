from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel

from src.embeddings.vector import retriever
from src.utils.db_services import ChatSession, Message, mongo_manager

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
    session_id: str


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        return templates.TemplateResponse(
            name="index.html", context={"request": request}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    question: str = Form(...),
    session_id: Optional[str] = Header(None, alias="Session-ID"),
):
    try:
        if question.lower() in ("e", "exit", "bye"):
            return AnswerResponse(answer="Goodbye! 👋", session_id=session_id or "")

        # Generate response
        data = retriever.invoke(question)
        response = chain.invoke({"data": data, "question": question})

        # Create messages
        user_message = Message(content=question, is_user=True)
        bot_message = Message(content=response, is_user=False)

        # Handle session
        if not session_id:
            # Create new session
            new_session = ChatSession(messages=[user_message, bot_message])
            session_id = await mongo_manager.create_chat_session(new_session)
        else:
            # Update existing session
            await mongo_manager.update_chat_session(session_id, user_message)
            await mongo_manager.update_chat_session(session_id, bot_message)

        return AnswerResponse(answer=response, session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_chat_session(session_id: str):
    session = await mongo_manager.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions", response_model=List[str])
async def get_all_sessions():
    """Get all session IDs from MongoDB"""
    sessions = await mongo_manager.chat_history.distinct("session_id")
    return sessions
