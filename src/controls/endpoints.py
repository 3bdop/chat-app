from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig

from src.embeddings.vector import retriever
from src.models.db_services import ChatSession, Message, mongo_manager

BASE_DIR = Path(__file__).parent.parent

router = APIRouter(tags=["RAG + Semantic-Kernel"])
templates = Jinja2Templates(directory=BASE_DIR / "view/templates")
# ----------------------RAG----------------------#
# Initialize LLM and prompt template
model = OllamaLLM(model="llama3")

rag_template = """
Your name is RAG and you are an assistant for Ebla Computer Consultancy. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know, don't try to make up an answer.

Relevant Context: {data}

Chat History (most recent first): {history}

Current Question: {question}
"""

prompt = ChatPromptTemplate.from_template(rag_template)
chain = prompt | model
# ----------------------RAG----------------------#

# ----------------------SK-----------------------#
kernel = Kernel()
chat_service = OllamaChatCompletion(ai_model_id="llama3", host="http://localhost:11434")
kernel.add_service(chat_service)

# Configure prompt template
sk_template = """
Your name is Semantic Kernel and you are an assistant for Ebla Computer Consultancy. 
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know, don't try to make up an answer.

Relevant Context: {{$data}}

Chat History:
{{$history}}

Current Question: {{$input}}

"""

# Create semantic function
prompt_config = PromptTemplateConfig(
    template=sk_template,
    template_format="semantic-kernel",
    input_variables=[
        {"name": "data", "description": "Relevant context"},
        {"name": "question", "description": "User question"},
        {"name": "history", "description": "Chat history"},
    ],
)

assistant_function = kernel.add_function(
    function_name="ebla_assistant",
    plugin_name="EblaPlugin",
    prompt_template_config=prompt_config,
)
# ----------------------SK-----------------------#


class AnswerResponse(BaseModel):
    answer: str
    session_id: str


# ------------------------CHAT APP UI------------------------#
@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        return templates.TemplateResponse(
            name="index.html", context={"request": request}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------RAG OLLAMA------------------------#
@router.post("/ask-rag", response_model=AnswerResponse)
async def ask_rag_question(
    question: str = Form(...),
    session_id: Optional[str] = Header(None, alias="Session-ID"),
):
    try:
        if question.lower() in ("e", "exit", "bye"):
            return AnswerResponse(answer="Goodbye! 👋", session_id=session_id or "")

        # Get history if chat session exists
        history_context = ""
        if session_id:
            history = await mongo_manager.get_chat_history_dict(session_id)
            if history:
                # Format history into conversation context
                history_pairs = zip(history["questions"], history["answers"])
                history_context = "\n".join(
                    [
                        f"Previous Question: {q['content']}\nPrevious Answer: {a['content']}"
                        for q, a in history_pairs
                    ]
                )

        # Generate response
        data = retriever.invoke(question)
        response = chain.invoke(
            {"data": data, "question": question, "history": history_context}
        )

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


# ------------------------SEMANTIC KERNEL OLLAMA------------------------#
@router.post("/ask-sk", response_model=AnswerResponse)
async def ask_sk_question(
    question: str = Form(...),
    session_id: Optional[str] = Header(None, alias="Session-ID"),
):
    try:
        if question.lower() in ("e", "exit", "bye", "q", "quit"):
            return AnswerResponse(answer="Goodbye! 👋", session_id=session_id or "")

        history_context = ""
        if session_id:
            history = await mongo_manager.get_chat_history_dict(session_id)
            if history:
                history_pairs = zip(history["questions"], history["answers"])
                history_context = "\n".join(
                    [
                        f"User: {q['content']}\nAssistant: {a['content']}"
                        for q, a in history_pairs
                    ]
                )

        data = retriever.invoke(question)  # Your existing retriever

        # Prepare arguments
        arguments = KernelArguments(data=data, input=question, history=history_context)

        # Get response
        response = await kernel.invoke(
            function=assistant_function,
            arguments=arguments,
        )

        user_message = Message(content=question, is_user=True)
        bot_message = Message(content=str(response), is_user=False)

        # Update chat history
        if not session_id:
            new_session = ChatSession(messages=[user_message, bot_message])
            session_id = await mongo_manager.create_chat_session(new_session)
        else:
            await mongo_manager.update_chat_session(session_id, user_message)
            await mongo_manager.update_chat_session(session_id, bot_message)

        return AnswerResponse(answer=str(response), session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------SEMANTIC KERNEL OLLAMA------------------------#


# ---------------------------CHAT APP GENERAL---------------------------#
@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_chat_session(session_id: str):
    session = await mongo_manager.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}", response_model=ChatSession)
async def delete_chat_session(session_id: str):
    success = await mongo_manager.delete_chat_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "message": "chat session is delete successfully"}


@router.get("/sessions", response_model=List[str])
async def get_all_sessions():
    """Get all session IDs from MongoDB"""
    sessions = await mongo_manager.chat_history.distinct("session_id")
    return sessions


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    history = await mongo_manager.get_chat_history_dict(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    return history


# ---------------------------CHAT APP GENERAL---------------------------#
