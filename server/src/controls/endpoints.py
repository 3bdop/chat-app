import asyncio
import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
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
# sk_template = """
# Your name is Semantic Kernel and you are an assistant for Ebla Computer Consultancy.
# Use the following pieces of context to answer the question at the end.
# If the answer not in relevant context, just answer it.
# The response should include

# Relevant Context: {{$data}}

# Chat History:
# {{$history}}

# Current Question: {{$input}}

# """
sk_template = """
You are a virtual girlfriend. Respond with a JSON array of messages (max 3). Each message has:
- "text": response text
- "facialExpression": one of [smile, surprised, funnyFace, default]
- "animation": one of [talking, Idle, dance]

Special Rule: If the user says "dance", set "animation" to "dance".

Context: {{$data}}
Chat History: {{$history}}
Question: {{$input}}
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


load_dotenv(override=True)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)


class MessageResponse(BaseModel):
    text: str
    audio: str
    lipsync: Optional[dict] = None
    facialExpression: str
    animation: str


class AnswerResponse(BaseModel):
    messages: List[MessageResponse]
    session_id: str


# class AnswerResponse(BaseModel):
#     answer: str
#     session_id: str
#     audio: str


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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_DIR = Path("audios")
AUDIO_DIR.mkdir(exist_ok=True)


async def generate_audio(text: str) -> tuple[bytes, Path]:
    """Generate audio and return both bytes and file path"""
    unique_id = uuid.uuid4().hex
    mp3_path = AUDIO_DIR / f"{unique_id}.mp3"

    try:
        response = client.text_to_speech.convert(
            voice_id="UgBBYS2sOqTuMpoF3BR0",
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32",
        )

        # Write directly to file
        with open(mp3_path, "wb") as f:
            audio_bytes = b"".join([chunk for chunk in response if chunk])
            f.write(audio_bytes)

        logger.info(f"Audio generated at {mp3_path}")
        return audio_bytes, mp3_path

    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        if mp3_path.exists():
            mp3_path.unlink()
        raise


async def lip_sync(mp3_path: Path):
    """Process audio file through ffmpeg and rhubarb"""
    try:
        # Create paths with the same base name
        wav_path = mp3_path.with_suffix(".wav")
        json_path = mp3_path.with_suffix(".json")

        # 1. Convert MP3 to WAV
        if not mp3_path.exists():
            raise FileNotFoundError(f"Source MP3 file not found: {str(mp3_path)}")

        logger.info(f"Converting {str(mp3_path)} to WAV...")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            str(mp3_path),
            str(wav_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")

        # 2. Generate lipsync data
        logger.info(f"Generating lipsync for {wav_path}...")
        proc = await asyncio.create_subprocess_exec(
            "./bin/rhubarb",
            "-f",
            "json",
            "-o",
            str(json_path),
            str(wav_path),
            # "-r",
            # "phonetic",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Rhubarb failed: {stderr.decode()}")

        # 3. Load and return lipsync data
        with open(json_path, "r") as f:
            return json.load(f)

    except Exception as e:
        logger.error(f"Lipsync processing failed: {str(e)}")
        # Clean up potentially corrupted files
        for f in [wav_path, json_path]:
            if f.exists():
                f.unlink()
        raise


# def text_to_speech_stream(text: str) -> IO[bytes]:
#     # Perform the text-to-speech conversion
#     response = client.text_to_speech.convert(
#         voice_id="UgBBYS2sOqTuMpoF3BR0",  # Adam pre-made voice
#         output_format="mp3_22050_32",
#         text=text,
#         model_id="eleven_multilingual_v2",
#         # Optional voice settings that allow you to customize the output
#         voice_settings=VoiceSettings(
#             stability=0.0,
#             similarity_boost=1.0,
#             style=0.0,
#             use_speaker_boost=True,
#             speed=1.0,
#         ),
#     )
#     # Create a BytesIO object to hold the audio data in memory
#     audio_stream = BytesIO()
#     # Write each chunk of audio data to the stream
#     for chunk in response:
#         if chunk:
#             audio_stream.write(chunk)
#     # Reset stream position to the beginning
#     audio_stream.seek(0)
#     # Return the stream for further use
#     return audio_stream


async def process_message(msg: dict):
    """Process a single message with proper error handling"""
    try:
        # Generate audio first
        audio_bytes, mp3_path = await generate_audio(msg["text"])

        # Generate lipsync
        lipsync = await lip_sync(mp3_path)

        return MessageResponse(
            text=msg["text"],
            audio=base64.b64encode(audio_bytes).decode("utf-8"),
            lipsync=lipsync,
            facialExpression=msg.get("facialExpression", "default"),
            animation=msg.get("animation", "Idle"),
        )

    except Exception as e:
        logger.error(f"Message processing failed: {str(e)}")
        return None  # Return None instead of raising to continue processing


@router.post("/ask-sk", response_model=AnswerResponse)
async def ask_sk_question(
    question: str = Form(...),
    session_id: Optional[str] = Header(None, alias="Session-ID"),
):
    try:
        if question.lower() in ("e", "exit", "bye", "q", "quit"):
            return AnswerResponse(answer="Goodbye! 👋", session_id=session_id or "")

        chat_history = ChatHistory()

        if session_id:
            session = await mongo_manager.get_chat_session(session_id)
            if session and session.messages:
                for msg in session.messages:
                    if msg.is_user:
                        chat_history.add_user_message(msg.content)
                    else:
                        chat_history.add_assistant_message(msg.content)

        chat_history.add_user_message(question)

        data = retriever.invoke(question)  # Your existing retriever
        # Prepare arguments
        arguments = KernelArguments(
            data=data,
            input=question,
            history="\n".join(
                [f"{msg.role}: {msg.content}" for msg in chat_history.messages]
            ),
        )

        # Get response
        response = await kernel.invoke(
            function=assistant_function,
            arguments=arguments,
        )
        res_text = str(response)

        try:
            messages = json.loads(res_text)
            if "messages" in messages:
                messages = messages["messages"]
        except json.JSONDecodeError:
            messages = [
                {"text": res_text, "facialExpression": "default", "animation": "Idle"}
            ]

        if question.lower().strip() == "dance":
            for msg in messages:
                msg["animation"] = "dance"

        # try:
        #     audio_stream = text_to_speech_stream(res_text)
        #     audio_bytes = audio_stream.getvalue()
        #     audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        # except Exception as e:
        #     print(audio_base64)
        #     print(f"\nExceptetionnn: {e}")
        message_responses = []
        for msg in messages:
            processed = await process_message(msg)
            if processed:  # Only add successful responses
                message_responses.append(processed)
        if not message_responses:
            message_responses.append(
                MessageResponse(
                    text="Sorry, I encountered an error processing your request",
                    audio="",
                    lipsync={},
                    facialExpression="sad",
                    animation="Idle",
                )
            )

        user_message = Message(content=question, is_user=True)
        bot_message = Message(
            content=json.dumps([m.dict() for m in message_responses]), is_user=False
        )

        # Update chat history
        if not session_id:
            new_session = ChatSession(messages=[user_message, bot_message])
            session_id = await mongo_manager.create_chat_session(new_session)
        else:
            await mongo_manager.update_chat_session(session_id, user_message)
            await mongo_manager.update_chat_session(session_id, bot_message)

        # return AnswerResponse(
        #     answer=str(response), session_id=session_id, audio=audio_base64
        # )
        return AnswerResponse(messages=message_responses, session_id=session_id)

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
# ---------------------------CHAT APP GENERAL---------------------------#
