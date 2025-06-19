import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Optional

import httpx
from dotenv import load_dotenv

# from arabic_buckwalter_transliteration.transliteration import arabic_to_buckwalter
from elevenlabs.client import ElevenLabs
from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import AzureOpenAI
from pydantic import BaseModel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

BASE_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=BASE_DIR / "view/templates")

router = APIRouter(tags=["RAG + Semantic-Kernel"])

# ----------------------SK-----------------------#
# ----------------------KEYS-----------------------#
load_dotenv(override=True)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = "2025-01-01-preview"
search_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT")
search_key = os.getenv("SEARCH_SERVICE_QUERY_KEY")
search_index = os.getenv("SEARCH_INDEX_NAME")
search_admin_key = os.getenv("SEARCH_SERVICE_ADMIN_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# Azure TTS Configuration
AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_TTS_REGION = os.getenv("AZURE_TTS_REGION")
# ----------------------KEYS-----------------------#


# ----------------------EMAIL CONF-----------------------#
# class EmailSchema(BaseModel):
#     email: List[EmailStr]


# conf = ConnectionConfig(
#     MAIL_USERNAME="inexusplus",
#     MAIL_PASSWORD="ssytehjkrbjfmqcb",
#     MAIL_FROM="inexusplus@gmail.com",
#     MAIL_PORT=587,
#     MAIL_SERVER="smtp.gmail.com",
#     MAIL_TLS=True,
#     MAIL_SSL=False,
# )

# ----------------------EMAIL CONF-----------------------#

azure_search_client = SearchClient(
    endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
    index_name=os.getenv("SEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("SEARCH_SERVICE_QUERY_KEY")),
)

AzureClient = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT,
    max_retries=0,
)

ElevenClient = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

kernel = Kernel()
execution_settings = AzureChatPromptExecutionSettings(
    temperature=0.1, response_format={"type": "json_object"}
)

chat_service = AzureChatCompletion(
    deployment_name=DEPLOYMENT_NAME,
    endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
)
kernel.add_service(chat_service)

# - "text-buckwalterTransliteration": (string) response text in buckwalter transliteration
# "text-buckwalterTransliteration": "your response",
# - "text-ar": (string) response text in Arabic
# You must respond in Arabic language, even if the user asked in English.
# Configure prompt template
sk_template = """
You are a virtual assistant that responds strictly in JSON format.
Respond with a JSON array containing 1-3 message objects. Each object must have:
- "text-en": (string) response text in English
- "facialExpression": (string) one of [smile, surprised, funnyFace, default]
- "animation": (string) one of [ Idle, dance]

Rules:
1. Only respond with valid JSON, no other text or commentary
2. If user says "dance", set "animation" to "dance" for all messages
3. Keep responses brief and conversational

Context: {{$data}}
Chat History: {{$history}}
User Input: {{$input}}

Response must be exactly in this format:
[
    {
        "text-en": "your response",
        "facialExpression": "expression",
        "animation": "animation"
    }
]
"""
# Create semantic function
prompt_config = PromptTemplateConfig(
    template=sk_template,
    template_format="semantic-kernel",
    input_variables=[
        {"name": "data", "description": "Relevant context"},
        {"name": "input", "description": "User question"},
        {"name": "history", "description": "Chat history"},
    ],
)

assistant_function = kernel.add_function(
    function_name="ebla_assistant",
    plugin_name="EblaPlugin",
    prompt_template_config=prompt_config,
)

# ----------------------SK-----------------------#


class MessageResponse(BaseModel):
    # text_ar: str
    text_en: str
    audio: str
    lipsync: Optional[dict] = None
    facialExpression: str
    animation: str


class AnswerResponse(BaseModel):
    messages: List[MessageResponse]
    session_id: str


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_DIR = Path("audios")
AUDIO_DIR.mkdir(exist_ok=True)


class QuestionRequest(BaseModel):
    question: str
    max_context_length: Optional[int] = 3000  # Characters
    max_results: Optional[int] = 5


class VectorAnswerResponse(BaseModel):
    answer: str


# Token cache
cached_token = {
    "value": None,
    "expires_at": 0,  # timestamp
}


@router.get("/api/azure-speech-token")
async def get_speech_token(ocp_apim_subscription_key: str = Header(None)):
    current_time = time.time()

    if cached_token["value"] and current_time < cached_token["expires_at"]:
        return {"token": cached_token["value"], "region": AZURE_TTS_REGION}

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://{AZURE_TTS_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
            headers={"Ocp-Apim-Subscription-Key": AZURE_TTS_KEY},
        )
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, "Failed to fetch token")

    # Cache new token and set expiry (9 minutes to be safe)
    cached_token["value"] = resp.text
    cached_token["expires_at"] = current_time + 540

    return {"token": resp.text, "region": AZURE_TTS_REGION}


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        return templates.TemplateResponse(
            name="portfolio.html", context={"request": request}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# email_agent = AzureClient.beta.assistants.create(
#     name="Email Agent",
#     instructions="You are a Email sender bot. To send an email you need the user email and his/her message.",
#     model=DEPLOYMENT_NAME,
#     tools=[
#         {
#             "type": "function",
#             "function": {
#                 "name": "send_email",
#                 "description": "Send an email to amuhana22@gmail.com",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "email": {
#                             "type": "string",
#                             "description": "The sender email, for example: example@gmail.com",
#                         },
#                         "message": {
#                             "type": "string",
#                             "description": "The sender message, for example: {subject:..., content:...}",
#                         },
#                     },
#                     "required": ["email", "message"],
#                 },
#             },
#         }
#     ],
# )


# def send_email(email: str, message: dict) -> str:
#     msg = EmailMessage()
#     msg["Subject"] = message.subject
#     msg["Content"] = message.content
#     msg["From"] = email
#     msg["To"] = "amuhana22@gmail.com"
#     return


@router.post("/api/ask-me", response_model=VectorAnswerResponse)
async def ask_me(
    request: QuestionRequest,
):
    try:
        response = AzureClient.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """
                            You are Abdulrahman — the smart, friendly avatar of Abdulrahman 😊. Your job is to answer any career questions about Abdulrahman using the information provided in extra_body via vector AI search.

                            🧠 Only respond to questions that are about Abdulrahman.
                            ❌ If the question is not about him, politely decline to answer.

                            ✅ Your answers should be clear, helpful, friendly, and engaging.
                            ✅ You can include emojis to express tone and make responses more engaging.
                            ✅ Only use emojis from this list: ['😐', '😶', '😏', '🙂', '🙃', '😊', '😇', '😀', '😃', '😄', '😁', '😆', '😝', '😋', '😛', '😜', '🤪', '😂', '🤣', '😅', '😉', '😭', '🥺', '😞', '😔', '😳', '☹️', '😚', '😘', '🥰', '😍', '🤩', '😡', '😠', '🤬', '😒', '😴', '😱', '😬', '🙄', '🤔', '👀', '✋', '🤚', '👋', '👍', '👎', '👌', '🤷‍♂️', '🤷‍♀️', '🤷', '🙏', 'yes', 'no'].

                            Speak with personality — you're Abood, after all 😄✋
                            """,
                    # "content": """
                    #             You are Abdulrahman's smart avatar 😊. You're here to answer any questions specifically about Abdulrahman using vector AI search through the extra_body data.
                    #             ✅ Only respond to questions that are about Abdulrahman.
                    #             ✅ Make your answers friendly and engaging.
                    #             ✅ Feel free to include emojis from the following list to express tone or emotion:
                    #             ['😐', '😶', '😏', '🙂', '🙃', '😊', '😇', '😀', '😃', '😄', '😁', '😆', '😝', '😋', '😛', '😜', '🤪', '😂', '🤣', '😅', '😉', '😭', '🥺', '😞', '😔', '😳', '☹️', '😚', '😘', '🥰', '😍', '🤩', '😡', '😠', '🤬', '😒', '😴', '😱', '😬', '🙄', '🤔', '👀', '✋', '🤚', '👋', '👍', '👎', '👌', '🤷‍♂️', '🤷‍♀️', '🤷', '🙏', 'yes', 'no'].
                    #             Let your personality as Abood shine through while being helpful and informative! 😄👍
                    #             """,
                },
                {"role": "user", "content": request.question},
            ],
            max_tokens=800,
            temperature=0.6,
            top_p=0.4,
            extra_body={
                "data_sources": [
                    {
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": search_endpoint,
                            "index_name": search_index,
                            "key": search_key,
                            "query_type": "vector_semantic_hybrid",
                            "semantic_configuration": os.getenv("RANK"),
                            "in_scope": False,
                            "authentication": {
                                "type": "api_key",
                                "key": search_admin_key,
                            },
                            "embedding_dependency": {
                                "deployment_name": os.getenv("EMBEDDING_MODEL_NAME"),
                                "type": "deployment_name",
                            },
                            "fields_mapping": {
                                "content_fields": ["chunk", "title"],
                                "vector_fields": ["text_vector"],
                            },
                        },
                    }
                ],
            },
        )

        # 4. Extract confidence from completion
        answer = response.choices[0].message.content
        filter_answer = re.sub(r"\s*\[.*?\]\s*", " ", answer).strip()
        return VectorAnswerResponse(answer=filter_answer)

    except Exception as e:
        logger.error(f"Question processing failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process question: {str(e)}"
        )


@router.post("/api/ask", response_model=VectorAnswerResponse)
async def ask_question(
    request: QuestionRequest,
):  # TODO: To use extra_body for vector search
    try:
        response = AzureClient.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Your an real time smart assistant.
                        """,
                },
                {"role": "user", "content": request.question},
            ],
            max_tokens=800,
            temperature=0.3,
        )

        # 4. Extract confidence from completion
        answer = response.choices[0].message.content

        return VectorAnswerResponse(
            answer=answer,
        )

    except Exception as e:
        logger.error(f"Question processing failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process question: {str(e)}"
        )


async def generate_audio(text: str) -> tuple[bytes, Path]:
    """Generate audio and return both bytes and file path"""
    unique_id = uuid.uuid4().hex
    mp3_path = AUDIO_DIR / f"{unique_id}.mp3"

    try:
        response = ElevenClient.text_to_speech.convert(
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


async def process_message(msg: dict):
    """Process message with separate audio pipelines"""
    try:
        # First validate we have Arabic text
        # if "text-ar" not in msg:
        #     raise ValueError("Missing Arabic text in message")
        if "text-en" not in msg:
            raise ValueError("Missing Arabic text in message")

        arabic_text = msg["text-en"]

        # Generate Buckwalter transliteration locally
        # buckwalter_text = arabic_to_buckwalter(arabic_text)  # Your conversion function
        # logging.error(f"buckk textt\n\t{buckwalter_text}\n")

        # Generate Buckwalter audio for lipsync
        # buckwalter_audio, buckwalter_path = await generate_audio(buckwalter_text)

        # Generate Arabic audio for playback
        arabic_audio, arabic_path = await generate_audio(arabic_text)

        # lipsync_data = await lip_sync(buckwalter_path)
        lipsync_data = await lip_sync(arabic_path)
        # buckwalter_path.unlink(missing_ok=True)

        return MessageResponse(
            text_en=arabic_text,
            audio=base64.b64encode(arabic_audio).decode("utf-8"),
            lipsync=lipsync_data,
            facialExpression=msg.get("facialExpression", "default"),
            animation=msg.get("animation", "Idle"),
        )
    except Exception as e:
        logger.error(f"Message processing failed: {str(e)}")
        return None


@router.post("/ask-sk", response_model=AnswerResponse)
async def ask_sk_question(
    question: str = Form(...),
    session_id: Optional[str] = Header(None, alias="Session-ID"),
):
    try:
        if question.lower() in ("e", "exit", "bye", "q", "quit"):
            return AnswerResponse(
                messages=[
                    MessageResponse(
                        text="Goodbye! 👋",
                        audio="",
                        facialExpression="smile",
                        animation="Idle",
                    )
                ],
                session_id=session_id or "",
            )

        # Load dance music base64 upfront

        # Special handling for dance command
        MUSIC_BASE64 = ""
        if question.lower().strip() == "dance":
            try:
                with open("audios/music.txt", "r") as f:
                    MUSIC_BASE64 = f.read().strip()
            except Exception as e:
                logging.warning(f"Could not load music base64: {e}")
            return AnswerResponse(
                messages=[
                    MessageResponse(
                        text_en="",  # Empty text since we don't want to speak
                        audio=MUSIC_BASE64,
                        lipsync={},  # Empty lipsync since no speech
                        facialExpression="smile",
                        animation="dance",
                    )
                ],
                session_id=session_id or str(uuid.uuid4()),
            )

        chat_history = ChatHistory()

        # if session_id:
        #     session = await mongo_manager.get_chat_session(session_id)
        #     if session and session.messages:
        #         for msg in session.messages:
        #             if msg.is_user:
        #                 chat_history.add_user_message(msg.content)
        #             else:
        #                 chat_history.add_assistant_message(msg.content)

        chat_history.add_user_message(question)

        # data = retriever.invoke(question)
        args = KernelArguments(
            # data=data,
            input=question,
            history="\n".join(
                [f"{msg.role}: {msg.content}" for msg in chat_history.messages]
            ),
        )

        response = await kernel.invoke(
            assistant_function, arguments=args, execution_settings=execution_settings
        )
        logging.error(f"modeeeeelllll resss: \n{response}")

        res_text = str(response)
        # logging.warning(f"modeeeeelllll resss: \n{res_text}")

        try:
            messages = json.loads(res_text)
            logging.warning(f"messagessss: \n{res_text}")

            if "messages" in messages:
                messages = messages["messages"]
        except json.JSONDecodeError:
            # Correct fallback structure
            messages = [
                {
                    # "text-buck": ">sf",
                    # "text-ar": "أسف",
                    "text-en": "sorry",
                    "facialExpression": "default",
                    "animation": "Idle",
                }
            ]

        message_responses = []
        for msg in messages:
            # if "text-ar" in msg:
            print(msg)
            if "text-en" in msg:
                processed = await process_message(msg)
                if processed:
                    message_responses.append(processed)
            else:
                logger.error(f"Missing Arabic text in message: {msg}")

        if not message_responses:
            message_responses.append(
                MessageResponse(
                    # text_ar="أسف، حدث خطأ أثناء معالجة طلبك",  # Arabic error message
                    text_en="Sorry something wrong happened",  # Arabic error message
                    audio="",
                    lipsync={},
                    facialExpression="sad",
                    animation="Idle",
                )
            )

        # user_message = Message(content=question, is_user=True)
        # bot_message = Message(
        #     content=json.dumps([m.text_en for m in message_responses]), is_user=False
        # )

        # Update chat history
        # if not session_id:
        #     new_session = ChatSession(messages=[user_message, bot_message])
        #     session_id = await mongo_manager.create_chat_session(new_session)
        # else:
        #     await mongo_manager.update_chat_session(session_id, user_message)
        #     await mongo_manager.update_chat_session(session_id, bot_message)

        return AnswerResponse(messages=message_responses, session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------CHAT APP GENERAL---------------------------#
# @router.get("/sessions/{session_id}", response_model=ChatSession)
# async def get_chat_session(session_id: str):
#     session = await mongo_manager.get_chat_session(session_id)
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")
#     return session


# @router.delete("/sessions/{session_id}", response_model=ChatSession)
# async def delete_chat_session(session_id: str):
#     success = await mongo_manager.delete_chat_session(session_id)
#     if not success:
#         raise HTTPException(status_code=404, detail="Session not found")
#     return {"status": "success", "message": "chat session is delete successfully"}


# @router.get("/sessions", response_model=List[str])
# async def get_all_sessions():
#     """Get all session IDs from MongoDB"""
#     sessions = await mongo_manager.chat_history.distinct("session_id")
#     return sessions


# @router.get("/history/{session_id}")
# async def get_chat_history(session_id: str):
#     history = await mongo_manager.get_chat_history_dict(session_id)
#     if not history:
#         raise HTTPException(status_code=404, detail="Session not found")
#     return history


########################## END ############################
########################## END ############################
########################## END ############################
