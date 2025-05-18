from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from src.config import MONGODB_DB_NAME, MONGODB_URI


class Message(BaseModel):
    content: str
    is_user: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    messages: list[Message] = []


class MongoDBManager:
    def __init__(self, connection_string: str, db_name: str):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[db_name]
        self.chat_history = self.db["chat-history"]

    async def create_chat_session(self, session: ChatSession) -> str:
        await self.chat_history.insert_one(session.dict())
        return session.session_id

    async def update_chat_session(
        self, session_id: str, message: Message, max_messages: Optional[int] = 100
    ) -> None:
        await self.chat_history.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {"$each": [message.dict()], "$slice": -max_messages}
                },
                "$set": {"updated_at": datetime.now()},
            },
        )

    async def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        session = await self.chat_history.find_one({"session_id": session_id})
        return ChatSession(**session) if session else None

    async def delete_chat_session(self, session_id: str) -> bool:
        result = await self.chat_history.delete_one({"session_id": session_id})
        return result.deleted_count > 0

    async def get_all_sessions(self) -> List[str]:
        """Get all unique session IDs"""
        return await self.chat_history.distinct("session_id")

    async def get_chat_history_dict(self, session_id: str) -> Optional[dict]:
        """Get chat history in {questions: [], answers: []} format"""
        session = await self.get_chat_session(session_id)
        if not session:
            return None

        history = {"questions": [], "answers": []}

        for message in session.messages:
            if message.is_user:
                history["questions"].append(
                    {"content": message.content, "timestamp": message.timestamp}
                )
            else:
                history["answers"].append(
                    {"content": message.content, "timestamp": message.timestamp}
                )

        return history


mongo_manager = MongoDBManager(MONGODB_URI, MONGODB_DB_NAME)
