# 🤖 Ebla Assistant — RAG-Powered Chat App

This project is a **chat assistant built for Ebla Computer Consultancy**, designed to provide helpful responses using Retrieval-Augmented Generation (RAG). It integrates:

- **FastAPI** for the backend API
- **Ollama** model for LLM responses
- **MongoDB** for chat session storage
- **Chroma** for vector database
- **Custom UI** for the chatbot interface

The assistant **remembers previous messages** in each chat session and uses RAG to provide context-aware responses by retrieving relevant knowledge chunks.

---

## 🎯 Purpose

This project was created to gain a **practical understanding of how Retrieval-Augmented Generation (RAG)** works. It demonstrates how RAG can improve the accuracy and relevance of AI responses by combining local context with external vectorized knowledge.

---

## 🖥️ Features

- 🔹 Clean chatbot UI for interaction
- 🔹 Session-based chat history with memory (stored in MongoDB)
- 🔹 Vector search integration using Chroma
- 🔹 Ollama model for natural language responses
- 🔹 Built with modular FastAPI backend

---

## 📦 Requirements

Make sure you have the following installed:

- Ollama
- Python 3.10+
- `pip` package manager
- `virtualenv` (recommended)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/3bdop/chat-app.git
cd chat-app
```

### 2. Install Ollama 🦙

Ollama lets you run large language models locally. Follow the instructions for your OS:

- **macOS / Linux / Windows**:  
  [https://ollama.com/download](https://ollama.com/download)

- After installation, verify it's working:

  ```bash
  ollama --version
  ```

- Pull llama3 model:

  ```bash
  ollama pull llama3
  ```

- Verify it's installed:

  ```bash
  ollama list
  ```

### 3. FastAPI setup

- Create environment (recommended):
  ```bash
  python -m venv .venv
  ```
- Activate environment:

  ```bash
  source .venv/bin/activate    #Windows: .\.venv\Scripts\activate.bat
  ```

- Install requirements:
  ```bash
  pip install -r requirements.txt
  ```

### 4. Environment variable setup

- Create a `.env` file in the root and the following with your own values:

  ```bash
  MONGODB_URI=<your db string>
  MONGODB_DB_NAME=<your db name>
  ```

### 5. Run the chat app

- To run the app type the following:
  ```bash
  fastapi dev src/main.py
  ```
