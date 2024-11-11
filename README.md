# Catalyst

It is RAG (Retrieval Augmented Generation) app in Python that can let you query/chat with your PDFs using generative AI.
It works completly locally with running nomic-embed-text model of ollama running localy which embeds the text in pdf and stores it in chromadb vector database and then
llama 3.2 llm also running locally takes the query and embeds it and search for matches in the chromadb vectordb and generates the response accordingly

## Features and Functionality

- Load PDF documents using `PyPDFLoader`.
- Split loaded documents into manageable chunks for better processing.
- Generate embeddings for text chunks with `OllamaEmbeddings`.
- Store and manage embeddings in a Chroma database.
- Query the database to retrieve relevant document chunks based on user input.
- Provide context-based answers using a language model.

## Tech Stack

- **Python**: Programming language used for the implementation.
- **LangChain**: Framework for building applications with LLMs, which includes:
  - `langchain_community.document_loaders`: For loading documents from different formats.
  - `langchain_text_splitters`: For splitting text into chunks.
  - `langchain_ollama`: For working with Ollama LLMs and embeddings.
  - `langchain_chroma`: For managing embeddings with Chroma.
- **Chroma**: A vector database for storing and querying embeddings.
