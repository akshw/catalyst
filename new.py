import argparse
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate



PROMPT_TEMPLATE = """
                    Answer the question based only on the following context:
                    {context}
                    
                    ---
                    Answer the question based on the above context: {question}
                    """

def load_documents():
    document_loader = PyPDFLoader('./data.pdf')
    return document_loader.load()

data = load_documents()

def split_documents(data: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
    )
    return text_splitter.split_documents(data)

chunks = split_documents(data)

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def add_to_chroma(chunks: list[Document]):
    embedding_fn = get_embedding_function()
    db = Chroma.from_documents(
        collection_name='my-data',
        documents=chunks,
         persist_directory='Chromadb', embedding_function=embedding_fn
    ).from_documents(chunks)
    db.persist()

def query(query_text: str):
    db = Chroma(persist_directory="Chromadb", embedding_function=get_embedding_function())
    results = db.similarity_search_with_score(query_text, k=2)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Generated Prompt:\n{prompt}")

    model = Ollama(model="llama3.2:latest")
    response_text = model.invoke(prompt)
    
    print(f"Model Response:\n{response_text}")

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    
    return formatted_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a query.")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    
    add_to_chroma(chunks)  # Indexing documents into Chroma
    query(args.query_text)