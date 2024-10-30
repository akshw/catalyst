import argparse
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate


path = './data.pdf'
db_path = 'chroma_db'


def load_data():
    data = PyPDFLoader(path)
    return data.load()

data = load_data()


def split_data(data: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 50,
        length_function = len,
        is_separator_regex = False
    )
    return text_splitter.split_documents(data)

chunks = split_data(data)


def get_embeddings():
    embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')
    return embeddings


def add_to_db(chunks: list[Document]):
    db = Chroma(
        persist_directory=db_path, embedding_function=get_embeddings()
    )
    db.reset_collection()
    db.add_documents(chunks)
    print(f"Total documents in DB: {db._collection.count()}")

add_to_db(chunks)



def query(query_text: str):
    db = Chroma(
        persist_directory=db_path, embedding_function=get_embeddings()
    )
    results = db.similarity_search(query_text, k=2)

    context = "\n\n---\n\n".join(doc.page_content for doc in results)
    prompt_template = 'Answer the question based on the' + str(context)
    prompt = prompt_template.format("{context}", context).replace("{question}", query_text)


    model = OllamaLLM(model='llama3.2:latest')
    llm_response = model.invoke(prompt)
    print(llm_response)

query("components used")    
    


