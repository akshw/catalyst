from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama


path = './data.pdf'
if(path):
    loader = UnstructuredPDFLoader(file_path = path)
    data = loader.load()
else:
    print('file not found')


data_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = data_splitter.split_documents(data)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text:latest"),
    collection_name="data-rag"
)


local_model = "llama3.2:latest"
llm = ChatOllama(model=local_model)
myprompt = PromptTemplate(
    input_variables=['questions'],
    template="""You are an AI language model assistant. Your task is to generate four different versions of the given user question to retrive relavent documents from a vector database.
      By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
      Provide these alternative questions seperated by newslines. Original question: {question}""",
)

retriver = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=myprompt
)
#rag prompt
template = '''Answer the question based only on the following context:
{context}
Question: {question}
'''
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {'context':retriver, 'question':RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#chain.invoke(input("type ur question "))
print(chain.invoke("What is the problem"))
print('hii')