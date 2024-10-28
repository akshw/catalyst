from langchain_community.document_loaders.pdf import PyPDFLoader


def load_documents():
    document_loader = PyPDFLoader('./data.pdf')
    return document_loader.load()


document = load_documents()
print(document)