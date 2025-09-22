from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def convert_data(file_path):
    extracted_documents = load_pdf(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text_chunks = splitter.split_documents(extracted_documents)
    return text_chunks

if __name__ == "__main__":
    pdf_path = r"/Users/aashutoshkumar/Documents/Projects/healthgraph-assistant/data/8205Oxford Handbook of Clinical Medicine 10th 2017 Edition_SamanSarKo - Copy (1).pdf"
    text_chunks = convert_data(pdf_path)
    print(len(text_chunks), "\n")
    print(text_chunks[:5])
