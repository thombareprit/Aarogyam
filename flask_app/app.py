# Import necessary libraries
import os
from ibm_watsonx_ai import Credentials

# Your credentials and API setup
credentials = Credentials(
    url="https://eu-gb.ml.cloud.ibm.com",  # Change if needed
    api_key="V_eGYYPGCZguAymSoheVcWlyKoj-0lh8ltj_WASW337G"  # Replace with your actual API key
)

# Handling project_id
try:
    project_id = os.environ["PROJECT_ID"]
except KeyError:
    project_id = "e0a3e29c-bb3b-4e9b-8302-08a77ed7fe1b"  # Replace with your actual project ID

from ibm_watsonx_ai import APIClient

api_client = APIClient(credentials=credentials, project_id=project_id)

# Extracting text from a PDF
import PyPDF2

pdf_file_path = "AarogyamDataset.pdf"  # Get the uploaded file name
pdf_text = ""

# Open and read the PDF
with open(pdf_file_path, "rb") as f:
    pdf_reader = PyPDF2.PdfReader(f)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

# Setup LangChain and IBM Watson for embeddings and model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_ibm import WatsonxEmbeddings

# Split the PDF text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len
)

chunks = text_splitter.split_text(pdf_text)
documents = [Document(page_content=chunk) for chunk in chunks]

# Initialize Watsonx Embeddings
embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id
)

from langchain_chroma import Chroma

# Create a vector store for document retrieval
docsearch = Chroma.from_documents(documents, embeddings)

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM

# Define model type and parameters
model_id = ModelTypes.GRANITE_13B_CHAT_V2
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}

# Initialize Watsonx Granite model
watsonx_granite = WatsonxLLM(
    model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)

from langchain.chains import RetrievalQA

# Build RetrievalQA using the vector store and Granite model
qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())

# Example query
query = "I have fever"
response = qa.invoke(query)
print(f"Answer: {response}")

# Flask application
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return f"Response from Watson AI: {response}"

if __name__ == "__main__":
    app.run(debug=True)

