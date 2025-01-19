# Import necessary libraries
import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # To handle cross-origin requests
from ibm_watsonx_ai import Credentials, APIClient
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# Flask app setup
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for API access

# Watsonx AI credentials setup
credentials = Credentials(
    url="https://eu-gb.ml.cloud.ibm.com",
    api_key="V_eGYYPGCZguAymSoheVcWlyKoj-0lh8ltj_WASW337G" #API key
)
project_id = os.getenv("PROJECT_ID", "e0a3e29c-bb3b-4e9b-8302-08a77ed7fe1b") #Project ID

# Initialize Watsonx Granite model
model_id = ModelTypes.GRANITE_13B_CHAT_V2
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}

watsonx_granite = WatsonxLLM(
    model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)

# PDF processing (if needed for context documents)
pdf_file_path = "AarogyamDataset.pdf" #PDF path
pdf_text = ""

with open(pdf_file_path, "rb") as f:
    pdf_reader = PyPDF2.PdfReader(f)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

# Split PDF text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_text(pdf_text)
documents = [Document(page_content=chunk) for chunk in chunks]

# Create vector store for document retrieval
embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id
)
docsearch = Chroma.from_documents(documents, embeddings)

@app.route('/watsonchat', methods=['POST'])
def watsonchat():
    try:
        # Parse the query from the request
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Build RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())

        # Get the response from Watson AI
        response = qa.invoke(query)
        return jsonify({"query": query, "response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "Watson Chat API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)