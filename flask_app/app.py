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
    api_key="U4EE4SOGSKX_3eKMPDY0wMR5l7gcw-WmsGz88EAwG8lF" #API key
)
project_id = os.getenv("PROJECT_ID", "fa51d7ee-4ca0-473a-b741-45dd29e0b2e4") #Project ID

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
        user_query = data.get('query')

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        # Format the query with additional instructions
        formatted_query = (
            f"Bullet points with short descriptions are preferred. Return the response as minimal HTML <body> content, structured with headings, bullet points, and bolded critical information."
            f"Ensure the output is plain text, concise, and suitable for mobile app display. Provide Ayurvedic remedies, dietary norms, yoga/exercise, and lifestyle precautions, avoiding allopathic medicines for the query: {user_query}"
        )

        # Build RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())

        # Get the response from Watson AI
        aiResponse = qa.invoke(formatted_query)
        return jsonify({"response": aiResponse.get("response")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "Watson Chat API is running!"

if __name__ == "__main__":
    app.run(debug=True)