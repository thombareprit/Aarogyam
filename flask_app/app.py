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
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# Flask app setup
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for API access

# Watsonx AI credentials setup
credentials = Credentials(
    url="https://eu-gb.ml.cloud.ibm.com",  # URL
    api_key="bzDb0wyHAsgFAl6FfMpvJ4urmKLp8l_S_Puwooce2p6p"  # API key
)
project_id = os.getenv("PROJECT_ID", "248c1435-612b-4a69-b8da-869749e2dac3")  # Project ID

# Initialize Watsonx Granite model
model_id = "ibm/granite-13b-instruct-v2"
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}

# Initialize the LLM
watsonx_granite = WatsonxLLM(
    model_id=model_id,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)

# Process the PDF file for context documents
pdf_file_path = "AarogyamDataset.pdf"  # PDF path
pdf_text = ""
try:
    with open(pdf_file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    print("info\nPDF successfully processed.")
except Exception as e:
    print(f"error\nFailed to process PDF: {e}")
    pdf_text = ""

# Split PDF text into chunks
try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(pdf_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    print(f"info\nTotal document chunks created: {len(documents)}")
except Exception as e:
    print(f"error\nFailed to split PDF text: {e}")
    documents = []

# Create vector store for document retrieval
try:
    embeddings = WatsonxEmbeddings(
        model_id="ibm/slate-30m-english-rtrvr",
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id=project_id
    )
    docsearch = Chroma.from_documents(documents, embeddings)
    print("info\nVector store successfully created.")
except Exception as e:
    print(f"error\nFailed to create vector store: {e}")
    docsearch = None

@app.route('/watsonchat', methods=['POST'])
def watsonchat():
    try:
        # Parse the query from the request
        data = request.get_json()
        user_query = data.get('query')

        if not user_query:
            print("error\nNo query provided in the request.")
            return jsonify({"error": "No query provided"}), 400

        print(f"info\nUser Query: {user_query}")

        # Format the query with additional instructions
        formatted_query = (
            f"Bullet points with short descriptions are preferred. Return the response as minimal HTML <body> content, structured with headings, bullet points, and bolded critical information."
            f"Ensure the output is plain text, concise, and suitable for mobile app display. Provide Ayurvedic remedies, dietary norms, yoga/exercise, and lifestyle precautions, avoiding allopathic medicines for the query: {user_query}"
        )

        print(f"critical\nFormatted Query: {formatted_query}")

        # Ensure the vector store is initialized
        if not docsearch:
            print("error\nVector store is not initialized.")
            return jsonify({"error": "Vector store is not initialized"}), 500

        # Build RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())

        # Get the response from Watson AI
        aiResponse = qa.invoke(formatted_query)
        print(f"critical\nAI Response: {aiResponse}")

        ai_response_result = aiResponse.get("result")
        if not ai_response_result:
            print("error\nAI did not return a valid result.")
            return jsonify({"error": "AI did not return a valid result."}), 500

        return jsonify({"response": ai_response_result})
    except Exception as e:
        print(f"error\nException: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "Watson Chat API is running!"

if __name__ == "__main__":
    app.run(debug=True)