import eventlet
eventlet.monkey_patch()

import os
import requests
import tempfile
import time
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import logging
from requests.exceptions import RequestException
import boto3

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "audiobucket12345")

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
transcript_text = ''
vector_store = None
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                          google_api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize S3 client
s3 = boto3.client('s3',
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

@app.route('/')
def index():
    return render_template('index.html')

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def update_vector_store(new_text):
    global vector_store
    chunks = get_text_chunks(new_text)
    if not chunks:
        logger.warning("No new text chunks to add to vector store.")
        return

    if vector_store is None:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    else:
        vector_store.add_texts(chunks)

    logger.info("Vector store updated with new text chunks.")

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant in a meeting. Use the provided context to answer questions.
    If the answer is not in the context, use your general knowledge to provide a helpful response.
    Always strive to give detailed and informative answers.

    Context: {context}

    Human: {question}

    AI Assistant: Let me address your question based on the context of our meeting and my general knowledge.
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,
                                   google_api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

@socketio.on('audio_data')
def handle_audio_data(data):
    global transcript_text
    # Implement your audio data handling logic here
    # Make sure to use proper error handling and avoid infinite loops

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        logger.error("No audio file provided in the request")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            logger.info(f"Temporary audio file saved: {temp_file.name}")

            s3.upload_file(temp_file.name, AWS_S3_BUCKET_NAME, os.path.basename(temp_file.name))
            logger.info(f"Audio file uploaded to S3 bucket: {AWS_S3_BUCKET_NAME}")

        audio_url = s3.generate_presigned_url('get_object',
                                              Params={'Bucket': AWS_S3_BUCKET_NAME,
                                                      'Key': os.path.basename(temp_file.name)},
                                              ExpiresIn=3600)
        logger.info(f"Presigned URL generated for audio file")

        return jsonify({"audio_url": audio_url})
    except Exception as e:
        logger.error(f"Error handling audio upload: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing the audio"}), 500

@app.route('/end_meeting', methods=['POST'])
def end_meeting():
    global vector_store
    if vector_store:
        try:
            vector_store.save_local("faiss_index")
            logger.info("Vector store saved to disk.")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}", exc_info=True)
            return jsonify({"status": "error", "message": "Failed to save meeting data."}), 500
    return jsonify({"status": "success", "message": "Meeting ended and vector store saved."})

@app.route('/chat', methods=['POST'])
def chat():
    global vector_store
    user_question = request.json.get('question', '')

    try:
        if vector_store is None:
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7,
                                           google_api_key=os.getenv("GOOGLE_API_KEY"))
            response = model.predict(user_question)
            return jsonify({"reply": response})

        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return jsonify({"reply": response["output_text"]})
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}", exc_info=True)
        return jsonify({"reply": "An error occurred while processing your question. Please try again."})

if __name__ == "__main__":
    socketio.run(app, debug=True, use_reloader=False)
