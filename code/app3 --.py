import streamlit as st
import boto3
import os
import uuid
import time
import json
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document 
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, SystemMessage

# Constants
PROMPT_TEMPLATE = """Human: Answer the question based only on the information provided in few sentences.
<context>
{}  #المعايير هنا 
</context>
</question>
Assistant:"""

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-west-2") #تغيير
S3_BUCKET = os.getenv("S3_BUCKET", "videos-rag-bucket") #تغيير 

# Initialize AWS clients
s3 = boto3.client('s3', region_name=AWS_REGION)
transcribe = boto3.client('transcribe', region_name=AWS_REGION)
bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)

# Streamlit page config
st.set_page_config(page_title="Meeting Analyzer Chat", page_icon="🤖")

# Sidebar for model selection and temperature setting
st.sidebar.title("Settings")
embedding_model = st.sidebar.selectbox(
    "Choose Embedding Model:",
    ["amazon.titan-embed-g1-text-02"]  
)

bedrock_model = st.sidebar.selectbox(
    "Choose Bedrock Model:",
    ["anthropic.claude-3-sonnet-20240229-v1:0"]  
)

temperature = st.sidebar.slider(
    "Select Temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

def upload_to_s3(file, bucket, key):
    try:
        s3.upload_fileobj(file, bucket, key)
        return True
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return False

def start_transcription_job(media_uri, job_name):
    try:
        response = transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': media_uri},
            MediaFormat=media_uri.split('.')[-1],
            OutputBucketName=S3_BUCKET,
            OutputKey=f"transcripts/{job_name}.json",
            LanguageCode='en-US'
        )
        return response['TranscriptionJob']['TranscriptionJobName']
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return None

def check_transcription_status(job_name):
    response = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    return response['TranscriptionJob']['TranscriptionJobStatus']



    response = llm(messages)
    
    answer_content = response.content.strip()
    answer_content = answer_content.replace("<answer>", "").replace("</answer>", "").strip()

    return answer_content  


# File Upload Section
uploaded_file = st.file_uploader("Upload meeting audio/video", type=["mp3", "wav", "mp4", "mov", "m4a"])
if uploaded_file:
    file_key = f"uploads/{uuid.uuid4()}.{uploaded_file.name.split('.')[-1]}"
    if upload_to_s3(uploaded_file, S3_BUCKET, file_key):
        st.success("File uploaded to S3!")
        media_uri = f"s3://{S3_BUCKET}/{file_key}"

        if st.session_state.transcription_job_name is None:
            job_name = f"transcription-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.session_state.transcription_job_name = job_name
            
            if start_transcription_job(media_uri, job_name):
                st.info("Transcription job started...")
                transcription_status = None
                while transcription_status not in ['COMPLETED', 'FAILED']:
                    transcription_status = check_transcription_status(job_name)
                    st.write(f"Current status: {transcription_status}")
                    time.sleep(15)
                
                if transcription_status == 'COMPLETED':
                    text_path = f"transcripts/{job_name}.json"
                    if process_transcript_to_faiss(text_path):
                        st.success("Knowledge base ready! Start chatting below.")
        else:
            st.success("Transcription job already initiated. Waiting for completion...")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the meeting"):
    if 'vectorstore' not in st.session_state:
        st.error("Please process a meeting file first")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_llm_response(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})