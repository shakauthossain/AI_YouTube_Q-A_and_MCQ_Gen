from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import os
from io import BytesIO
from datetime import timedelta
import textwrap
from docx import Document
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from prompt import mcq_prompt
from services import extract_text, generate_mcq_docs
from dotenv import load_dotenv

load_dotenv()

app = APIRouter()
chat = ChatGroq(api_key=os.getenv("GROQ_API"), model="llama-3.1-8b-instant", temperature=0.5)
youtube_dbs = {}

class QuestionRequest(BaseModel):
    video_url: str
    question: str

class MCQRequest(BaseModel):
    video_url: str
    num_questions: int

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@app.post("/youtube/load/")
async def load_youtube(video_url: str):
    try:
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = splitter.split_documents(transcript)

        # Use a persistent directory if you want to retain data
        db = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            # persist_directory=f"./chroma_store/{video_id}"
        )

        youtube_dbs[video_url] = {"db": db, "docs": docs}
        return {"message": "Transcript loaded successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/youtube/ask/")
async def ask_question(req: QuestionRequest):
    if req.video_url not in youtube_dbs:
        return JSONResponse(status_code=400, content={"error": "Transcript not loaded."})

    results = youtube_dbs[req.video_url]["db"].similarity_search(req.question, k=4)
    context = " ".join([doc.page_content for doc in results])

    system_template = """You are a helpful assistant answering questions about a YouTube video transcript.
Use only the factual information from the transcript to answer the question.

Transcript: {docs}
If unsure, say "I don't know."
"""
    human_template = "Answer the following question: {question}"
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    chain = LLMChain(llm=chat, prompt=prompt)
    # chain = prompt | chat
    result = chain.invoke({
        "question": req.question,
        "docs": context
    })
    answer = result.content.strip() if hasattr(result, "content") else result["text"].strip()
    return {"answer": textwrap.fill(answer.strip(), width=80)}


@app.post("/youtube/mcqs/")
async def youtube_mcqs(req: MCQRequest):
    if req.video_url not in youtube_dbs:
        return JSONResponse(status_code=400, content={"error": "Transcript not loaded."})
    docs = youtube_dbs[req.video_url]["docs"]
    max_chars = 8000  # ~3000 tokens
    full_text = " ".join([doc.page_content for doc in docs])[:max_chars]
    # full_text = " ".join([doc.page_content for doc in docs])
    chain = mcq_prompt | chat

    result = chain.invoke({
        "context": full_text,
        "num_questions": req.num_questions
    })
    # Handle both string and dict-style outputs
    mcqs = result.content.strip() if hasattr(result, "content") else result["text"].strip()
    return {"mcqs": mcqs}


@app.post("/file/mcqs/")
async def file_mcqs(file: UploadFile = File(...), num_questions: int = Form(...)):
    try:
        content = extract_text(file.filename, file.file)
        chain = mcq_prompt | chat
        result = chain.invoke({
            "context": content[:8000],  # Optional: limit tokens to avoid 413
            "num_questions": num_questions
        })
        mcqs = result.content.strip() if hasattr(result, "content") else result["text"].strip()
        return {"mcqs": mcqs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/mcqs/download/")
async def download_mcqs(format: str = Form(...), content: str = Form(...)):
    pdf_bytes, docx_file = generate_mcq_docs(content)
    if format == "pdf":
        return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf",
                                 headers={"Content-Disposition": "attachment; filename=mcqs.pdf"})
    else:
        return StreamingResponse(docx_file,
                                 media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                 headers={"Content-Disposition": "attachment; filename=mcqs.docx"})