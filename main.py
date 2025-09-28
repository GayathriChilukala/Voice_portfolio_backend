from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI(title="Resume Q&A Gateway")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

class QuestionRequest(BaseModel):
    question: str

# Resume context
RESUME_CONTEXT = """
Gayathri Chilukala
Bethesda, Maryland | gchilukala2023@fau.edu | (561) 875-3556

PROFESSIONAL EXPERIENCE
GEICO - Software Engineer, AI/ML (January 2025 – Present)
• Architected & deployed production-ready LLM backend services using Python and FastAPI
• Achieved 2.5% accuracy improvement and 50% latency reduction using LLaMA model
• Developed end-to-end Generative AI platform with intelligent document processing
• Built LLM evaluation pipeline with 30% processing time reduction
• Implemented token counting feature across 8 different LLM models

Florida Atlantic University - Graduate Teaching Assistant (January 2024 – December 2024)
• Developed and taught advanced AI/ML algorithms
• Improved student learning outcomes by 40-50%

PROJECTS
SafetyMapper - Intelligent Community Safety Platform using Python Flask and Google Gemini AI
Pic2Plot - Multi-Agent AI System achieving 40% cost reduction

SKILLS: Python, FastAPI, Django, TensorFlow, PyTorch, Azure, AWS, React.js

EDUCATION
Master of Science in Computer Science, AI Specialization
Florida Atlantic University | GPA: 3.96/4.0

ACHIEVEMENTS
• Hugging Face Hackathon Winner (Caption Creator Pro)
• Gangal Family Endowed Graduate Scholarship Award (Top 1% of CS students)
"""

@app.get("/")
def home():
    return {
        "message": "Gayathri's Resume Q&A API",
        "status": "active",
        "version": "1.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "deepseek"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if len(question) > 500:
            raise HTTPException(status_code=400, detail="Question too long")

        # Check if API key exists
        if not os.getenv("OPENROUTER_API_KEY"):
            raise HTTPException(status_code=500, detail="API key not configured")
        
        # Call OpenRouter
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {
                    "role": "system", 
                    "content": f"You are helping someone learn about Gayathri Chilukala's background. Answer based on this resume: {RESUME_CONTEXT}. Be professional and concise."
                },
                {"role": "user", "content": question}
            ],
            max_tokens=250,
            temperature=0.7
        )
        
        answer = completion.choices[0].message.content
        
        return {
            "question": question,
            "answer": answer,
            "model": "deepseek"
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")  # For Vercel logs
        raise HTTPException(status_code=500, detail="Unable to process question right now")

# For Vercel
handler = app
