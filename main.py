from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx

app = FastAPI(title="Resume Q&A Gateway")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "version": "2.0"
    }

@app.get("/health")
def health():
    api_key_status = "configured" if os.getenv("OPENROUTER_API_KEY") else "missing"
    return {
        "status": "healthy", 
        "api_key": api_key_status,
        "model": "deepseek"
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if len(question) > 500:
            raise HTTPException(status_code=400, detail="Question too long")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        # Use httpx directly instead of OpenAI client to avoid version issues
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://gayathri-portfolio.vercel.app",
                    "X-Title": "Resume Q&A Bot",
                },
                json={
                    "model": "deepseek/deepseek-chat",
                    "messages": [
                        {
                            "role": "system", 
                            "content": f"You are helping someone learn about Gayathri Chilukala's background. Answer based on this resume: {RESUME_CONTEXT}. Be professional and concise."
                        },
                        {"role": "user", "content": question}
                    ],
                    "max_tokens": 250,
                    "temperature": 0.7
                }
            )
        
        if response.status_code != 200:
            print(f"OpenRouter API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="API request failed")
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        return {
            "question": question,
            "answer": answer,
            "model": "deepseek"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Unable to process question right now")

# For Vercel
handler = app
