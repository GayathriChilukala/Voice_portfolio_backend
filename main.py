from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import time
from collections import defaultdict

app = FastAPI(title="Resume Q&A Gateway")

# CORS - Allow your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Simple in-memory storage (Note: Vercel functions are stateless)
request_counts = defaultdict(list)
cache = {}

class QuestionRequest(BaseModel):
    question: str

# Resume context
RESUME_CONTEXT = """
Gayathri Chilukala
Bethesda, Maryland | gchilukala2023@fau.edu | (561) 875-3556

PROFESSIONAL EXPERIENCE
GEICO - Software Engineer, AI/ML (January 2025 – Present)
• LLM Backend Services & Infrastructure: Architected & deployed production-ready LLM backend services using Python and FastAPI, leveraging Azure managed compute and serverless infrastructure with vLLM
• Model Optimization & Evaluation: Achieved 2.5% accuracy improvement and 50% latency reduction using LLaMA model through systematic prompt fine-tuning
• Generative AI Platform Engineering: Developed end-to-end Generative AI platform featuring intelligent document processing with multi-format support (PDF, CSV, DOCX, TXT) and OCR capabilities
• ML Pipeline & Analytics: Developed LLM evaluation pipeline with 30% processing time reduction through parallel executions
• Advanced Token Management: Implemented sophisticated token counting feature for images and documents across 8 different LLM models
• Event-Driven Logging System: Built scalable logging service consuming encrypted and non-encrypted Kafka streams with PostgreSQL storage
• Developer Experience Platform: Built comprehensive MCP (Model Context Protocol) tools marketplace

Florida Atlantic University - AI and C++ Graduate Teaching Assistant (January 2024 – December 2024)
• ML Algorithm Implementation & Education: Developed and taught advanced AI/ML algorithms including gradient descent, ridge regression, and coordinate descent
• Developer Productivity Enhancement: Improved student learning outcomes by 40-50% through innovative ML performance visualization tools

TECHNICAL PROJECTS
SafetyMapper - Intelligent Community Safety Platform
• Developed platform using Python Flask, Google Cloud Firestore, and Google Gemini AI for real-time incident reporting

Pic2Plot - Multi-Agent AI System for Spatial Analysis
• Engineered sophisticated multi-agent AI system with 4 specialized modules using GPT-4.1, LLaMA 3 Vision, Phi-3, and DALL-E 3, achieving 40% cost reduction

TECHNICAL SKILLS
Backend: Python, FastAPI, Django, Node.js, Java, JavaScript, C++, Go, C#, PHP, TypeScript
Frontend: React.js, Next.js, Express.js, React Native
AI/ML: TensorFlow, Keras, PyTorch, Hugging Face Transformers, LangChain, OpenAI API, scikit-learn
Cloud: Azure, AWS (EC2, S3, Lambda), GCP, Docker, Kubernetes, Git, Kafka
Databases: PostgreSQL, MySQL, MongoDB, Firestore, Snowflake
AI Specializations: LLMs, RAG, Generative AI, NLP, Computer Vision, Agentic AI Systems

EDUCATION
Master of Science in Computer Science, Secondary Major in Artificial Intelligence
Florida Atlantic University | GPA: 3.96/4.0 | Aug 2023 - Dec 2024

Bachelor of Technology with Honours in Computer Science
SASTRA Deemed University, Tamil Nadu | June 2019 - June 2023

ACHIEVEMENTS
• Hugging Face Hackathon Winner (Caption Creator Pro): Architected high-performance multimodal AI platform achieving sub-2.1s caption generation
• Gangal Family Endowed Graduate Scholarship Award: Awarded to top 1% student in Computer Science department
"""

def check_rate_limit(ip: str) -> bool:
    """Simple rate limiting: 10 requests per minute"""
    now = time.time()
    request_counts[ip] = [t for t in request_counts[ip] if now - t < 60]
    
    if len(request_counts[ip]) >= 10:
        return False
    
    request_counts[ip].append(now)
    return True

def get_cached_response(question: str):
    """Simple caching: store responses for 10 minutes"""
    if question in cache:
        cached_time, cached_answer = cache[question]
        if time.time() - cached_time < 600:  # 10 minutes
            return cached_answer
    return None

def cache_response(question: str, answer: str):
    """Cache the response"""
    cache[question] = (time.time(), answer)

@app.get("/")
def home():
    return {
        "message": "Gayathri's Resume Q&A API",
        "status": "active",
        "endpoints": {
            "ask": "POST /ask",
            "health": "GET /health",
            "stats": "GET /stats"
        },
        "usage": "POST /ask with {'question': 'your question about Gayathri'}"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "cached_responses": len(cache),
        "model": "deepseek/deepseek-chat-v3.1:free"
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest, req: Request):
    # Get client IP
    client_ip = req.client.host if req.client else "unknown"
    
    # Rate limiting
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please wait a minute before asking another question."
        )
    
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Question too long. Please keep it under 500 characters.")
    
    # Check cache first
    cached = get_cached_response(question)
    if cached:
        return {
            "question": question,
            "answer": cached,
            "cached": True,
            "model": "deepseek/deepseek-chat-v3.1:free"
        }
    
    try:
        # Call OpenRouter with DeepSeek
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://gayathri-portfolio.vercel.app",
                "X-Title": "Resume Q&A Bot",
            },
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {
                    "role": "system", 
                    "content": f"""You are an AI assistant helping people learn about Gayathri Chilukala's professional background. 

Answer questions based ONLY on this resume information:
{RESUME_CONTEXT}

Instructions:
- Be professional, helpful, and enthusiastic about Gayathri's accomplishments
- If asked about something not in the resume, politely say it's not mentioned in her resume
- Keep responses concise but informative
- Highlight her key achievements and technical expertise
- For contact questions, provide her email: gchilukala2023@fau.edu"""
                },
                {"role": "user", "content": question}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        answer = completion.choices[0].message.content
        
        # Cache the response
        cache_response(question, answer)
        
        return {
            "question": question,
            "answer": answer,
            "cached": False,
            "model": "deepseek/deepseek-chat-v3.1:free"
        }
        
    except Exception as e:
        # Log error but don't expose internal details
        error_msg = "I'm having trouble processing your question right now. Please try again in a moment."
        
        if "api key" in str(e).lower():
            error_msg = "API configuration issue. Please contact support."
        elif "rate limit" in str(e).lower():
            error_msg = "API rate limit reached. Please try again later."
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/stats")
def get_stats():
    """API usage statistics"""
    total_ips = len(request_counts)
    total_requests = sum(len(reqs) for reqs in request_counts.values())
    cached_responses = len(cache)
    
    return {
        "unique_visitors": total_ips,
        "total_questions": total_requests,
        "cached_responses": cached_responses,
        "rate_limit": "10 requests per minute per IP",
        "cache_duration": "10 minutes"
    }

# For Vercel serverless functions
def handler(request):
    return app

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)