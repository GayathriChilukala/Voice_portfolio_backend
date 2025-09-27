from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import time
from collections import defaultdict

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
    api_key=os.getenv("OPENROUTER_API_KEY", "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")
)

# Simple rate limiting
request_counts = defaultdict(list)
cache = {}

class QuestionRequest(BaseModel):
    question: str

# Your resume - UPDATE THIS
RESUME_CONTEXT = """
Gayathri Chilukala
Bethesda, Maryland | gchilukala2023@fau.edu | (561) 875-3556 | LinkedIn | GitHub PROFESSIONAL EXPERIENCE
GEICO - Software Engineer, AI/ML Chevy Chase, Maryland | January 2025 – Present
• LLM Backend Services & Infrastructure: Architected & deployed production-ready LLM backend services using Python and FastAPI, leveraging Azure managed compute and serverless infrastructure with vLLM for scalable AI application development.
• Model Optimization & Evaluation: Evaluated multiple LLMs for claims processing, achieving 2.5% accuracy improvement and 50% latency reduction using LLaMA model through systematic prompt fine-tuning while minimizing operational costs
• Generative AI Platform Engineering: Developed end-to-end Generative AI platform featuring intelligent document processing with multi-format support (PDF, CSV, DOCX, TXT) and OCR capabilities, implementing secure role-based access control with dynamic UI rendering based on user permission hierarchies to enhance developer productivity
• ML Pipeline & Analytics: Developed LLM evaluation pipeline with 30% processing time reduction through parallel executions, created Snowflake-powered dashboard for comparative model analysis and performance monitoring.
• Advanced Token Management: Implemented sophisticated token counting feature for images and documents across 8 different LLM models, developing model-specific calculation formulas and enforcing dynamic token limits with automated error handling
• Event-Driven Logging System: Built scalable logging service consuming encrypted and non-encrypted Kafka streams with PostgreSQL storage, providing comprehensive audit trails and real-time system monitoring.
• Developer Experience Platform: Built comprehensive MCP (Model Context Protocol) tools marketplace where users can register existing MCP tools (with links, configurations, and parameters) into a centralized library, enabling discovery and testing of tools for specific use case
Florida Atlantic University - AI and C++ Graduate Teaching Assistant Boca Raton, Florida | January 2024 – December 2024
• ML Algorithm Implementation & Education: Developed and taught advanced AI/ML algorithms including gradient descent, ridge regression, and coordinate descent, creating comprehensive hyperparameter tuning frameworks using cross-validation techniques (k- fold, LASSO, ridge regression)
• Developer Productivity Enhancement: Improved student learning outcomes by 40-50% through innovative ML performance visualization tools built with ggplot2, while instructing 300+ students in C++ object-oriented programming and algorithm optimization.
TECHNICAL PROJECTS
SafetyMapper - Intelligent Community Safety Platform
• Developed platform using Python Flask, Google Cloud Firestore, and Google Gemini AI for real-time incident reporting and AI- powered safety recommendations. Implemented route safety analysis system that evaluates transportation paths against incident data with multi-modal risk assessment and visual safety indicators.
Pic2Plot - Multi-Agent AI System for Spatial Analysis
• Engineered sophisticated multi-agent AI system with 4 specialized modules (image-to-floorplan generation, text-to-floorplan conversion, real estate automation, health recommendations) using GPT-4.1, LLaMA 3 Vision, Phi-3, and DALL-E 3, implementing adaptive cost optimization achieving 40% cost reduction and 3× performance improvement through FastMCP framework.
      TECHNICAL SKILLS
Backend Development: Python, FastAPI, Django, Node.js, Java, JavaScript, C++, Go, C#, PHP, TypeScript
Frontend &FullStack: React.js, Next.js, Express.js,
React Native.
Cloud & DevOps: Azure, AWS (EC2, S3, Lambda), GCP, Docker, Kubernetes, Git, Kafka, Enterprise CI/CD Pipelines.
AI/ML Frameworks: TensorFlow, Keras, PyTorch, Hugging Face Transformers, LangChain, OpenAI API, scikit-learn, NLTK Databases & Analytics: PostgreSQL, MySQL, MongoDB, Firestore, Snowflake
AI Specializations: Large Language Models (LLMs), Retrieval Augmented Generation (RAG), Generative AI, Natural Language Processing, Computer Vision, Agentic AI Systems
 EDUCATION
Master of Science in Computer Science, Secondary Major in Artificial Intelligence
Florida Atlantic University, Boca Raton, FL | GPA: 3.96/4.0 | Aug 2023 - Dec 2024
Bachelor of Technology with Honours in Computer Science | SASTRA Deemed University, Tamil Nadu |June 2019 - June 2023 ACHIEVEMENTS
• Hugging Face Hackathon Winner(Caption Creator Pro): Architected high-performance multimodal AI platform integrating SambaNova Llama models with sophisticated multi-provider translation system, achieving sub-2.1s caption generation and 40% performance improvement over industry standards.
• Gangal Family Endowed Graduate Scholarship Award: Awarded to top 1% student in Computer Science department for academic excellence and TA contributions; only one recipient selected department wide.
  
"""

def check_rate_limit(ip: str) -> bool:
    """Simple rate limiting: 5 requests per minute"""
    now = time.time()
    # Clean old requests (older than 1 minute)
    request_counts[ip] = [t for t in request_counts[ip] if now - t < 60]
    
    if len(request_counts[ip]) >= 5:
        return False
    
    request_counts[ip].append(now)
    return True

def get_cached_response(question: str):
    """Simple caching: store responses for 5 minutes"""
    if question in cache:
        cached_time, cached_answer = cache[question]
        if time.time() - cached_time < 300:  # 5 minutes
            return cached_answer
    return None

def cache_response(question: str, answer: str):
    """Cache the response"""
    cache[question] = (time.time(), answer)

@app.get("/")
def home():
    return {
        "message": "Resume Q&A API Gateway",
        "features": ["Rate Limited", "Cached Responses", "Portfolio Ready"],
        "usage": "POST /ask with {'question': 'your question'}"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "cached_responses": len(cache)}

@app.post("/ask")
async def ask_question(request: QuestionRequest, client_ip: str = "127.0.0.1"):
    # Rate limiting
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Too many requests. Try again in a minute."
        )
    
    question = request.question.strip()
    
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
        # Call OpenRouter
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://your-portfolio.com",
                "X-Title": "Resume Q&A Bot",
            },
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {
                    "role": "system", 
                    "content": f"You're helping someone learn about this person's background. Answer questions based on this resume:\n\n{RESUME_CONTEXT}\n\nBe helpful and professional. If something isn't in the resume, say so politely."
                },
                {"role": "user", "content": question}
            ],
            max_tokens=400,
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/stats")
def get_stats():
    """Simple stats endpoint"""
    total_ips = len(request_counts)
    total_requests = sum(len(reqs) for reqs in request_counts.values())
    cached_responses = len(cache)
    
    return {
        "total_unique_visitors": total_ips,
        "total_requests": total_requests,
        "cached_responses": cached_responses,
        "rate_limit": "5 requests per minute per IP"
    }

# Quick test endpoints
@app.get("/test")
def quick_test():
    return {"message": "API is working! Try POST /ask with a question."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)