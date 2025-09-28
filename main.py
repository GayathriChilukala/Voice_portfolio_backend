from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from typing import Optional

app = FastAPI(title="Gayathri's Resume Assistant - GPT-4.1 Nano", description="Ask questions about Gayathri Chilukala's resume using GPT-4.1 Nano")

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    model: str

@app.get("/")
async def root():
    return {
        "message": "Gayathri Chilukala's AI Resume Assistant is running!",
        "description": "Ask me anything about Gayathri's professional experience, skills, education, or projects",
        "endpoints": {
            "POST /ask": "Ask questions about Gayathri's resume",
            "GET /": "This endpoint",
            "GET /health": "Health check"
        },
        "example_questions": [
            "What is Gayathri's current job?",
            "What programming languages does she know?",
            "Tell me about her AI/ML experience",
            "What projects has she worked on?",
            "What is her educational background?"
        ]
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise HTTPException(
                status_code=500, 
                detail="GITHUB_TOKEN environment variable is not set"
            )
        
        endpoint = "https://models.github.ai/inference"
        model_name = "openai/gpt-4.1-nano"
        
        # Resume context for Gayathri Chilukala - Exact copy from resume
        resume_context = """
        Gayathri Chilukala
        Bethesda, Maryland | gchilukala2023@fau.edu | (561) 875-3556 | LinkedIn | GitHub

        PROFESSIONAL EXPERIENCE
        GEICO - Software Engineer, AI/ML Chevy Chase, Maryland | January 2025 – Present
        • LLM Backend Services & Infrastructure: Architected & deployed production-ready LLM backend services using Python and FastAPI, leveraging Azure managed compute and serverless infrastructure with vLLM for scalable AI application development.
        • Model Optimization & Evaluation: Evaluated multiple LLMs for claims processing, achieving 2.5% accuracy improvement and 50% latency reduction using LLaMA model through systematic prompt fine-tuning while minimizing operational costs
        • Generative AI Platform Engineering: Developed end-to-end Generative AI platform featuring intelligent document processing with multi-format support (PDF, CSV, DOCX, TXT) and OCR capabilities, implementing secure role-based access control with dynamic UI rendering based on user permission hierarchies to enhance developer productivity
        • ML Pipeline & Analytics: Developed LLM evaluation pipeline with 30% processing time reduction through parallel executions, created Snowflake-powered dashboard for comparative model analysis and performance monitoring.
        • Advanced Token Management: Implemented sophisticated token counting feature for images and documents across 8 different LLM models, developing model-specific calculation formulas and enforcing dynamic token limits with automated error handling
        • Event-Driven Logging System: Built scalable logging service consuming encrypted and non-encrypted Kafka streams with PostgreSQL storage, providing comprehensive audit trails and real-time system monitoring.
        • Developer Experience Platform: Built comprehensive MCP (Model Context Protocol) tools marketplace where users can register existing MCP tools (with links, configurations, and parameters) into a centralized library, enabling discovery and testing of tools for specific use case

        Florida Atlantic University - AI and C++ Graduate Teaching Assistant Boca Raton, Florida | January 2024 – December 2024
        • ML Algorithm Implementation & Education: Developed and taught advanced AI/ML algorithms including gradient descent, ridge regression, and coordinate descent, creating comprehensive hyperparameter tuning frameworks using cross-validation techniques (k-fold, LASSO, ridge regression)
        • Developer Productivity Enhancement: Improved student learning outcomes by 40-50% through innovative ML performance visualization tools built with ggplot2, while instructing 300+ students in C++ object-oriented programming and algorithm optimization.

        TECHNICAL PROJECTS
        SafetyMapper - Intelligent Community Safety Platform
        • Developed platform using Python Flask, Google Cloud Firestore, and Google Gemini AI for real-time incident reporting and AI-powered safety recommendations. Implemented route safety analysis system that evaluates transportation paths against incident data with multi-modal risk assessment and visual safety indicators.

        Pic2Plot - Multi-Agent AI System for Spatial Analysis
        • Engineered sophisticated multi-agent AI system with 4 specialized modules (image-to-floorplan generation, text-to-floorplan conversion, real estate automation, health recommendations) using GPT-4.1, LLaMA 3 Vision, Phi-3, and DALL-E 3, implementing adaptive cost optimization achieving 40% cost reduction and 3× performance improvement through FastMCP framework.

        TECHNICAL SKILLS
        Backend Development: Python, FastAPI, Django, Node.js, Java, JavaScript, C++, Go, C#, PHP, TypeScript
        Frontend & FullStack: React.js, Next.js, Express.js, React Native.
        Cloud & DevOps: Azure, AWS (EC2, S3, Lambda), GCP, Docker, Kubernetes, Git, Kafka, Enterprise CI/CD Pipelines.
        AI/ML Frameworks: TensorFlow, Keras, PyTorch, Hugging Face Transformers, LangChain, OpenAI API, scikit-learn, NLTK
        Databases & Analytics: PostgreSQL, MySQL, MongoDB, Firestore, Snowflake
        AI Specializations: Large Language Models (LLMs), Retrieval Augmented Generation (RAG), Generative AI, Natural Language Processing, Computer Vision, Agentic AI Systems

        EDUCATION
        Master of Science in Computer Science, Secondary Major in Artificial Intelligence
        Florida Atlantic University, Boca Raton, FL | GPA: 3.96/4.0 | Aug 2023 - Dec 2024
        Bachelor of Technology with Honours in Computer Science | SASTRA Deemed University, Tamil Nadu | June 2019 - June 2023

        ACHIEVEMENTS
        • Hugging Face Hackathon Winner (Caption Creator Pro): Architected high-performance multimodal AI platform integrating SambaNova Llama models with sophisticated multi-provider translation system, achieving sub-2.1s caption generation and 40% performance improvement over industry standards.
        • Gangal Family Endowed Graduate Scholarship Award: Awarded to top 1% student in Computer Science department for academic excellence and TA contributions; only one recipient selected department wide.
        """
        
        # Prepare the request payload with resume context
        payload = {
            "messages": [
                {"role": "system", "content": f"You are a professional resume assistant for Gayathri Chilukala. Answer questions about her resume directly and concisely. Do not show your thinking process or reasoning. Give factual, to-the-point answers based on this resume:\n\n{resume_context}\n\nProvide clear, brief answers with specific dates and details when asked."},
                {"role": "user", "content": request.question}
            ],
            "max_tokens": 500,
            "temperature": 0.1,
            "top_p": 1.0,
            "model": model_name
        }
        
        # Make the request to GitHub Models API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                error_detail = response.text
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"API request failed: {error_detail}"
                )
            
            result = response.json()
            
            if "choices" not in result or len(result["choices"]) == 0:
                raise HTTPException(
                    status_code=500,
                    detail="No response generated by the model"
                )
            
            answer = result["choices"][0]["message"]["content"]
            
            # Clean up the response - remove thinking tags and extra formatting
            import re
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)  # Remove bold formatting
            answer = answer.strip()
            
            return QuestionResponse(
                question=request.question,
                answer=answer,
                model=model_name
            )
            
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=408,
            detail="Request timeout - the AI model took too long to respond"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
