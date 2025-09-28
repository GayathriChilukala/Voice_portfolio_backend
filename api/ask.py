import json
import os
import httpx
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Handle GET requests
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response = {
            "message": "Gayathri's Resume Q&A API",
            "status": "active",
            "usage": "Send POST request to /api/ask with {'question': 'your question'}"
        }
        
        self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            question = data.get('question', '').strip()
            
            if not question:
                self.send_error_response(400, "Question is required")
                return
            
            if len(question) > 500:
                self.send_error_response(400, "Question too long")
                return
            
            # Get API response
            answer = self.get_ai_response(question)
            
            # Send successful response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            response = {
                "question": question,
                "answer": answer,
                "model": "deepseek"
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"Error: {str(e)}")
            self.send_error_response(500, "Internal server error")

    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def send_error_response(self, status_code, message):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = {"error": message}
        self.wfile.write(json.dumps(error_response).encode())

    def get_ai_response(self, question):
        # Resume context
        resume_context = """
        Gayathri Chilukala
        Bethesda, Maryland | gchilukala2023@fau.edu | (561) 875-3556

        PROFESSIONAL EXPERIENCE
        GEICO - Software Engineer, AI/ML (January 2025 – Present)
        • Architected production-ready LLM backend services using Python and FastAPI
        • Achieved 2.5% accuracy improvement and 50% latency reduction using LLaMA model
        • Developed end-to-end Generative AI platform with intelligent document processing
        • Built LLM evaluation pipeline with 30% processing time reduction
        • Implemented token counting feature across 8 different LLM models

        Florida Atlantic University - Graduate Teaching Assistant (January 2024 – December 2024)
        • Developed and taught advanced AI/ML algorithms
        • Improved student learning outcomes by 40-50%

        PROJECTS
        SafetyMapper - Community Safety Platform using Python Flask and Google Gemini AI
        Pic2Plot - Multi-Agent AI System achieving 40% cost reduction

        EDUCATION
        Master of Science in Computer Science, AI Specialization
        Florida Atlantic University | GPA: 3.96/4.0

        ACHIEVEMENTS
        • Hugging Face Hackathon Winner (Caption Creator Pro)
        • Gangal Family Endowed Graduate Scholarship Award (Top 1% of CS students)
        """
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "I'm having trouble accessing the AI service right now. Please try again later."
        
        try:
            # Make synchronous request using httpx
            with httpx.Client(timeout=25.0) as client:
                response = client.post(
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
                                "content": f"You are helping someone learn about Gayathri Chilukala's background. Answer based on this resume: {resume_context}. Be professional and concise."
                            },
                            {"role": "user", "content": question}
                        ],
                        "max_tokens": 200,
                        "temperature": 0.7
                    }
                )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return self.get_fallback_response(question)
                
        except Exception as e:
            print(f"API call failed: {str(e)}")
            return self.get_fallback_response(question)

    def get_fallback_response(self, question):
        """Fallback responses when API is unavailable"""
        q = question.lower()
        
        if any(word in q for word in ['current', 'job', 'role', 'position', 'work']):
            return "Gayathri is currently a Software Engineer, AI/ML at GEICO in Chevy Chase, Maryland since January 2025. She's architecting production-ready LLM backend services using Python and FastAPI."
            
        elif any(word in q for word in ['skill', 'technology', 'programming', 'technical']):
            return "Gayathri's expertise includes Python, FastAPI, Django, TensorFlow, PyTorch, Hugging Face Transformers, Azure, AWS, React.js, and specializes in LLMs, Generative AI, and Computer Vision."
            
        elif any(word in q for word in ['education', 'degree', 'university', 'school']):
            return "Gayathri has a Master's in Computer Science with AI specialization from Florida Atlantic University (3.96/4.0 GPA) and received the Gangal Family Endowed Graduate Scholarship Award for being in the top 1% of CS students."
            
        elif any(word in q for word in ['project', 'portfolio', 'built', 'created']):
            return "Gayathri has built impressive projects including Pic2Plot (multi-agent AI system with 40% cost reduction), SafetyMapper (community safety platform with Google Gemini AI), and Caption Creator Pro (Hugging Face Hackathon winner)."
            
        elif any(word in q for word in ['achievement', 'award', 'accomplishment', 'recognition']):
            return "Key achievements: Hugging Face Hackathon Winner, Gangal Family Scholarship (top 1% CS students), 50% latency reduction in LLM optimization, and 40-50% improvement in student learning outcomes as TA."
            
        elif any(word in q for word in ['contact', 'email', 'reach', 'hire']):
            return "You can contact Gayathri at gchilukala2023@fau.edu. She's based in Bethesda, Maryland and currently working as an AI/ML Engineer at GEICO."
            
        else:
            return "Gayathri is an AI/ML Software Engineer at GEICO with expertise in LLMs, Python, and cloud platforms. She has a Master's degree with 3.96 GPA, won the Hugging Face Hackathon, and achieved significant performance improvements in production systems. What specific aspect would you like to know more about?"