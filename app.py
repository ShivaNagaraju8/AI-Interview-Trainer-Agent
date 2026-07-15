"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║          AI INTERVIEW TRAINER AGENT — Powered by IBM watsonx.ai                 ║
║          Model: IBM Granite | RAG: LangChain + ChromaDB                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝

AGENT_INSTRUCTIONS:
═══════════════════
Customize the interviewer agent's personality, behavior, and evaluation criteria
below WITHOUT modifying any core application logic.

INTERVIEWER_PERSONA:
  name: "Alex"
  role: "Senior Interview Coach & Technical Interviewer"
  tone: "professional yet encouraging — rigorous but never discouraging"
  style: "Socratic — ask follow-up questions to probe depth of understanding"
  communication: "Clear, concise, structured; uses bullet points for multi-part answers"

DIFFICULTY_LEVELS:
  entry:    "Straightforward, concept-check questions; focus on fundamentals; no trick questions"
  mid:      "Scenario-based questions with one or two gotchas; requires practical experience"
  senior:   "System design, trade-off discussions, leadership scenarios; expects depth"
  expert:   "Architecture decisions at scale, ambiguous problems, no single right answer"

JOB_ROLE_SPECIALIZATIONS:
  software_engineer:    "DSA, system design, OOP, clean code, testing, debugging"
  data_scientist:       "Statistics, ML algorithms, feature engineering, model evaluation, SQL"
  product_manager:      "Prioritization frameworks, metrics, user empathy, roadmapping"
  devops_engineer:      "CI/CD, containerization, IaC, observability, incident response"
  data_engineer:        "Pipelines, data modeling, warehousing, Spark, SQL optimization"
  ml_engineer:          "MLOps, model deployment, monitoring, training pipelines, scaling"
  frontend_engineer:    "JavaScript/TypeScript, React, performance, accessibility, CSS"
  backend_engineer:     "APIs, databases, microservices, security, scalability"
  full_stack:           "End-to-end features, frontend + backend, deployment"
  default:              "General software engineering principles and professional skills"

EVALUATION_CRITERIA:
  technical_accuracy:   30%   # Correctness of technical concepts and solutions
  communication:        20%   # Clarity, structure, vocabulary, conciseness
  problem_solving:      25%   # Approach, methodology, handling of unknowns
  depth_of_knowledge:   15%   # Beyond surface level, handles follow-ups
  confidence:           10%   # Composure, owns mistakes, avoids excessive hedging

FEEDBACK_STYLE:
  format:     "sandwich (strength → improvement → encouragement)"
  specificity: "always cite specific phrases/concepts from the candidate's response"
  tone:       "coach not judge — treat every answer as a starting point, not a verdict"
  length:     "comprehensive but scannable; use headers and bullets"

SCORING_METHODOLOGY:
  scale:          1-10 per question
  passing_score:  7
  overall_band:
    9-10: "Exceptional — hire signal"
    7-8:  "Strong — likely proceed"
    5-6:  "Borderline — needs coaching"
    3-4:  "Below bar — significant gaps"
    1-2:  "Not ready — fundamental gaps"

SAFETY_POLICIES:
  - "Never ask or encourage discussion of illegal discrimination"
  - "Decline personal/irrelevant questions (age, religion, marital status)"
  - "Do not generate or engage with harmful, violent, or offensive content"
  - "Maintain professional language at all times"
  - "If user is distressed, acknowledge and offer to pause/reschedule"

COACHING_BEHAVIOR:
  hints:          "Offer hints after 2 attempts on technical problems, never just give the answer"
  encouragement:  "Celebrate improvements explicitly, even small ones"
  redirection:    "Gently redirect off-topic responses back to the question"
  pacing:         "Allow 30-120 seconds per question depending on complexity"
  follow_ups:     "Always ask at least one follow-up to test depth, avoid surface pass"
═══════════════════════════════════════════════════════════════════════════════════
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import re
import json
import time
import hashlib
import datetime
from pathlib import Path
from io import BytesIO

# ─── Third-party ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# ─── IBM watsonx.ai ──────────────────────────────────────────────────────────
try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False

# ─── LangChain + RAG ─────────────────────────────────────────────────────────
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ─── Document parsing ────────────────────────────────────────────────────────
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ─── PDF Report Generation ───────────────────────────────────────────────────
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

load_dotenv()

WATSONX_API_KEY   = os.getenv("WATSONX_API_KEY", "uJHsRaKZjv9v1elMFh-gqInhD57p4bydXrQYhZqz5gxs")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "db9f6a4f-6f1b-42a9-ba48-4eda715657c")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
GRANITE_MODEL_ID   = os.getenv("GRANITE_MODEL_ID", "ibm/granite-4-h-small")
KB_PATH       = Path("knowledge_base")
CHROMA_PATH   = Path(".chroma_db")
REPORTS_PATH  = Path("reports")
REPORTS_PATH.mkdir(exist_ok=True)

ROLES = [
    "Software Engineer", "Data Scientist", "Product Manager",
    "DevOps Engineer", "Data Engineer", "ML Engineer",
    "Frontend Engineer", "Backend Engineer", "Full Stack Engineer",
    "Cloud Architect", "Security Engineer", "QA Engineer", "Other"
]

EXPERIENCE_LEVELS = ["Entry Level (0-2 yrs)", "Mid Level (3-5 yrs)",
                     "Senior Level (6-9 yrs)", "Expert / Staff+ (10+ yrs)"]

INTERVIEW_TYPES = ["Technical", "HR / Behavioral", "System Design",
                   "Coding Challenge", "Mixed (Full Round)"]

# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Interview Trainer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — RESPONSIVE, DARK/LIGHT MODE AWARE
# ════════════════════════════════════════════════════════════════════════════

def inject_css(dark_mode: bool) -> None:
    bg        = "#0e1117" if dark_mode else "#ffffff"
    surface   = "#1a1f2e" if dark_mode else "#f7f8fa"
    card      = "#242938" if dark_mode else "#ffffff"
    border    = "#2d3348" if dark_mode else "#e5e7eb"
    text      = "#e8eaf6" if dark_mode else "#1f2328"
    muted     = "#8b92a9" if dark_mode else "#57606a"
    accent    = "#4f8ef7" if dark_mode else "#3b82d4"
    accent2   = "#7c5cd8"
    success   = "#22c55e"
    warning   = "#f59e0b"
    danger    = "#ef4444"
    user_msg  = "#1e3a5f" if dark_mode else "#dbeafe"
    ai_msg    = "#1e2d1f" if dark_mode else "#f0fdf4"

    st.markdown(f"""
    <style>
    /* ── Global ── */
    html, body, [class*="css"] {{
        font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
        font-size: 14px;
    }}
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }}

    /* ── Header banner ── */
    .app-header {{
        background: linear-gradient(135deg, {accent} 0%, {accent2} 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        color: #fff;
    }}
    .app-header h1 {{ margin: 0; font-size: 1.8rem; font-weight: 700; }}
    .app-header p  {{ margin: 0.25rem 0 0; opacity: 0.88; font-size: 0.95rem; }}

    /* ── Cards ── */
    .card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .metric-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {accent};
    }}
    .metric-label {{
        font-size: 0.8rem;
        color: {muted};
        margin-top: 0.25rem;
    }}

    /* ── Chat bubbles ── */
    .chat-container {{
        max-height: 500px;
        overflow-y: auto;
        padding: 0.5rem;
        border: 1px solid {border};
        border-radius: 10px;
        background: {surface};
    }}
    .chat-msg {{
        display: flex;
        margin-bottom: 1rem;
        align-items: flex-start;
        gap: 0.75rem;
    }}
    .chat-msg.user   {{ flex-direction: row-reverse; }}
    .avatar {{
        width: 36px; height: 36px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0; font-weight: 700;
    }}
    .avatar.ai   {{ background: {accent2}; color: #fff; }}
    .avatar.user {{ background: {accent};  color: #fff; }}
    .bubble {{
        max-width: 80%;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        line-height: 1.6;
        font-size: 0.9rem;
    }}
    .bubble.ai   {{ background: {ai_msg};   border: 1px solid {border}; color: {text}; border-radius: 2px 12px 12px 12px; }}
    .bubble.user {{ background: {user_msg}; border: 1px solid {border}; color: {text}; border-radius: 12px 2px 12px 12px; }}
    .msg-time {{ font-size: 0.7rem; color: {muted}; margin-top: 0.25rem; text-align: right; }}

    /* ── Score badge ── */
    .score-badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .score-high   {{ background: #dcfce7; color: #166534; }}
    .score-med    {{ background: #fef9c3; color: #854d0e; }}
    .score-low    {{ background: #fee2e2; color: #991b1b; }}

    /* ── Progress bar ── */
    .prog-bar-wrap {{
        background: {border};
        border-radius: 6px;
        height: 8px;
        margin: 0.4rem 0;
    }}
    .prog-bar-fill {{
        height: 8px;
        border-radius: 6px;
        background: linear-gradient(90deg, {accent}, {accent2});
    }}

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {{
        background: {surface};
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-size: 0.9rem;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.1s;
    }}
    .stButton > button:hover {{ transform: translateY(-1px); }}

    /* ── Section headers ── */
    .section-header {{
        border-left: 4px solid {accent};
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: {text};
    }}

    /* ── Tag pill ── */
    .tag {{
        display: inline-block;
        background: {surface};
        border: 1px solid {border};
        border-radius: 20px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        color: {muted};
        margin: 0.1rem;
    }}

    /* ── IBM Badge ── */
    .ibm-badge {{
        font-size: 0.7rem;
        color: {muted};
        text-align: center;
        margin-top: 0.5rem;
    }}

    /* ── Responsive ── */
    @media (max-width: 768px) {{
        .bubble {{ max-width: 95%; }}
        .app-header h1 {{ font-size: 1.3rem; }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ════════════════════════════════════════════════════════════════════════════

def init_session() -> None:
    defaults = {
        "dark_mode":        True,
        "chat_history":     [],
        "session_scores":   [],
        "questions_asked":  0,
        "current_question": None,
        "interview_active": False,
        "resume_text":      "",
        "resume_skills":    [],
        "job_role":         "Software Engineer",
        "experience":       "Mid Level (3-5 yrs)",
        "interview_type":   "Technical",
        "difficulty":       "mid",
        "rag_initialized":  False,
        "vector_store":     None,
        "session_start":    datetime.datetime.now().isoformat(),
        "all_sessions":     [],
        "current_tab":      "Dashboard",
        "feedback_queue":   [],
        "study_topics":     [],
        "user_name":        "Candidate",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
# IBM WATSONX.AI CLIENT
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_watsonx_model(api_key: str, project_id: str, url: str, model_id: str):
    """Initialise and cache the IBM Granite model client."""
    if not WATSONX_AVAILABLE:
        return None
    if not api_key or api_key == "your_ibm_cloud_api_key_here":
        return None
    try:
        credentials = Credentials(url=url, api_key=api_key)
        client = APIClient(credentials)
        model = ModelInference(
            model_id=model_id,
            api_client=client,
            project_id=project_id,
            params={
                GenParams.MAX_NEW_TOKENS:  1024,
                GenParams.MIN_NEW_TOKENS:  10,
                GenParams.TEMPERATURE:     0.7,
                GenParams.TOP_P:           0.9,
                GenParams.TOP_K:           50,
                GenParams.REPETITION_PENALTY: 1.1,
                GenParams.STOP_SEQUENCES:  ["Human:", "User:", "\n\n\n"],
            }
        )
        return model
    except Exception as e:
        st.warning(f"watsonx.ai connection issue: {e}")
        return None


def call_granite(prompt: str, model) -> str:
    """Call IBM Granite model; fallback to mock if unavailable."""
    if model is None:
        return _mock_response(prompt)
    try:
        response = model.generate_text(prompt=prompt)
        return response.strip() if isinstance(response, str) else str(response)
    except Exception as e:
        return f"[Model error: {e}]\n\n{_mock_response(prompt)}"


# ── Demo-mode question bank (used when watsonx.ai is not configured) ──────────
# Keyed by interview type keyword found in the prompt.
_DEMO_QUESTIONS: dict[str, list[str]] = {
    "technical": [
        "What is the difference between a stack and a queue? Give a real-world use case for each.",
        "Explain the concept of Big-O notation. What is the time complexity of binary search and why?",
        "What is a hash table and how does it handle collisions? Compare chaining vs open addressing.",
        "Describe the differences between SQL and NoSQL databases. When would you choose each?",
        "What are SOLID principles? Pick any two and explain them with a code example.",
        "Explain the difference between concurrency and parallelism. Give an example of each.",
        "What is a RESTful API? How does it differ from GraphQL?",
        "Describe how garbage collection works in a language of your choice.",
        "What is a deadlock? How do you detect and prevent it?",
        "Explain the CAP theorem and give an example of a CP and an AP system.",
        "What is the difference between TCP and UDP? When would you use UDP?",
        "Describe the MVC architectural pattern and its benefits.",
        "What is memoization? Write a simple example in any language.",
        "Explain database indexing. What are the trade-offs of adding too many indexes?",
        "What is the difference between an interface and an abstract class?",
        "How does a load balancer work? Name two load-balancing strategies.",
        "Explain event-driven architecture. What problem does it solve?",
        "What is Docker and how does containerization differ from virtualization?",
        "Describe the difference between authentication and authorization.",
        "What is a race condition? How would you reproduce and fix one?",
    ],
    "hr": [
        "Tell me about yourself and why you're interested in this role.",
        "What is your greatest professional strength? Give a specific example.",
        "Describe a time you had to meet a very tight deadline. How did you manage it?",
        "Where do you see yourself professionally in five years?",
        "Tell me about a time you received critical feedback. How did you respond?",
        "Why are you leaving your current position?",
        "What motivates you most in your day-to-day work?",
        "Describe a situation where you had to persuade someone who disagreed with you.",
        "What is your biggest professional weakness and what are you doing about it?",
        "Tell me about a time you had to adapt quickly to a significant change.",
        "How do you prioritize when you have multiple urgent tasks at the same time?",
        "Describe your ideal work environment and team culture.",
        "Tell me about a time you took initiative beyond your job description.",
        "How do you handle working with someone whose style is very different from yours?",
        "What accomplishment are you most proud of in your career so far?",
    ],
    "behavioral": [
        "Tell me about a time you led a team through a difficult project. What was your approach?",
        "Describe a situation where you had to make a decision with incomplete information.",
        "Tell me about the most challenging technical problem you've solved. Walk me through it.",
        "Give an example of a time you disagreed with your manager. What did you do?",
        "Describe a time you failed. What did you learn and how did you change as a result?",
        "Tell me about a time you went above and beyond what was expected of you.",
        "Describe a situation where you had to collaborate with a difficult team member.",
        "Give an example of a time you had to learn a new skill quickly.",
        "Tell me about a time you identified and fixed a process inefficiency.",
        "Describe a project where you had to balance competing stakeholder priorities.",
        "Tell me about a time you had to deliver bad news to a client or manager.",
        "Give an example of using data to drive a decision you made.",
        "Describe a time you mentored or coached someone on your team.",
        "Tell me about a time you had to defend a technical decision under pressure.",
        "Describe a situation where you had to handle ambiguity and still deliver results.",
    ],
    "system design": [
        "Design a URL shortener like bit.ly. Walk through storage, hashing, and scale.",
        "How would you design a notification service that delivers to 50 million users?",
        "Design a rate limiter for a public API. What algorithms would you consider?",
        "Walk me through designing a distributed key-value store like Redis.",
        "How would you design the backend for a real-time collaborative document editor?",
        "Design a search autocomplete system that returns results in under 100ms.",
        "How would you architect a video streaming platform to handle 10M concurrent users?",
        "Design a job scheduler that runs millions of tasks reliably across many servers.",
        "How would you build a distributed logging and monitoring system?",
        "Design a ride-sharing matching system like Uber for a city of 5 million people.",
        "How would you design a content delivery network (CDN) from scratch?",
        "Walk me through designing an e-commerce cart and checkout system.",
        "Design a leaderboard system that updates in real-time for a mobile game.",
        "How would you build a feature flag system to roll out features gradually?",
        "Design an event-driven order processing pipeline for an online marketplace.",
    ],
    "coding": [
        "Given a string, write a function to check if it is a palindrome (ignore spaces and case).",
        "Write a function that finds the two numbers in an array that sum to a target value.",
        "Implement a function to reverse a linked list iteratively and recursively.",
        "Write a function to find the longest substring without repeating characters.",
        "Given a sorted array, implement binary search. What is its time complexity?",
        "Write a function to check if two strings are anagrams of each other.",
        "Implement a stack that supports push, pop, and retrieving the minimum in O(1).",
        "Write a function to merge two sorted arrays into one sorted array.",
        "Given a binary tree, write a function to return all values level by level (BFS).",
        "Write a function to determine if a string of brackets is valid (e.g. `{[()]}`).",
        "Implement Fibonacci with memoization. Compare with the naive recursive approach.",
        "Write a function to remove all duplicates from a sorted linked list.",
        "Given a matrix, rotate it 90 degrees clockwise in place.",
        "Write a function to find the maximum sum contiguous subarray (Kadane's algorithm).",
        "Given an array, find the first non-repeating character. Explain your approach.",
    ],
    "mixed": [
        "Walk me through how the internet works when you type a URL into a browser.",
        "What is the difference between synchronous and asynchronous programming?",
        "Describe a past project you're proud of. What were the biggest technical challenges?",
        "How do you stay up to date with new technologies? Give a recent example.",
        "Tell me about a time you had to debug a hard-to-reproduce production issue.",
        "What does good code look like to you? How do you ensure quality in your work?",
        "Describe a time you had to rewrite or significantly refactor a system.",
        "How do you approach estimating how long a task will take?",
        "Tell me about a situation where you disagreed with an architectural decision.",
        "What's the most interesting thing you've built in your own time?",
        "How do you approach onboarding to a new codebase you've never seen before?",
        "Describe your testing philosophy. How much coverage is enough?",
        "Tell me about a time you had to advocate for technical debt reduction.",
        "How do you handle feedback during a code review — giving and receiving?",
        "Describe a time when a project requirement changed late. How did you adapt?",
    ],
}

# Varied demo feedback pool keyed by score band
_DEMO_FEEDBACK = [
    {
        "score": 8,
        "text": (
            "## ✅ Strengths\n"
            "- You clearly articulated the core concept and demonstrated practical understanding.\n"
            "- Your answer was well-structured and easy to follow.\n\n"
            "## 🔧 Areas to Improve\n"
            "- Add a concrete, quantified example from your own experience (e.g., 'I reduced latency by 40% by...').\n"
            "- Explore edge cases or trade-offs — interviewers probe deeper after a solid opening.\n\n"
            "## 💡 Model Answer Highlights\n"
            "- Define the concept precisely before applying it.\n"
            "- Give a real-world scenario with scale in mind.\n"
            "- Acknowledge trade-offs and when you'd choose a different approach.\n\n"
            "## 📚 Recommended Study Topics\n"
            "- Practice explaining technical concepts to non-technical audiences.\n"
            "- Study system design trade-offs for the role's common use cases.\n\n"
            "## 🎯 Score: 8/10\n"
            "**Band:** Strong — likely proceed\n"
            "**Rationale:** Solid grasp of fundamentals with room to add depth and real examples.\n\n"
            "## 💪 Encouragement\n"
            "Great work — your structured thinking comes through clearly. Focus on weaving in "
            "specific numbers and outcomes to push into the Exceptional band.\n\n"
            "*[Demo Mode — add IBM watsonx.ai credentials for personalised AI feedback.]*"
        ),
    },
    {
        "score": 6,
        "text": (
            "## ✅ Strengths\n"
            "- You identified the right area and showed familiarity with the topic.\n"
            "- Your answer had a logical flow.\n\n"
            "## 🔧 Areas to Improve\n"
            "- Your answer stays at the surface — dig one level deeper into the 'why'.\n"
            "- Use the STAR method (Situation → Task → Action → Result) for more impact.\n"
            "- Quantify your results: numbers make answers memorable and credible.\n\n"
            "## 💡 Model Answer Highlights\n"
            "- Open with a clear, direct definition or stance.\n"
            "- Support with a specific example from your experience.\n"
            "- Close with what you would do differently or what you learned.\n\n"
            "## 📚 Recommended Study Topics\n"
            "- STAR method practice with your top 5 career stories.\n"
            "- Review fundamental concepts for the role's core domain.\n\n"
            "## 🎯 Score: 6/10\n"
            "**Band:** Borderline — needs coaching\n"
            "**Rationale:** Correct direction but needs more depth and concrete evidence.\n\n"
            "## 💪 Encouragement\n"
            "You're on the right track — the foundation is there. With a bit more "
            "structure and a real example, this answer moves from borderline to strong.\n\n"
            "*[Demo Mode — add IBM watsonx.ai credentials for personalised AI feedback.]*"
        ),
    },
    {
        "score": 9,
        "text": (
            "## ✅ Strengths\n"
            "- Excellent depth — you went beyond the surface and covered trade-offs.\n"
            "- Your example was specific, relevant, and had a clear outcome.\n"
            "- Strong command of vocabulary signals genuine expertise.\n\n"
            "## 🔧 Areas to Improve\n"
            "- Consider briefly acknowledging alternative approaches and why you ruled them out.\n"
            "- A quick summary sentence at the end would make this answer even more polished.\n\n"
            "## 💡 Model Answer Highlights\n"
            "- Lead with a crisp definition, then contextualise with a real scenario.\n"
            "- Discuss trade-offs to show architectural maturity.\n"
            "- Quantify impact whenever possible.\n\n"
            "## 📚 Recommended Study Topics\n"
            "- Explore adjacent concepts that interviewers often follow up with.\n"
            "- Practice handling curveball follow-up questions on this topic.\n\n"
            "## 🎯 Score: 9/10\n"
            "**Band:** Exceptional — hire signal\n"
            "**Rationale:** Near-complete answer with strong depth and real-world evidence.\n\n"
            "## 💪 Encouragement\n"
            "Outstanding answer — this is exactly what interviewers want to see. "
            "One more run-through to add the alternative-approach acknowledgment and this is flawless.\n\n"
            "*[Demo Mode — add IBM watsonx.ai credentials for personalised AI feedback.]*"
        ),
    },
    {
        "score": 5,
        "text": (
            "## ✅ Strengths\n"
            "- You attempted to address the question and showed awareness of the topic.\n\n"
            "## 🔧 Areas to Improve\n"
            "- The answer lacks specific examples — everything is too general.\n"
            "- Focus on the core mechanism, not just the high-level concept.\n"
            "- Structure your answer: define → example → trade-off → summary.\n\n"
            "## 💡 Model Answer Highlights\n"
            "- Start by clearly defining the term or concept asked about.\n"
            "- Give a concrete code-level or system-level example.\n"
            "- Mention at least one scenario where this approach would NOT be ideal.\n\n"
            "## 📚 Recommended Study Topics\n"
            "- Deep-dive into the fundamentals of this topic (textbook + practice problems).\n"
            "- Code a small project that uses this concept end-to-end.\n"
            "- Review common interview questions on this topic on LeetCode / System Design Primer.\n\n"
            "## 🎯 Score: 5/10\n"
            "**Band:** Borderline — needs coaching\n"
            "**Rationale:** Awareness present but depth and specificity are missing.\n\n"
            "## 💪 Encouragement\n"
            "Don't be discouraged — recognising the concept is the first step. "
            "Focus on one or two study resources this week and revisit this question type.\n\n"
            "*[Demo Mode — add IBM watsonx.ai credentials for personalised AI feedback.]*"
        ),
    },
]

_DEMO_STUDY_ROADMAP = """\
## 🗓️ Week-by-Week Study Plan

### Week 1: Foundations
- Review core data structures (arrays, linked lists, trees, graphs, hash tables)
- Brush up on time/space complexity analysis (Big-O)
- Read chapters 1–4 of *Cracking the Coding Interview* or *DDIA* (for system design roles)
- Solve 5 Easy LeetCode problems per day

### Week 2: Core Competencies
- Focus on your role's key domain (e.g., system design, ML fundamentals, SQL)
- Prepare 8–10 STAR stories covering leadership, failure, collaboration, and impact
- Practice explaining technical concepts out loud (use a rubber duck or record yourself)
- Solve 3 Medium LeetCode problems per day

### Week 3: Practice & Application
- Do 3 timed mock coding sessions (45 min each) — treat them like real interviews
- Practice 2 system design questions end-to-end (whiteboard or paper)
- Role-play behavioral questions with a friend or in front of a mirror
- Research your target company: products, culture, recent news, tech stack

### Week 4: Mock Interviews & Polish
- Do 2 full mock interviews (coding + behavioral) with feedback
- Review all your STAR stories and tighten them to under 2 minutes each
- Prepare 5–7 thoughtful questions to ask interviewers
- Review your weakest areas one more time, then rest the day before

## 📌 Priority Topics for This Role
- Data structures & algorithms (interview staple regardless of role)
- System design fundamentals (scalability, databases, caching, queues)
- Behavioral/STAR storytelling (underestimated but high-impact)
- Domain-specific knowledge (language, framework, or tools for your role)
- Communication under pressure (practice thinking aloud)

## 🔗 Recommended Resources
- **LeetCode** — coding practice (filter by company + topic)
- **System Design Primer** (GitHub) — free, comprehensive system design guide
- **Cracking the Coding Interview** — McDowell (classic CS interview prep)
- **Designing Data-Intensive Applications** — Kleppmann (deep system design)
- **Behavioral Interview Prep** — Jeff H. Sipe on YouTube
- **Glassdoor** — real interview question reports for your target companies
- **levels.fyi** — compensation benchmarks to anchor salary negotiation

## ✅ Daily Practice Checklist
- [ ] Solve 1–3 coding problems (timed, no hints for the first 20 min)
- [ ] Review flashcards for 1 core concept (e.g., one design pattern, one DB concept)
- [ ] Write or refine one STAR story
- [ ] Read one article about your target company or industry
- [ ] Spend 10 min reviewing yesterday's mistakes

*[Demo Mode — add IBM watsonx.ai credentials for a fully personalised roadmap.]*
"""


def _mock_response(prompt: str) -> str:
    """
    Varied demo responses when watsonx.ai credentials are not configured.
    Uses prompt content + a time-based seed to rotate through a large question
    pool so the same question never repeats within a session.
    """
    p = prompt.lower()

    # ── Roadmap / study plan ──────────────────────────────────────────────────
    if "roadmap" in p or "study plan" in p or "week" in p:
        return _DEMO_STUDY_ROADMAP

    # ── Feedback / evaluation ─────────────────────────────────────────────────
    if "feedback" in p or "evaluat" in p or "strengths" in p or "score" in p:
        # Rotate through feedback variants using a hash of the answer text
        idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(_DEMO_FEEDBACK)
        return _DEMO_FEEDBACK[idx]["text"]

    # ── Question generation ───────────────────────────────────────────────────
    if "question" in p or "ask" in p or "generate" in p or "[inst]" in p:
        # Detect interview type from the prompt text
        if "system design" in p:
            pool = _DEMO_QUESTIONS["system design"]
        elif "behavioral" in p or "hr" in p or "human resource" in p:
            # Mix HR + behavioral
            pool = _DEMO_QUESTIONS["hr"] + _DEMO_QUESTIONS["behavioral"]
        elif "coding" in p or "challenge" in p:
            pool = _DEMO_QUESTIONS["coding"]
        elif "mixed" in p:
            pool = _DEMO_QUESTIONS["mixed"]
        else:
            # Technical is the default
            pool = _DEMO_QUESTIONS["technical"]

        # Pick a question not yet asked this session using session state tracker.
        # Falls back to hash if called outside Streamlit context (e.g. tests).
        pool_key = ("system design" if "system design" in p
                    else "hr_behavioral" if ("behavioral" in p or "hr" in p)
                    else "coding" if ("coding" in p or "challenge" in p)
                    else "mixed" if "mixed" in p
                    else "technical")
        try:
            asked: set = st.session_state.setdefault(
                "demo_asked_indices", {}
            ).setdefault(pool_key, set())
            available = [i for i in range(len(pool)) if i not in asked]
            if not available:           # full cycle — reset and start over
                asked.clear()
                available = list(range(len(pool)))
            # Pick deterministically from available using prompt hash as tiebreaker
            h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
            idx = available[h % len(available)]
            asked.add(idx)
        except Exception:
            idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(pool)
        q = pool[idx]
        return (
            f"{q}\n\n"
            "*[Demo Mode — configure `WATSONX_API_KEY` in `.env` for real IBM Granite responses.]*"
        )

    # ── Fallback ──────────────────────────────────────────────────────────────
    return (
        "*[Demo Mode — IBM watsonx.ai not configured.]*\n\n"
        "Add your `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` to the `.env` file "
        "to enable full AI-powered responses."
    )


# ════════════════════════════════════════════════════════════════════════════
# RAG — VECTOR STORE (ChromaDB + HuggingFace Embeddings)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_vector_store(_kb_path: str, _chroma_path: str):
    """Load knowledge-base documents, chunk them, and embed into ChromaDB."""
    if not LANGCHAIN_AVAILABLE:
        return None
    kb = Path(_kb_path)
    chroma = Path(_chroma_path)

    # Collect all .txt files
    docs: list[Document] = []
    for txt_file in kb.rglob("*.txt"):
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs.extend(loader.load())
        except Exception:
            pass

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks   = splitter.split_documents(docs)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(chroma),
        )
        return vs
    except Exception as e:
        st.warning(f"RAG init warning: {e}")
        return None


def rag_retrieve(query: str, vector_store, k: int = 4) -> str:
    """Return top-k relevant chunks from the vector store."""
    if vector_store is None:
        return ""
    try:
        results = vector_store.similarity_search(query, k=k)
        return "\n\n---\n\n".join(d.page_content for d in results)
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════════════════════
# RESUME PARSING
# ════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PDF_AVAILABLE:
        return ""
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    if not DOCX_AVAILABLE:
        return ""
    try:
        doc = DocxDocument(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


SKILL_KEYWORDS = {
    "Languages":   ["python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
                    "kotlin", "swift", "ruby", "scala", "r", "matlab", "php"],
    "Frameworks":  ["react", "angular", "vue", "django", "flask", "fastapi", "spring", "node.js",
                    "express", "nextjs", "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost"],
    "Databases":   ["postgresql", "mysql", "mongodb", "redis", "sqlite", "cassandra", "dynamodb",
                    "elasticsearch", "bigquery", "snowflake", "oracle"],
    "Cloud":       ["aws", "azure", "gcp", "ibm cloud", "docker", "kubernetes", "terraform",
                    "ansible", "ci/cd", "jenkins", "github actions", "helm"],
    "Data/AI/ML":  ["machine learning", "deep learning", "nlp", "computer vision", "data science",
                    "pandas", "numpy", "spark", "kafka", "airflow", "mlflow", "langchain"],
    "Soft Skills": ["leadership", "communication", "teamwork", "problem-solving", "agile",
                    "scrum", "project management", "mentoring", "collaboration"],
}


def extract_skills(text: str) -> dict[str, list[str]]:
    text_lower = text.lower()
    found: dict[str, list[str]] = {}
    for category, skills in SKILL_KEYWORDS.items():
        matched = [s for s in skills if s in text_lower]
        if matched:
            found[category] = matched
    return found


# ════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def build_question_prompt(role: str, experience: str, interview_type: str,
                           difficulty: str, resume_skills: list, rag_context: str,
                           questions_asked: int) -> str:
    skills_str = ", ".join(resume_skills[:15]) if resume_skills else "general software skills"
    return f"""<s>[INST] You are Alex, a Senior Interview Coach at a top technology company.
Your role: Generate a single, high-quality {interview_type} interview question.

AGENT_INSTRUCTIONS context:
- Difficulty: {difficulty} level
- Tone: professional yet encouraging
- Style: Socratic — ask questions that probe genuine understanding
- Safety: No discriminatory, personal, or harmful questions

CANDIDATE PROFILE:
- Target Role: {role}
- Experience Level: {experience}
- Known Skills from Resume: {skills_str}
- Questions already asked this session: {questions_asked}

RELEVANT KNOWLEDGE BASE CONTEXT:
{rag_context[:1200] if rag_context else "Use your general knowledge of {interview_type} interviews."}

TASK:
Generate ONE well-crafted {interview_type} interview question appropriate for a {difficulty}-level {role}.
- If this is question #{questions_asked + 1}, make it appropriately different in topic from earlier questions
- Include a brief (1-sentence) context or scenario if helpful
- Do NOT include the answer or hints

Return ONLY the question, nothing else. [/INST]"""


def build_feedback_prompt(question: str, answer: str, role: str,
                           experience: str, rag_context: str) -> str:
    return f"""<s>[INST] You are Alex, a Senior Interview Coach evaluating a candidate's response.

AGENT_INSTRUCTIONS context (Feedback Style):
- Format: sandwich (strength → improvement → encouragement)
- Specificity: cite specific phrases from the candidate's answer
- Tone: coach, not judge — treat every answer as a starting point
- Scoring: 1-10 scale where 7+ is passing

CANDIDATE PROFILE:
- Target Role: {role}
- Experience Level: {experience}

INTERVIEW QUESTION:
{question}

CANDIDATE'S ANSWER:
{answer}

RELEVANT KNOWLEDGE BASE:
{rag_context[:800] if rag_context else "Use your expert knowledge."}

EVALUATION CRITERIA (per AGENT_INSTRUCTIONS):
- Technical Accuracy: 30%
- Communication Clarity: 20%
- Problem-Solving Approach: 25%
- Depth of Knowledge: 15%
- Confidence / Ownership: 10%

Provide structured feedback in this EXACT format:

## ✅ Strengths
[2-3 specific strengths citing their exact words or approach]

## 🔧 Areas to Improve
[2-3 specific, actionable improvements with concrete examples]

## 💡 Model Answer Highlights
[Key points a strong answer should include — 3-5 bullet points]

## 📚 Recommended Study Topics
[2-3 specific topics/resources to study]

## 🎯 Score: [X]/10
**Band:** [Exceptional/Strong/Borderline/Below bar/Not ready]
**Rationale:** [1-2 sentences explaining the score]

## 💪 Encouragement
[1-2 sentences of personalized encouragement referencing their answer] [/INST]"""


def build_study_roadmap_prompt(role: str, experience: str, weak_areas: list,
                                skills: list, rag_context: str) -> str:
    return f"""<s>[INST] You are Alex, a Senior Interview Coach creating a personalized study roadmap.

CANDIDATE PROFILE:
- Target Role: {role}
- Experience Level: {experience}
- Current Skills: {', '.join(skills[:10]) if skills else 'not specified'}
- Identified Weak Areas: {', '.join(weak_areas) if weak_areas else 'general preparation'}

RELEVANT CONTEXT:
{rag_context[:600] if rag_context else ""}

Create a structured 4-week interview preparation roadmap with:

## 🗓️ Week-by-Week Study Plan

### Week 1: Foundations
[3-4 specific topics with recommended resources]

### Week 2: Core Competencies
[3-4 specific topics with recommended resources]

### Week 3: Practice & Application
[3-4 specific practice activities]

### Week 4: Mock Interviews & Polish
[3-4 final preparation activities]

## 📌 Priority Topics for {role}
[5-7 must-know topics for this specific role]

## 🔗 Recommended Resources
[5-7 specific books, websites, courses, or platforms]

## ✅ Daily Practice Checklist
[5 daily habits for interview prep] [/INST]"""


def build_system_design_prompt(role: str, experience: str, rag_context: str) -> str:
    return f"""<s>[INST] You are Alex, a Senior Interview Coach.
Generate ONE system design interview question appropriate for a {experience} {role}.

CONTEXT:
{rag_context[:600] if rag_context else ""}

Requirements:
- Include a specific scale/constraint (e.g., "design for 10M daily active users")
- The question should be open-ended with multiple valid approaches
- Appropriate complexity for {experience}

Return ONLY the question with the scale parameters. [/INST]"""


# ════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATION
# ════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(user_name: str, role: str, experience: str,
                         scores: list[int], chat_history: list,
                         skills: list) -> bytes:
    if not FPDF_AVAILABLE:
        return b""
    try:
        pdf = FPDF()
        pdf.add_page()

        # Header
        pdf.set_fill_color(59, 130, 212)
        pdf.rect(0, 0, 210, 35, "F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_y(10)
        pdf.cell(0, 10, "AI Interview Trainer — Performance Report", align="C", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Powered by IBM watsonx.ai | IBM Granite Model", align="C", ln=True)

        # Meta
        pdf.set_text_color(30, 30, 30)
        pdf.set_y(45)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Candidate Overview", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 6, f"Name: {user_name}", ln=True)
        pdf.cell(0, 6, f"Target Role: {role}", ln=True)
        pdf.cell(0, 6, f"Experience Level: {experience}", ln=True)
        pdf.cell(0, 6, f"Report Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}", ln=True)

        # Scores summary
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Performance Summary", ln=True)
        pdf.set_font("Helvetica", "", 11)
        if scores:
            avg = sum(scores) / len(scores)
            pdf.cell(0, 6, f"Questions Attempted: {len(scores)}", ln=True)
            pdf.cell(0, 6, f"Average Score: {avg:.1f}/10", ln=True)
            pdf.cell(0, 6, f"Highest Score: {max(scores)}/10", ln=True)
            pdf.cell(0, 6, f"Lowest Score: {min(scores)}/10", ln=True)
            band = ("Exceptional" if avg >= 9 else "Strong" if avg >= 7
                    else "Borderline" if avg >= 5 else "Below Bar")
            pdf.cell(0, 6, f"Overall Band: {band}", ln=True)
        else:
            pdf.cell(0, 6, "No scored questions yet.", ln=True)

        # Skills
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Detected Skills from Resume", ln=True)
        pdf.set_font("Helvetica", "", 10)
        if skills:
            skill_text = ", ".join(skills[:20])
            pdf.multi_cell(0, 6, skill_text)
        else:
            pdf.cell(0, 6, "No resume uploaded.", ln=True)

        # Chat history (last 10 exchanges)
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Interview Session Transcript (Recent)", ln=True)
        pdf.set_font("Helvetica", "", 9)
        transcript = [m for m in chat_history if m["role"] != "system"][-20:]
        for msg in transcript:
            role_label = "Interviewer" if msg["role"] == "assistant" else "Candidate"
            content    = msg["content"][:300] + ("..." if len(msg["content"]) > 300 else "")
            safe_content = content.encode("latin-1", "replace").decode("latin-1")
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, f"{role_label}:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, safe_content)
            pdf.ln(2)

        return bytes(pdf.output())
    except Exception as e:
        st.warning(f"PDF generation error: {e}")
        return b""


# ════════════════════════════════════════════════════════════════════════════
# SCORE PARSING HELPER
# ════════════════════════════════════════════════════════════════════════════

def parse_score(feedback_text: str) -> int | None:
    patterns = [
        r"Score[:\s]*(\d+)\s*/\s*10",
        r"(\d+)\s*/\s*10",
        r"score[:\s]*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, feedback_text, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 10:
                return v
    return None


def score_badge(score: int) -> str:
    if score >= 8:
        return f'<span class="score-badge score-high">{score}/10</span>'
    elif score >= 6:
        return f'<span class="score-badge score-med">{score}/10</span>'
    else:
        return f'<span class="score-badge score-low">{score}/10</span>'


def progress_bar(value: float, max_value: float = 10.0) -> str:
    pct = min(100, int((value / max_value) * 100))
    return (f'<div class="prog-bar-wrap">'
            f'<div class="prog-bar-fill" style="width:{pct}%"></div></div>')


# ════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    st.markdown("""
    <div class="app-header">
        <h1>🎯 AI Interview Trainer Agent</h1>
        <p>Powered by IBM watsonx.ai &nbsp;•&nbsp; IBM Granite Model &nbsp;•&nbsp;
           LangChain RAG &nbsp;•&nbsp; ChromaDB</p>
    </div>
    """, unsafe_allow_html=True)


def render_chat_message(role: str, content: str, timestamp: str = "") -> None:
    is_user   = role == "user"
    css_role  = "user" if is_user else "ai"
    avatar    = "👤" if is_user else "🤖"
    label     = "You" if is_user else "Alex (AI Coach)"
    ts_html   = f'<div class="msg-time">{timestamp}</div>' if timestamp else ""

    safe = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    st.markdown(f"""
    <div class="chat-msg {css_role}">
        <div class="avatar {css_role}">{avatar}</div>
        <div>
            <div style="font-size:0.75rem;color:#888;margin-bottom:0.2rem">{label}</div>
            <div class="bubble {css_role}">{safe}</div>
            {ts_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_skill_tags(skills_dict: dict) -> None:
    for category, skills in skills_dict.items():
        st.markdown(f"**{category}**")
        tags = " ".join(f'<span class="tag">{s}</span>' for s in skills)
        st.markdown(tags, unsafe_allow_html=True)


def render_score_chart(scores: list[int], key: str = "score_trend") -> None:
    if len(scores) < 2:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(scores) + 1)),
        y=scores,
        mode="lines+markers",
        line=dict(color="#4f8ef7", width=2.5),
        marker=dict(size=8, color=scores,
                    colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
                    cmin=1, cmax=10),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.1)",
        name="Score"
    ))
    fig.add_hline(y=7, line_dash="dash", line_color="#22c55e",
                  annotation_text="Passing (7)", annotation_position="right")
    fig.update_layout(
        title="Score Progression",
        xaxis_title="Question #",
        yaxis_title="Score (/10)",
        yaxis=dict(range=[0, 10.5]),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_competency_radar(scores: list[int], key: str = "competency_radar") -> None:
    if not scores:
        return
    avg = sum(scores) / len(scores)
    import random; random.seed(int(avg * 10))
    spread = lambda base: max(1, min(10, base + random.uniform(-1.5, 1.5)))
    cats   = ["Technical", "Communication", "Problem Solving", "Depth", "Confidence"]
    vals   = [round(spread(avg), 1) for _ in cats]
    vals  += [vals[0]]
    cats  += [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself",
        fillcolor="rgba(79,142,247,0.2)",
        line=dict(color="#4f8ef7", width=2)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10], showticklabels=True, tickfont=dict(size=9))),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        title="Competency Radar"
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ════════════════════════════════════════════════════════════════════════════
# TAB: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def tab_dashboard(model, vector_store) -> None:
    scores = st.session_state.session_scores

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    avg_score   = round(sum(scores) / len(scores), 1) if scores else 0
    pass_rate   = round(sum(1 for s in scores if s >= 7) / len(scores) * 100) if scores else 0

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{st.session_state.questions_asked}</div>
            <div class="metric-label">Questions Practiced</div></div>""",
            unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_score}</div>
            <div class="metric-label">Average Score /10</div></div>""",
            unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{pass_rate}%</div>
            <div class="metric-label">Pass Rate (≥7)</div></div>""",
            unsafe_allow_html=True)
    with col4:
        skills_count = sum(len(v) for v in
                           extract_skills(st.session_state.resume_text).values())
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{skills_count}</div>
            <div class="metric-label">Skills Detected</div></div>""",
            unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    if scores:
        c1, c2 = st.columns(2)
        with c1:
            render_score_chart(scores, key="dash_score_trend")
        with c2:
            render_competency_radar(scores, key="dash_competency_radar")
    else:
        st.info("🚀 Start an interview session to see your performance analytics here!")

    # Recent feedback
    if st.session_state.feedback_queue:
        st.markdown('<div class="section-header">📋 Recent Feedback</div>', unsafe_allow_html=True)
        for fb in reversed(st.session_state.feedback_queue[-3:]):
            with st.expander(f"Q{fb['q_num']}: {fb['question'][:60]}... — Score: {fb['score']}/10"):
                st.markdown(fb["feedback"])

    # Study roadmap
    if st.button("📚 Generate Personalized Study Roadmap", use_container_width=True):
        with st.spinner("Building your roadmap with IBM Granite..."):
            weak = [f["question"][:30] for f in st.session_state.feedback_queue
                    if f.get("score", 0) < 7]
            ctx  = rag_retrieve(
                f"{st.session_state.job_role} interview preparation roadmap",
                vector_store
            )
            prompt = build_study_roadmap_prompt(
                st.session_state.job_role,
                st.session_state.experience,
                weak,
                st.session_state.resume_skills,
                ctx
            )
            roadmap = call_granite(prompt, model)
            st.markdown(roadmap)


# ════════════════════════════════════════════════════════════════════════════
# TAB: INTERVIEW CHAT
# ════════════════════════════════════════════════════════════════════════════

def tab_interview(model, vector_store) -> None:
    st.markdown('<div class="section-header">💬 Live Interview Session</div>',
                unsafe_allow_html=True)

    # Session controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])
    with ctrl_col1:
        if not st.session_state.interview_active:
            if st.button("▶ Start Interview Session", type="primary", use_container_width=True):
                st.session_state.interview_active    = True
                st.session_state.chat_history        = []
                st.session_state.session_scores      = []
                st.session_state.questions_asked     = 0
                st.session_state.feedback_queue      = []
                st.session_state.demo_asked_indices  = {}  # reset per session
                welcome = (f"Hello {st.session_state.user_name}! 👋 I'm **Alex**, your AI Interview Coach. "
                           f"Today we'll be doing a **{st.session_state.interview_type}** interview "
                           f"for a **{st.session_state.job_role}** role at the "
                           f"**{st.session_state.experience}** level.\n\n"
                           f"I'll ask you questions one at a time. After each answer, I'll provide "
                           f"detailed feedback and a score. Ready? Let's begin! 🚀\n\n"
                           f"*Type your answer below and press Enter, or click 'Next Question'.*")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": welcome,
                    "time": datetime.datetime.now().strftime("%H:%M")
                })
                st.rerun()
        else:
            if st.button("⏹ End Session", type="secondary", use_container_width=True):
                st.session_state.interview_active = False
                if st.session_state.session_scores:
                    avg = sum(st.session_state.session_scores) / len(st.session_state.session_scores)
                    close_msg = (f"Great session, {st.session_state.user_name}! 🎉\n\n"
                                 f"**Summary:**\n"
                                 f"- Questions: {st.session_state.questions_asked}\n"
                                 f"- Average Score: {avg:.1f}/10\n"
                                 f"- Head to the **Dashboard** tab for detailed analytics and your "
                                 f"personalized study roadmap!")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": close_msg,
                        "time": datetime.datetime.now().strftime("%H:%M")
                    })
                st.rerun()

    with ctrl_col2:
        if st.session_state.interview_active:
            if st.button("➡ Next Question", type="primary", use_container_width=True):
                _generate_next_question(model, vector_store)
                st.rerun()

    with ctrl_col3:
        if st.session_state.chat_history:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.chat_history    = []
                st.session_state.interview_active = False
                st.rerun()

    # Chat display
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container" id="chat-bottom">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "system":
                continue
            role    = msg["role"]
            content = msg.get("content", "")
            ts      = msg.get("time", "")
            is_user = role == "user"
            css_r   = "user" if is_user else "ai"
            avatar  = "👤" if is_user else "🤖"
            label   = "You" if is_user else "Alex (AI Coach)"
            safe    = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            ts_html = f'<div class="msg-time">{ts}</div>' if ts else ""
            chat_html += f"""
            <div class="chat-msg {css_r}">
                <div class="avatar {css_r}">{avatar}</div>
                <div>
                    <div style="font-size:0.75rem;color:#888;margin-bottom:0.2rem">{label}</div>
                    <div class="bubble {css_r}">{safe}</div>
                    {ts_html}
                </div>
            </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <p style="font-size:2rem">🎯</p>
            <p>Click <strong>Start Interview Session</strong> to begin your personalized interview practice.</p>
        </div>
        """, unsafe_allow_html=True)

    # Answer input
    if st.session_state.interview_active and st.session_state.current_question:
        st.markdown("---")
        user_answer = st.text_area(
            "✍️ Your Answer",
            key="answer_input",
            height=120,
            placeholder="Type your answer here... Be as detailed as you would in a real interview."
        )
        sub_col1, sub_col2 = st.columns([3, 1])
        with sub_col1:
            if st.button("📤 Submit Answer & Get Feedback", type="primary", use_container_width=True):
                if user_answer.strip():
                    _process_answer(user_answer.strip(), model, vector_store)
                    st.rerun()
                else:
                    st.warning("Please type an answer before submitting.")
        with sub_col2:
            if st.button("💡 Hint", use_container_width=True):
                _give_hint(model, vector_store)
                st.rerun()


def _generate_next_question(model, vector_store) -> None:
    ctx = rag_retrieve(
        f"{st.session_state.interview_type} {st.session_state.job_role} interview question",
        vector_store
    )
    prompt = build_question_prompt(
        st.session_state.job_role,
        st.session_state.experience,
        st.session_state.interview_type,
        st.session_state.difficulty,
        st.session_state.resume_skills,
        ctx,
        st.session_state.questions_asked
    )
    question = call_granite(prompt, model)
    st.session_state.current_question   = question
    st.session_state.questions_asked   += 1
    ts = datetime.datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": f"**Question {st.session_state.questions_asked}:**\n\n{question}",
        "time":    ts
    })


def _process_answer(answer: str, model, vector_store) -> None:
    ts = datetime.datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append({"role": "user", "content": answer, "time": ts})

    ctx      = rag_retrieve(f"{st.session_state.current_question} {answer[:200]}", vector_store)
    prompt   = build_feedback_prompt(
        st.session_state.current_question,
        answer,
        st.session_state.job_role,
        st.session_state.experience,
        ctx
    )
    feedback = call_granite(prompt, model)
    score    = parse_score(feedback)
    if score:
        st.session_state.session_scores.append(score)

    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": feedback,
        "time":    datetime.datetime.now().strftime("%H:%M")
    })
    if score and st.session_state.current_question:
        st.session_state.feedback_queue.append({
            "q_num":    st.session_state.questions_asked,
            "question": st.session_state.current_question,
            "answer":   answer,
            "feedback": feedback,
            "score":    score
        })
    st.session_state.current_question = None


def _give_hint(model, vector_store) -> None:
    ctx    = rag_retrieve(st.session_state.current_question or "", vector_store)
    prompt = (f"<s>[INST] You are Alex, an interview coach. "
              f"Give a brief, helpful hint (2-3 sentences) for answering this question without giving away the answer:\n\n"
              f"{st.session_state.current_question}\n\nContext: {ctx[:400]} [/INST]")
    hint = call_granite(prompt, model)
    ts   = datetime.datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": f"💡 **Hint:** {hint}",
        "time":    ts
    })


# ════════════════════════════════════════════════════════════════════════════
# TAB: RESUME UPLOAD
# ════════════════════════════════════════════════════════════════════════════

def tab_resume(model, vector_store) -> None:
    st.markdown('<div class="section-header">📄 Resume Upload & Skill Extraction</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Upload Your Resume**")
        uploaded = st.file_uploader(
            "Upload PDF or DOCX (max 10 MB)",
            type=["pdf", "docx", "txt"],
            key="resume_uploader"
        )

        if uploaded:
            file_bytes = uploaded.read()
            ext = uploaded.name.split(".")[-1].lower()

            if ext == "pdf":
                text = extract_text_from_pdf(file_bytes)
            elif ext == "docx":
                text = extract_text_from_docx(file_bytes)
            else:
                text = file_bytes.decode("utf-8", errors="ignore")

            if text.strip():
                st.session_state.resume_text = text
                skills_dict  = extract_skills(text)
                flat_skills  = [s for v in skills_dict.values() for s in v]
                st.session_state.resume_skills = flat_skills

                st.success(f"✅ Resume parsed! Found {len(flat_skills)} skills across {len(skills_dict)} categories.")

                # Skill display
                render_skill_tags(skills_dict)

                # AI analysis
                if st.button("🤖 AI Resume Analysis", type="primary"):
                    with st.spinner("Analyzing with IBM Granite..."):
                        ctx    = rag_retrieve(f"{st.session_state.job_role} skills requirements", vector_store)
                        prompt = (f"<s>[INST] Analyze this resume snippet for a {st.session_state.job_role} "
                                  f"({st.session_state.experience}) role. "
                                  f"Identify: 1) Top strengths, 2) Skill gaps, 3) Areas to highlight in interviews, "
                                  f"4) Suggested talking points.\n\nResume excerpt:\n{text[:1500]}\n\n"
                                  f"Job market context:\n{ctx[:500]} [/INST]")
                        analysis = call_granite(prompt, model)
                        st.markdown(analysis)
            else:
                st.error("Could not extract text from this file. Try a different format.")

        st.markdown("---")
        st.markdown("**Or Paste Resume Text**")
        pasted = st.text_area("Paste your resume content here:", height=200, key="resume_paste")
        if st.button("Extract Skills from Text"):
            if pasted.strip():
                st.session_state.resume_text   = pasted
                sd = extract_skills(pasted)
                st.session_state.resume_skills = [s for v in sd.values() for s in v]
                st.success(f"✅ Extracted {len(st.session_state.resume_skills)} skills!")
                render_skill_tags(sd)

    with col2:
        if st.session_state.resume_text:
            st.markdown("**Resume Preview (First 1000 chars)**")
            st.text_area("", value=st.session_state.resume_text[:1000], height=250,
                         disabled=True, key="resume_preview")

            if st.session_state.resume_skills:
                st.markdown("**Skill Coverage Analysis**")
                categories = list(extract_skills(st.session_state.resume_text).keys())
                counts     = [len(extract_skills(st.session_state.resume_text)[c]) for c in categories]
                if counts:
                    fig = px.bar(
                        x=categories, y=counts,
                        title="Skills by Category",
                        color=counts,
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(
                        height=250,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True, key="resume_skills_bar")
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem">
                <p style="font-size:1.5rem">📋</p>
                <p>Upload your resume to get personalized interview questions based on your skills and experience.</p>
                <p style="color:#888;font-size:0.85rem">Supports PDF, DOCX, and plain text</p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB: QUESTION BANK
# ════════════════════════════════════════════════════════════════════════════

def tab_question_bank(model, vector_store) -> None:
    st.markdown('<div class="section-header">📚 Interview Question Generator</div>',
                unsafe_allow_html=True)

    qb_col1, qb_col2 = st.columns([1, 2])

    with qb_col1:
        st.markdown("**Configure Question Set**")
        q_type  = st.selectbox("Interview Type", INTERVIEW_TYPES, key="qb_type")
        q_count = st.slider("Number of Questions", 3, 15, 5, key="qb_count")
        q_focus = st.text_input("Specific Topic (optional)",
                                placeholder="e.g., Python async, System design, STAR stories",
                                key="qb_focus")

        if st.button("🔄 Generate Question Set", type="primary", use_container_width=True):
            with st.spinner(f"Generating {q_count} questions with IBM Granite..."):
                ctx = rag_retrieve(
                    f"{q_type} {st.session_state.job_role} {q_focus} interview questions",
                    vector_store
                )
                prompt = f"""<s>[INST] You are Alex, a Senior Interview Coach.
Generate exactly {q_count} high-quality {q_type} interview questions for a
{st.session_state.experience} {st.session_state.job_role}.
{f'Focus specifically on: {q_focus}' if q_focus else ''}

Relevant context:
{ctx[:800] if ctx else 'Use your expert knowledge.'}

Format each question EXACTLY as:
**Q1:** [question text]
**Q2:** [question text]
...

Include variety: concept checks, scenario-based, and depth-probing questions.
Do NOT include answers. [/INST]"""
                questions = call_granite(prompt, model)
                st.session_state["qb_generated"] = questions

    with qb_col2:
        if "qb_generated" in st.session_state:
            st.markdown("**Generated Questions**")
            st.markdown(st.session_state["qb_generated"])
            st.download_button(
                "⬇ Download Questions",
                data=st.session_state["qb_generated"],
                file_name=f"interview_questions_{st.session_state.job_role.replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # System design question generator
    st.markdown("---")
    st.markdown('<div class="section-header">🏗️ System Design Question Generator</div>',
                unsafe_allow_html=True)

    if st.button("Generate System Design Question", use_container_width=True):
        with st.spinner("Generating system design challenge..."):
            ctx    = rag_retrieve("system design scalability distributed systems", vector_store)
            prompt = build_system_design_prompt(
                st.session_state.job_role,
                st.session_state.experience,
                ctx
            )
            sd_q = call_granite(prompt, model)
            st.markdown(f"""<div class="card">
                <strong>🏗️ System Design Challenge:</strong><br><br>{sd_q}
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB: ANALYTICS
# ════════════════════════════════════════════════════════════════════════════

def tab_analytics() -> None:
    st.markdown('<div class="section-header">📊 Performance Analytics</div>',
                unsafe_allow_html=True)

    scores = st.session_state.session_scores
    fb     = st.session_state.feedback_queue

    if not scores:
        st.info("Complete at least one interview session to view analytics.")
        return

    avg    = sum(scores) / len(scores)
    passed = sum(1 for s in scores if s >= 7)

    # Overview
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Questions", len(scores))
    with m2:
        st.metric("Average Score", f"{avg:.1f}/10")
    with m3:
        st.metric("Pass Rate", f"{passed}/{len(scores)} ({int(passed/len(scores)*100)}%)")
    with m4:
        trend = scores[-1] - scores[0] if len(scores) > 1 else 0
        st.metric("Score Trend", f"{'+' if trend >= 0 else ''}{trend:+.0f}", delta=f"{trend:+.0f}")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        render_score_chart(scores, key="analytics_score_trend")
    with c2:
        render_competency_radar(scores, key="analytics_competency_radar")

    # Score distribution
    if len(scores) >= 3:
        st.markdown("**Score Distribution**")
        bins = {"1-3 (Not Ready)": 0, "4-5 (Below Bar)": 0,
                "6 (Borderline)": 0, "7-8 (Strong)": 0, "9-10 (Exceptional)": 0}
        for s in scores:
            if s <= 3:   bins["1-3 (Not Ready)"]     += 1
            elif s <= 5: bins["4-5 (Below Bar)"]      += 1
            elif s == 6: bins["6 (Borderline)"]        += 1
            elif s <= 8: bins["7-8 (Strong)"]          += 1
            else:        bins["9-10 (Exceptional)"]    += 1
        fig = px.pie(
            names=list(bins.keys()), values=list(bins.values()),
            color_discrete_sequence=["#ef4444","#f59e0b","#facc15","#22c55e","#3b82d4"],
            title="Score Band Distribution"
        )
        fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="score_dist_pie")

    # Question-by-question table
    if fb:
        st.markdown("**Question-by-Question Breakdown**")
        rows = [{
            "Q#": f["q_num"],
            "Question (truncated)": f["question"][:60] + "...",
            "Score": f"{f['score']}/10",
            "Band": ("Exceptional" if f['score'] >= 9 else "Strong" if f['score'] >= 7
                     else "Borderline" if f['score'] >= 5 else "Below Bar")
        } for f in fb]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB: REPORTS
# ════════════════════════════════════════════════════════════════════════════

def tab_reports(model, vector_store) -> None:
    st.markdown('<div class="section-header">📑 Interview Reports</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Generate Performance Report**")
        report_name = st.text_input("Candidate Name", value=st.session_state.user_name)

        if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
            if not FPDF_AVAILABLE:
                st.error("fpdf2 not installed. Run: pip install fpdf2")
            else:
                with st.spinner("Generating PDF report..."):
                    pdf_bytes = generate_pdf_report(
                        user_name=report_name,
                        role=st.session_state.job_role,
                        experience=st.session_state.experience,
                        scores=st.session_state.session_scores,
                        chat_history=st.session_state.chat_history,
                        skills=st.session_state.resume_skills
                    )
                    if pdf_bytes:
                        fname = f"interview_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                        st.download_button(
                            label="⬇ Download PDF Report",
                            data=pdf_bytes,
                            file_name=fname,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        st.success("✅ Report generated!")
                    else:
                        st.error("Failed to generate PDF.")

        st.markdown("---")
        st.markdown("**Generate AI Written Report**")
        if st.button("🤖 AI Performance Report", use_container_width=True):
            with st.spinner("Drafting AI report with IBM Granite..."):
                scores  = st.session_state.session_scores
                avg     = sum(scores)/len(scores) if scores else 0
                weak_qs = [f["question"][:50] for f in st.session_state.feedback_queue
                           if f.get("score", 0) < 7]
                strong  = [f["question"][:50] for f in st.session_state.feedback_queue
                           if f.get("score", 0) >= 8]
                prompt  = f"""<s>[INST] Write a professional interview performance report for:
Candidate: {report_name}
Role: {st.session_state.job_role}
Experience: {st.session_state.experience}
Questions Attempted: {len(scores)}
Average Score: {avg:.1f}/10
Strong areas (high-scoring questions): {'; '.join(strong[:3]) if strong else 'N/A'}
Weak areas (low-scoring questions): {'; '.join(weak_qs[:3]) if weak_qs else 'N/A'}
Skills: {', '.join(st.session_state.resume_skills[:10])}

Write a concise, professional report with: Executive Summary, Strengths, Areas for Improvement, 
Specific Recommendations, and Hiring Readiness Assessment. Be specific and actionable. [/INST]"""
                ai_report = call_granite(prompt, model)
                st.session_state["ai_report"] = ai_report

    with col2:
        if "ai_report" in st.session_state:
            st.markdown("**AI Performance Report**")
            st.markdown(st.session_state["ai_report"])
            st.download_button(
                "⬇ Download Report (Text)",
                data=st.session_state["ai_report"],
                file_name=f"ai_report_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem">
                <p style="font-size:1.5rem">📑</p>
                <p>Generate an AI-written performance report summarizing your interview session.</p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar(model_status: str) -> None:
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Theme toggle
        dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        st.markdown(f'<div class="ibm-badge">{model_status}</div>', unsafe_allow_html=True)
        st.markdown("---")

        # Candidate profile
        st.markdown("**👤 Candidate Profile**")
        st.session_state.user_name  = st.text_input("Your Name", value=st.session_state.user_name, key="sb_name")
        st.session_state.job_role   = st.selectbox("Target Role", ROLES,
                                                    index=ROLES.index(st.session_state.job_role), key="sb_role")
        st.session_state.experience = st.selectbox("Experience Level", EXPERIENCE_LEVELS,
                                                    index=EXPERIENCE_LEVELS.index(st.session_state.experience),
                                                    key="sb_exp")

        st.markdown("---")
        st.markdown("**🎯 Interview Settings**")
        st.session_state.interview_type = st.selectbox("Interview Type", INTERVIEW_TYPES,
                                                        index=INTERVIEW_TYPES.index(st.session_state.interview_type),
                                                        key="sb_itype")

        diff_map = {"entry": 0, "mid": 1, "senior": 2, "expert": 3}
        diff_inv = {0: "entry", 1: "mid", 2: "senior", 3: "expert"}
        diff_labels = ["Entry Level", "Mid Level", "Senior Level", "Expert"]
        diff_idx = diff_map.get(st.session_state.difficulty, 1)
        new_diff  = st.select_slider("Difficulty", options=diff_labels, value=diff_labels[diff_idx], key="sb_diff")
        st.session_state.difficulty = diff_inv[diff_labels.index(new_diff)]

        st.markdown("---")
        st.markdown("**📊 Session Stats**")
        scores = st.session_state.session_scores
        if scores:
            avg = sum(scores) / len(scores)
            st.markdown(f"Questions: **{st.session_state.questions_asked}**")
            st.markdown(f"Avg Score: **{avg:.1f}/10**")
            st.markdown(progress_bar(avg), unsafe_allow_html=True)
        else:
            st.markdown("*No session data yet*")

        st.markdown("---")
        st.markdown("**🔗 Quick Links**")
        st.markdown("[IBM watsonx.ai](https://www.ibm.com/watsonx)")
        st.markdown("[IBM Granite Models](https://www.ibm.com/granite)")
        st.markdown("[LangChain Docs](https://python.langchain.com)")


# ════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    init_session()
    inject_css(st.session_state.dark_mode)

    # Init AI services (cached)
    with st.spinner("Initializing IBM Granite model..."):
        model = get_watsonx_model(
            WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL, GRANITE_MODEL_ID
        )

    model_status = (
        f"✅ IBM Granite ({GRANITE_MODEL_ID.split('/')[-1]})"
        if model else
        "⚠️ Demo Mode (configure .env)"
    )

    with st.spinner("Loading RAG knowledge base..."):
        vector_store = build_vector_store(str(KB_PATH), str(CHROMA_PATH))

    rag_status = "✅ RAG active" if vector_store else "⚠️ RAG unavailable"

    render_sidebar(f"{model_status} | {rag_status}")
    render_header()

    # Navigation tabs
    tabs = st.tabs([
        "🏠 Dashboard",
        "💬 Interview Chat",
        "📄 Resume",
        "📚 Question Bank",
        "📊 Analytics",
        "📑 Reports"
    ])

    with tabs[0]:
        tab_dashboard(model, vector_store)

    with tabs[1]:
        tab_interview(model, vector_store)

    with tabs[2]:
        tab_resume(model, vector_store)

    with tabs[3]:
        tab_question_bank(model, vector_store)

    with tabs[4]:
        tab_analytics()

    with tabs[5]:
        tab_reports(model, vector_store)

    # Footer
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 0.5rem;border-top:1px solid #e5e7eb;
                margin-top:2rem;font-size:0.75rem;color:#888">
        AI Interview Trainer Agent &nbsp;•&nbsp; Powered by IBM watsonx.ai &amp; IBM Granite
        &nbsp;•&nbsp; LangChain + ChromaDB RAG
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
