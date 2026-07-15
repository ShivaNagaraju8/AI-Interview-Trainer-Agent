# 🎯 AI Interview Trainer Agent

An AI-powered, full-featured interview preparation platform built with **IBM watsonx.ai**, **IBM Granite**, **LangChain**, **ChromaDB**, and **Streamlit**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Live Interview Chat** | Conversational AI interviewer powered by IBM Granite with STAR-aware follow-ups |
| 📄 **Resume Upload & Skill Extraction** | PDF/DOCX parsing with automatic skill categorization |
| 🏠 **Personalized Dashboard** | Metrics, charts, score trends, and AI-generated study roadmaps |
| 📚 **Question Bank Generator** | Bulk generate Technical, HR, Behavioral, or System Design questions |
| 📊 **Performance Analytics** | Score progression, competency radar, distribution charts |
| 📑 **Downloadable Reports** | AI-written performance reports + PDF export |
| 🔍 **RAG Knowledge Base** | ChromaDB-backed retrieval of interview guides, company experiences, best practices |
| 🌙 **Dark / Light Mode** | Full dark and light theme toggle |
| 🎛️ **AGENT_INSTRUCTIONS** | Customize AI personality, difficulty, scoring, and feedback style without touching core logic |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- IBM Cloud account ([register free](https://cloud.ibm.com/registration))
- IBM watsonx.ai project ([create here](https://dataplatform.cloud.ibm.com/projects/))

### 2. Clone & Setup

```bash
git clone <repo-url>
cd ai-interview-trainer

# Automated setup
python setup.py

# Or manually:
pip install -r requirements.txt
cp .env.example .env
```

### 3. Configure IBM watsonx.ai

Edit `.env`:

```env
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-13b-chat-v2
```

**Get your credentials:**
- API Key: [IBM Cloud → IAM → API Keys](https://cloud.ibm.com/iam/apikeys)
- Project ID: [watsonx.ai Projects](https://dataplatform.cloud.ibm.com/projects/)

### 4. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) 🎉

---

## 🏗️ Project Structure

```
ai-interview-trainer/
├── app.py                          # Main Streamlit application (AGENT_INSTRUCTIONS at top)
├── setup.py                        # Quick-start setup script
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore
├── knowledge_base/
│   ├── technical/
│   │   └── software_engineering.txt   # DSA, system design, OOP, Python
│   ├── hr/
│   │   └── hr_questions.txt           # HR questions, salary negotiation
│   ├── behavioral/
│   │   └── behavioral_questions.txt   # STAR method, competency frameworks
│   └── company/
│       └── company_experiences.txt    # FAANG patterns, salary benchmarks
└── reports/                        # Generated PDF/text reports
```

---

## 🎛️ Customizing the Agent (AGENT_INSTRUCTIONS)

At the top of [`app.py`](app.py), the `AGENT_INSTRUCTIONS` docstring lets you tune:

```python
AGENT_INSTRUCTIONS:
  INTERVIEWER_PERSONA:   name, tone, communication style
  DIFFICULTY_LEVELS:     entry / mid / senior / expert definitions
  JOB_ROLE_SPECIALIZATIONS: per-role focus areas
  EVALUATION_CRITERIA:  weights for technical/communication/problem-solving
  FEEDBACK_STYLE:       sandwich format, specificity, length
  SCORING_METHODOLOGY:  scale, passing score, band labels
  SAFETY_POLICIES:      content guardrails
  COACHING_BEHAVIOR:    hints, encouragement, pacing, follow-ups
```

No changes to core logic required.

---

## 🤖 IBM Granite Model

This app uses the **IBM Granite 13B Chat** model via watsonx.ai:

- Model ID: `ibm/granite-13b-chat-v2`
- All prompts follow IBM watsonx.ai best practices
- Uses `[INST]...[/INST]` instruction format
- Parameters: temperature 0.7, top_p 0.9, max_tokens 1024

**Available IBM Granite Models:**
- `ibm/granite-13b-chat-v2` (default — balanced)
- `ibm/granite-13b-instruct-v2` (instruction-following)
- `ibm/granite-3-8b-instruct` (efficient)

---

## 🔍 RAG Architecture

```
User Query
    ↓
HuggingFace MiniLM Embeddings (sentence-transformers/all-MiniLM-L6-v2)
    ↓
ChromaDB Similarity Search (top-4 chunks)
    ↓
Context injected into IBM Granite prompt
    ↓
AI Response with grounded knowledge
```

Knowledge base covers: DSA, system design, HR best practices, STAR behavioral framework, FAANG interview patterns, salary benchmarks.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `ibm-watsonx-ai` | IBM Granite model inference |
| `langchain` + `langchain-community` | RAG orchestration |
| `chromadb` | Vector store |
| `sentence-transformers` | Text embeddings (local) |
| `PyPDF2` | PDF resume parsing |
| `python-docx` | DOCX resume parsing |
| `fpdf2` | PDF report generation |
| `plotly` | Interactive charts |
| `pandas` | Data manipulation |

---

## 🔒 Security

- IBM API key stored in `.env` (never in code)
- `.env` excluded from version control via `.gitignore`
- No user data persisted beyond session state
- Content safety policies enforced via AGENT_INSTRUCTIONS

---

## 🆓 IBM Cloud Lite (Free Tier)

This app is designed for IBM Cloud Lite:
- watsonx.ai Lite: 20,000 free tokens/month
- No credit card required for Lite tier
- Scale up seamlessly when ready

---

## 📜 License

MIT License — See [LICENSE](LICENSE)

---

<div align="center">
Built with ❤️ using IBM watsonx.ai · IBM Granite · LangChain · Streamlit
</div>
