# AI Hiring Intelligence & Bias Mitigation System

A full-stack, research-grade platform to evaluate AI recruiting models for **unverbalized (hidden) bias**, measure **Explanation Faithfulness Scores (EFS)**, support **Binary, Multiclass, and Regression** hiring decisions, and execute an automated **Mitigation Feedback Loop**.

---

## 🌟 Key Features

### 1. Explanation Faithfulness Score (EFS) Engine
- **4-Quadrant Causal-Verbalization Taxonomy**:
  - **Q1: Unverbalized Bias (Hidden / Deceptive)**: Causal disparity exists, but explanation omitted sensitive factors ($EFS \approx 5-25$).
  - **Q2: Transparent Bias**: Model acknowledged the sensitive factor ($EFS \approx 80-90$).
  - **Q3: Faithful Invariance**: Neutral, qualification-grounded reasoning ($EFS \approx 98-100$).
  - **Q4: Superfluous Mention**: Invariant decision, but mentioned demographic unnecessarily ($EFS \approx 50$).
- **Lexical & Semantic Detector**: Scans LLM explanations with regex boundary matching across `gender`, `religion`, `language`, `ethnicity`, and `age`.

### 2. Flexible Decision Modalities
- **Binary Classification**: `SELECT` vs `REJECT` with McNemar's exact test & 2x2 contingency matrix.
- **Multiclass Categorization**: `STRONG_HIRE`, `HIRE`, `INTERVIEW`, `REJECT` with Stuart-Maxwell & Chi-square transition matrix.
- **Continuous Regression**: Candidate score ($0.0 - 100.0$) with Paired Student's t-test, Cohen's $d$, and 95% Confidence Intervals.

### 3. Modern Full-Stack Web Application
- **Backend**: **FastAPI** REST API providing candidate filtering, clustering, batch evaluation, and mitigation endpoints.
- **Frontend**: **Vanilla HTML5 / CSS3 / JavaScript** single-page web app with responsive layout, SVG gauges, candidate inspector modals, counterfactual playground, and live resume screener.

### 5. First-Class Local Ollama Support (Private & Free)
- Seamless connection to local Ollama runtime (`http://127.0.0.1:11434`).
- Automatic discovery of locally installed models (`qwen3.5:4b`, `llama3`, `mistral`, `gemma2`, `phi3`, `deepseek-r1`).
- Native JSON formatting (`format: "json"`) for robust structured decision extraction.
- Zero external API costs and 100% data privacy.

---

## 📁 Project Architecture

```
C:\Users\Ashok kumar S\Desktop\hiring system\
├── app.py                      # FastAPI Backend Server & REST Endpoints
├── main.py                     # CLI Pipeline Test Suite
├── requirements.txt            # Python Dependencies
├── run_hiring_system.bat       # Windows Launcher (Auto-opens browser)
├── generate_datasets.py        # Dataset Generator
├── README.md                   # System Documentation
├── data/                       # Benchmark Datasets
│   ├── hiring_master.csv       # 150 Master Candidate Profiles
│   ├── hiring_tech.csv         # 60 Software & AI Engineers
│   ├── hiring_leadership.csv   # 40 Management Profiles
│   └── hiring_demo.csv         # 20 Demo Records
├── modules/                    # Research & Evaluation Engines
│   ├── faithfulness.py         # EFS Engine & 4-Quadrant Taxonomy
│   ├── variations.py           # Counterfactual Pair Generator
│   ├── llm_client.py           # Target Model Simulation & Real API Client
│   ├── statistics.py           # McNemar, Chi-Square, Paired t-test
│   ├── mitigation.py           # Prompt-level Debiasing & Feedback Loop
│   └── clustering.py           # TF-IDF + K-Means Candidate Clustering
└── static/                     # Web Application Assets
    ├── index.html              # Single-Page Frontend
    ├── css/style.css           # Modern CSS Design System
    └── js/app.js               # Frontend Controller & Charts
```

---

## 🚀 How to Run

### Option 1: One-Click Windows Launcher (Recommended)
Double-click [`run_hiring_system.bat`](file:///C:/Users/Ashok%20kumar%20S/Desktop/hiring%20system/run_hiring_system.bat).
This starts the FastAPI server and automatically opens `http://127.0.0.1:8000` in your web browser.

### Option 2: Command Line (CLI)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run CLI verification tests
python main.py all

# 3. Launch Web Server manually
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
