# AgenticAI_Assignment1
Agentic AI assignment regarding an student chatbot for policy handbook

```markdown
🎓 AIDTM Intelligent Student Support Agent

An intelligent, multi-turn ReAct (Reason + Act) agent built for the **Adani Institute of Technology Management (AIDTM)**. The system provides automated institutional support by combining targeted Retrieval-Augmented Generation (RAG) over official policy documents, secure student record lookups, persistent user preference memory, and robust error handling.


📂 Project Structure

aidtm-student-support-agent/
│
├── app.py                      # Main Streamlit user interface and application loop
├── tools.py                    # Tool definitions (RAG search, fee status, memory management)
├── prompts.py                  # System prompts and role instructions
├── student_data.json           # Secure local database for individual student records
├── AIDTM Admission Policy.pdf  # Official institutional policy document for RAG retrieval
├── user_profile.json           # Persistent storage for non-sensitive user preferences
├── traces/
│   └── run_logs.jsonl          # Execution audit logs and ReAct turn traces
└── README.md                   # Project documentation

```

---

## ⚙️ Core Architecture & Features

1. **Manual ReAct Loop (`app.py`):** Implements an iterative reasoning and tool-dispatching cycle using Groq LLM inference, ensuring clean tool selection and observation feedback loops.
2. **Targeted Chunk-Based RAG (`tools.py`):** Extracts text from the institutional PDF, chunks paragraphs logically, and scores keyword overlap to pass **only relevant context chunks** rather than dumping the entire document into the LLM context window.
3. **Persistent Non-Sensitive Memory (`user_profile.json`):** Remembers user preferences (e.g., preferred program of study) across browser sessions while strictly blocking sensitive or financial data from being written to disk.
4. **Robust Error Handling & Edge Cases:**
* Gracefully handles missing parameters (e.g., prompting for a missing `student_id`).
* Rejects out-of-domain queries (e.g., weather or general trivia) outside AIDTM's institutional scope.
* Sanitizes message payloads to prevent API validation errors.



---

## 🚀 Setup & Installation Instructions

### 1. Prerequisites

* Python **3.10+** (Python 3.12 recommended)
* A valid **Groq API Key**

### 2. Clone or Setup Workspace

Navigate to your project directory:

```bash
cd aidtm-student-support-agent

```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install streamlit groq pypdf

```

### 4. Configure Environment Variables

Set your Groq API key in your environment terminal:

* **Linux / macOS:**
```bash
export GROQ_API_KEY="your-groq-api-key-here"

```


* **Windows (Command Prompt):**
```cmd
set GROQ_API_KEY="your-groq-api-key-here"

```


* **Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your-groq-api-key-here"

```



---

## ▶️ Execution Steps

To launch the interactive Streamlit user interface, run:

```bash
streamlit run app.py

```

This will automatically open the web application in your default browser at `http://localhost:8501`.

---

## 📊 Tracing & Evaluation

To generate and update your evaluation report:

1. Run the agent application (`app.py`) to execute user interactions and record raw logs.
2. Run `format_trace.py` in your terminal to compile the trace logs into `agent_evaluation_log.xlsx`.

Each time `format_trace.py` is executed, it parses all accumulated traces from `traces/run_logs.jsonl` and appends/updates the structured Excel log with per-turn token counts, latency, and retrieved evidence.

## 🧪 Evaluation & Testing Guide

You can evaluate the agent against the assignment requirements using these test prompts:

* **Persistent Memory:**
* *User:* "I am interested in the MBA Digital Media program. Remember that."
* *(Restart app / refresh session)*
* *User:* "What program did I say I was interested in earlier?" -> *Agent recalls preference from `user_profile.json`.*


* **Targeted RAG Retrieval:**
* *User:* "What is the refund policy for fee cancellation?" -> *Agent invokes `search_student_policy` and retrieves targeted policy chunks.*


* **Missing Information Handling:**
* *User:* "Can you check my fee status?" -> *Agent detects missing `student_id` and asks a clarifying question.*


* **Out-of-Domain Guardrail:**
* *User:* "What is the weather forecast in Tokyo today?" -> *Agent politely declines based on institutional scope.*



```

```

