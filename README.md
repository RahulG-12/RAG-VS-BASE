# 🔬 LLM Experimental Evaluation Framework  
## RAG vs Base Model Benchmarking

An advanced research-oriented benchmarking system designed to quantitatively evaluate Retrieval-Augmented Generation (RAG) against base Large Language Model (LLM) performance using open-source models.

---

## 🚀 Overview

This project implements a modular experimental framework to compare:

- Base LLM inference
- Retrieval-Augmented Generation (RAG)

The system measures:

- Precision
- Recall
- F1 Score
- Hallucination Rate
- Average Latency

It includes a premium Streamlit dashboard for real-time experimental visualization and metric reporting.

---

## 🧠 System Architecture

The system consists of:

1. RAG Pipeline
   - Document ingestion
   - Text chunking
   - Embedding generation
   - Vector similarity search (FAISS)
   - Context-augmented LLM inference

2. Base LLM Pipeline
   - Direct LLM inference without retrieval

3. Evaluation Engine
   - Automated metric computation
   - Dataset-driven benchmarking
   - Latency tracking
   - Hallucination detection

4. Research Dashboard
   - Metric cards
   - Performance visualization
   - Latency gauge
   - Downloadable experiment results

---

## 🛠️ Tech Stack

- Python
- Ollama (Local LLM Inference)
- phi3 (Open-source model)
- nomic-embed-text (Embedding model)
- FAISS (Vector Similarity Search)
- LangChain
- Streamlit
- Plotly
- scikit-learn
- Docker (for scalable system version)

---

## 📂 Project Structure

rag_research_system/
│
├── app.py
├── run_experiment.py
├── models/
│   └── rag_pipeline.py
├── evaluation/
│   └── metrics.py
├── data/
│   └── dataset.csv
└── README.md

---

## ⚙️ Installation

1. Clone Repository

git clone https://github.com/yourusername/rag-research-system.git
cd rag-research-system

2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate     (Windows)
# source venv/bin/activate  (Mac/Linux)

3. Install Dependencies

pip install streamlit pandas scikit-learn plotly langchain langchain-ollama faiss-cpu

---

## 🤖 Install Required Ollama Models

Make sure Ollama is installed.

ollama pull phi3:mini
ollama pull nomic-embed-text

---

## ▶️ Run Research Dashboard

streamlit run app.py

Open browser and click:
"Run Full Research Experiment"

---

## 📊 Evaluation Metrics

Precision  
Measures proportion of correct positive predictions.

Recall  
Measures how many relevant results were retrieved.

F1 Score  
Harmonic mean of precision and recall.

Hallucination Rate  
Measures responses that do not overlap with expected ground truth.

Latency  
Average inference time per query.

---

## 🧪 Experimental Design

- Dataset-driven evaluation
- Controlled comparison between Base LLM and RAG
- Word-overlap scoring approximation
- Latency benchmarking per query
- Automated result aggregation

---

## 🎯 Key Highlights

- Modular RAG architecture
- Local open-source LLM inference
- Automated benchmarking system
- Quantitative evaluation framework
- Research-grade performance dashboard
- Designed for AI Research and GenAI roles

---

## 🚀 Future Improvements

- Multi-model comparison (Mistral, DeepSeek)
- Statistical significance testing
- Confusion matrix visualization
- Embedding-based semantic similarity scoring
- Large-scale dataset benchmarking

---

## 👨‍💻 Author

Rahul Giri  
AI Developer | Generative AI | LLM Systems  

---

## 📄 License

For educational and research demonstration purposes.
