# FiReco — AI-Powered Financial Product Recommendation Platform

FiReco is an AI-powered recommendation platform that delivers personalized credit card recommendations using semantic search, transformer embeddings, explainable AI, and LLM-generated recommendation summaries.

The platform is designed to simulate a production-style financial recommendation workflow by combining modern NLP retrieval techniques with scalable backend architecture and interactive user interfaces.

---

## Problem Statement

Traditional recommendation systems often rely on keyword matching and static filtering rules, limiting their ability to understand user intent expressed in natural language.

FiReco addresses this challenge by implementing:
- semantic similarity retrieval,
- embedding-based recommendation ranking,
- explainable recommendation reasoning,
- and AI-generated recommendation summaries.

The system enables users to describe financial goals naturally while receiving personalized and context-aware financial product recommendations.

---

# Features

- Transformer-based semantic recommendation engine
- Personalized credit card recommendations
- Explainable AI recommendation reasoning
- LLM-generated recommendation summaries
- FastAPI backend API
- Interactive Streamlit frontend
- SQLite-based product storage
- Modular preprocessing and ranking pipeline
- Real-time recommendation workflow

---

# System Architecture

```text
User Query
    ↓
Semantic Embedding Generation
    ↓
Embedding Similarity Retrieval
    ↓
Recommendation Ranking
    ↓
Explainability Layer
    ↓
LLM Recommendation Summarization
    ↓
Frontend Visualization
```

---

# Technology Stack

| Layer | Technologies |
|---|---|
| Backend API | FastAPI, Python |
| Frontend | Streamlit |
| NLP / AI | Sentence Transformers, OpenAI API |
| Recommendation Engine | Semantic Similarity Retrieval |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| API Communication | Requests |
| Deployment | Uvicorn |

---

# Project Structure

```text
FiReco/
│
├── ai/
│   └── assistant.py
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── pipeline/
│   └── preprocess.py
│
├── recommender/
│   ├── ranker.py
│   └── explain.py
│
├── database/
│   └── finance.db
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
└── README.md
```

---

# Recommendation Workflow

## 1. Data Preprocessing
Financial product data is cleaned, normalized, tagged, and transformed into searchable semantic representations.

## 2. Semantic Embedding Generation
Product descriptions are converted into dense vector embeddings using:

```python
all-MiniLM-L6-v2
```

This enables semantic search beyond keyword matching.

## 3. Similarity-Based Retrieval
User queries are embedded and matched against financial product embeddings using cosine similarity.

Additional ranking boosts are applied based on:
- financial usage category,
- annual fee preferences,
- and semantic relevance.

## 4. Explainability Layer
The explainability module generates recommendation reasoning based on:
- reward alignment,
- category relevance,
- fee compatibility,
- and semantic matching.

## 5. LLM-Based Recommendation Summarization
OpenAI-powered summaries generate concise and personalized recommendation insights for end users.

---

# API Endpoints

## Health Check

```http
GET /health
```

### Response

```json
{
  "status": "ok"
}
```

---

## Recommendation Endpoint

```http
POST /recommend
```

### Request Example

```json
{
  "goal": "best travel credit card with lounge access",
  "preferred_category": "travel",
  "max_annual_fee": 5000,
  "top_k": 5,
  "use_ai_summary": true
}
```

### Response Example

```json
{
  "recommendations": [
    {
      "rank": 1,
      "product_name": "Example Card",
      "bank_name": "Example Bank",
      "score": 0.91,
      "reason": "strong travel rewards and benefits"
    }
  ],
  "ai_summary": "Recommended based on travel-focused rewards and lounge access."
}
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/<username>/FiReco.git
cd FiReco
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Data Preprocessing

```bash
python pipeline/preprocess.py
```

---

## Start FastAPI Backend

```bash
uvicorn api.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

---

## Launch Streamlit Frontend

```bash
streamlit run app/streamlit_app.py
```

Frontend URL:

```text
http://127.0.0.1:8501
```

---

# Example Queries

- Best cashback card for online shopping
- Travel credit card with lounge access
- Student-friendly low annual fee card
- Premium business rewards card
- Fuel rewards credit card

---

# Future Improvements

- FAISS vector database integration
- Hybrid recommendation architectures
- Recommendation evaluation metrics
- Real-time personalization
- User feedback learning loop
- Cloud deployment
- Authentication and analytics dashboard
