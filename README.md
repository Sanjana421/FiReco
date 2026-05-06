# 💳 FiReco AI Assistant

FiReco is an AI-powered financial product recommendation platform that delivers personalized credit card recommendations using semantic search, transformer embeddings, explainable AI, and LLM-generated recommendation summaries.

The platform combines modern NLP-based retrieval techniques with scalable backend architecture to provide intelligent financial product matching based on user preferences and spending goals.

---

# 🚀 Features

- Semantic recommendation engine using transformer embeddings
- Personalized financial product matching
- Explainable AI recommendation reasoning
- LLM-generated recommendation summaries
- FastAPI REST backend
- Interactive Streamlit frontend
- SQLite-based product storage
- Modular preprocessing and ranking pipeline
- Real-time recommendation workflow

---

# 🏗️ System Architecture

```text
User Query
    ↓
Semantic Embedding Generation
    ↓
Similarity-Based Retrieval
    ↓
Recommendation Ranking
    ↓
Explainability Layer
    ↓
LLM Summary Generation
    ↓
Frontend Visualization
```

---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Backend | FastAPI, Python |
| Frontend | Streamlit |
| NLP / AI | Sentence Transformers, OpenAI API |
| ML / Ranking | Semantic Similarity, Embedding Search |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| API Communication | Requests |
| Deployment Ready | Uvicorn |

---

# 📂 Project Structure

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

# ⚙️ Recommendation Pipeline

## 1. Data Preprocessing
Financial product data is cleaned, normalized, and enriched with searchable metadata and descriptive tags.

## 2. Semantic Embedding Generation
Sentence Transformer embeddings are generated using:

```python
all-MiniLM-L6-v2
```

This enables semantic similarity search beyond keyword matching.

## 3. Similarity-Based Ranking
User queries are converted into embeddings and matched against product embeddings using cosine similarity.

Additional ranking boosts are applied for:
- preferred categories
- annual fee constraints
- user intent alignment

## 4. Explainability Layer
The system generates transparent recommendation explanations based on:
- reward alignment
- spending category match
- fee compatibility
- semantic relevance

## 5. LLM-Powered Summarization
OpenAI-generated summaries provide concise and personalized recommendation insights for users.

---

# 🔌 API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Get Recommendations

```http
POST /recommend
```

Request:

```json
{
  "goal": "best travel credit card with lounge access",
  "preferred_category": "travel",
  "max_annual_fee": 5000,
  "top_k": 5,
  "use_ai_summary": true
}
```

Response:

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

# 💻 Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd FiReco
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run Data Preprocessing

```bash
python pipeline/preprocess.py
```

---

## 4. Start FastAPI Backend

```bash
uvicorn api.main:app --reload
```

Backend runs on:

```text
http://127.0.0.1:8000
```

---

## 5. Start Streamlit Frontend

```bash
streamlit run app/streamlit_app.py
```

Frontend runs on:

```text
http://127.0.0.1:8501
```

---

# 🧪 Example User Queries

- Best cashback card for online shopping
- Travel credit card with lounge access
- Student-friendly low annual fee card
- Premium business rewards card
- Fuel rewards credit card

---

# 📈 Key Engineering Highlights

- Implemented transformer-based semantic retrieval for financial product recommendations
- Built modular AI pipeline architecture with preprocessing, ranking, explainability, and summarization layers
- Designed scalable REST API using FastAPI
- Developed interactive recommendation frontend using Streamlit
- Integrated LLM-generated recommendation explanations
- Applied semantic similarity search for personalized recommendation workflows

---

# 🔮 Future Improvements

- FAISS vector database integration
- Hybrid recommendation models
- User feedback learning loop
- Real-time personalization
- Cloud deployment
- Recommendation evaluation metrics
- Authentication and user profiles
- Analytics dashboard

---

# 📌 Resume Summary

FiReco demonstrates applied AI engineering skills across:
- NLP and semantic retrieval
- backend API development
- recommendation systems
- explainable AI
- LLM integration
- full-stack AI application architecture
