import streamlit as st
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd

# ✅ Secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
DEEPINFRA_API_KEY = st.secrets["DEEPINFRA_API_KEY"]

# ✅ Config
COLLECTION_NAME = "earnings_call_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en"
DEEPINFRA_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

class EarningsCallAnalyzer:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.llm_client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")

    def extract_speaker_from_query(self, query: str) -> Optional[str]:
        execs = ["Tim Cook", "Satya Nadella", "Jensen Huang", "Lisa Su", 
                 "Elon Musk", "Sundar Pichai", "Mark Zuckerberg"]
        for name in execs:
            if name.lower() in query.lower():
                return name
        return None

    def search_earnings_calls(self, query: str, limit: int = 10, min_score: float = 0.5) -> List[Dict[str, Any]]:
        speaker_filter = self.extract_speaker_from_query(query)
        filter_conditions = []
        if speaker_filter:
            filter_conditions.append(models.FieldCondition(key="speaker", match=models.MatchValue(value=speaker_filter)))
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        query_vector = self.embedder.encode(query).tolist()

        results = self.qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter
        )

        return [
            {
                "score": r.score,
                "speaker": r.payload.get("speaker", "Unknown"),
                "text": r.payload.get("text", "")
            }
            for r in results if r.score >= min_score
        ]

    def generate_summary(self, query: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant results found."

        context = "\n\n".join([f"[{r['speaker']}]: {r['text']}" for r in results[:3]])

        try:
            chat_completion = self.llm_client.chat.completions.create(
                model=DEEPINFRA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant that analyzes earnings call transcripts and explains them to users. "
                                   "If the user greets you or asks your purpose, introduce yourself as a financial AI assistant."
                    },
                    {
                        "role": "user",
                        "content": f"My question: {query}\n\nRelevant statements:\n{context}"
                    }
                ]
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"

# ✅ Streamlit UI
st.set_page_config(page_title="Earnings Call Analyzer", layout="wide")
st.title("📈 Earnings Call Analyzer (Apple Q2 2025)")
st.markdown("Ask questions about earnings call transcripts. The AI will return relevant quotes and a summarized answer.")

example_queries = [
    "What did Tim Cook say about AI?",
    "How did Apple perform in China?",
    "What were the financial highlights?",
    "Did they discuss inflation?",
]

with st.sidebar:
    st.header("Settings")
    min_score = st.slider("Minimum Relevance Score", 0.5, 0.95, 0.7, step=0.05)
    max_results = st.slider("Max Results", 3, 20, 10)
    example = st.selectbox("Examples", [""] + example_queries)

query_input = st.text_area("Your question:", example if example else "", height=100)
submit = st.button("🔍 Search")

if submit and query_input.strip():
    analyzer = EarningsCallAnalyzer()
    with st.spinner("Running semantic search and summarizing with LLM..."):
        results = analyzer.search_earnings_calls(query_input, max_results, min_score)
        summary = analyzer.generate_summary(query_input, results)

    st.subheader("💡 Summary")
    st.markdown(summary)

    st.subheader(f"📄 Relevant Statements ({len(results)})")
    if results:
        df = pd.DataFrame([
            {
                "Relevance": f"{r['score']:.3f}",
                "Speaker": r["speaker"],
                "Statement": r["text"][:500] + "..." if len(r["text"]) > 500 else r["text"]
            } for r in results
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No statements found.")

