# LLM_EARNINGS

# 📈 Earnings Call Analyzer

This app lets you semantically search and summarize earnings call transcripts using vector search (Qdrant), LLMs (DeepInfra), and Hugging Face embeddings.

Built with:
- [Streamlit](https://streamlit.io)
- [Qdrant](https://qdrant.tech) for vector similarity search
- [Hugging Face](https://huggingface.co) for embedding generation
- [DeepInfra](https://deepinfra.com) for LLM-powered summarization

---

## 🔍 Features

- 🔎 Semantic search across earnings call transcripts
- 📊 Relevance filtering and speaker-specific query matching
- 💬 LLM-generated summaries grounded in retrieved context
- 🧠 Uses RAG-style architecture (retrieval-augmented generation)

 following TOML (secure format):

```toml
QDRANT_URL = "your-qdrant-url"
QDRANT_API_KEY = "your-qdrant-api-key"
HF_TOKEN = "your-huggingface-api-token"
DEEPINFRA_API_KEY = "your-deepinfra-api-key"
