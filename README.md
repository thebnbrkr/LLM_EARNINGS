#  LLM_EARNINGS – Earnings Call Analyzer (APPLE Q2 2025)

This Streamlit app allows you to **semantically search and summarize earnings call transcripts** using a combination of vector databases (Qdrant), transformer embeddings (Hugging Face), and LLMs (DeepInfra-hosted LLaMA). It follows a **retrieval-augmented generation (RAG)** pattern to make AI-generated answers transparent, explainable, and grounded in actual transcript content.

---

##  What Problem Does This Solve?

Most earnings call summaries and transcripts are long, hard to search, and offer little context. This app helps you:

- Ask **natural language questions** like "What did Tim Cook say about AI?"
- Get answers **grounded in evidence**, not hallucinated
- Understand **exactly which speaker said what**
- Filter results by **speaker relevance** and **semantic similarity**

---

## 🔍 How It Works — Under the Hood

```
User Question
    ↓
[1] Embedding Generation
    ↓
[2] Vector Search in Qdrant
    ↓
[3] Filter + Format Relevant Chunks
    ↓
[4] Summarization with LLM (DeepInfra)
    ↓
Final Answer + Quotes Shown to User
```

1. 🔢 **Embedding the Query**  
   The app uses BAAI/bge-small-en to transform your natural language query into a dense vector embedding. This captures semantic meaning, not just exact words.

   Example:
   
   "How did Apple perform in China?"  
   might semantically match "Tim Cook discussed weakening demand in Asia".

2. 📦 **Semantic Retrieval with Qdrant**  
   The app then queries Qdrant, a high-performance vector database, to retrieve the top-k transcript chunks most similar to your question. Qdrant performs cosine similarity search on pre-embedded chunks from earnings call transcripts.
   
   You can optionally filter by speaker (e.g., Tim Cook, Lisa Su), and set a minimum similarity threshold and result count.

3. 🧩 **Compiling Context**  
   The top-matching transcript snippets are:
   
   - Grouped by speaker
   - Ranked by semantic similarity
   - Compiled into a structured input to guide the LLM
   
   These are then formatted like:
   
   ```
   [Lisa Su]: We're seeing strong enterprise demand across AI workloads.
   [Tim Cook]: In China, we saw a 12% revenue decline, partially offset by services growth.
   ```

4. 📝 **LLM Summarization via DeepInfra**  
   The structured context + original question is passed to a hosted LLaMA model (meta-llama/Llama-4-Maverick-17B) via the DeepInfra OpenAI-compatible API.
   
   The prompt is designed to:
   
   - Stay within retrieved context
   - Avoid speculation
   - Clearly explain what was said and by whom
   
   The result: an interpretable, well-grounded summary.

## 💡 Example Queries

- "What did Tim Cook say about AI?"
- "How did Microsoft perform in the cloud segment?"
- "Was inflation mentioned in Q4?"

## 🛠️ Stack Overview

| Layer | Tool |
|-------|------|
| UI + Controls | Streamlit |
| Vector DB | Qdrant |
| Embeddings | BAAI/bge-small-en |
| LLM (summarization) | DeepInfra hosting LLaMA 4 |
| Vector Format | JSON chunks (title, speaker, text, timestamp) |

## 🗃️ File Structure

```
├── app.py                      # Main Streamlit app logic
├── requirements.txt            # Python dependencies
└── .streamlit/
    └── secrets.toml            # API keys (kept private)
```

## 🧪 Live Demo

👉 Try it here (Streamlit) : https://llm-earnings.streamlit.app/



## 🧠 Why This Matters

Unlike traditional search, this app understands intent using embeddings and shows only the most relevant parts of the transcript — no more skimming through 30-page PDFs.

Because you see both:
- the model's answer, and
- the retrieved evidence

…you can trust the output and trace back what was said.

