# 90 Days to Become an AI Engineer

This project is inspired by [90 Days to Become an AI Engineer: The Oct → Dec Roadmap (12 Weeks, 12 Projects)](https://ai.plainenglish.io/90-days-to-become-an-ai-engineer-the-oct-dec-roadmap-12-weeks-12-projects-7aabbca445d6).

## Week 1 · Oct 1–7 · Python for AI + LLM APIs

- **Learn:** HTTP requests, environment variables, async basics, and Pydantic for data validation. Then read the model provider's quickstart docs end to end.
- **Ship:** A duck.ai-style chatbot that remembers the conversation. Bonus: let the user switch between two different models and compare answers.

### Project 1: Compare Chat

A duck.ai-style chatbot with conversation memory, a two-model switch, and a side-by-side compare mode. It runs against a local LM Studio server or NVIDIA NIM. Memory can be saved to and loaded from JSON.

- [GitHub](https://github.com/jimmy-chihungchong/90-day-ai-engineer/tree/main/week-1/compare-chat)
- [Streamlit deployment](https://compare-chat-dkmkj54ctabolqf6mknxuh.streamlit.app/)

## Week 2 · Oct 8–14 · How LLMs actually work

- **Learn:** Tokens, context windows, temperature and sampling, why models hallucinate, and what embeddings are. You need intuition here, not equations.
- **Ship:** A token and cost visualizer. Paste any text, see how it splits into tokens and what it would cost across a few models.

### Project 2: Token & Cost Visualizer

A Streamlit app for learning how LLMs work without the math. Paste text to see how it splits into tokens and what it would cost across models. You can also explore context windows, temperature and sampling, why models hallucinate, and what embeddings are. It runs against a local LM Studio server or NVIDIA NIM.

- [GitHub](https://github.com/jimmy-chihungchong/90-day-ai-engineer/tree/main/week-2/token-cost-visualizer)
- [Streamlit deployment](https://90-day-ai-engineer-dnzqfoubuzc874mqvnvuoq.streamlit.app/)

## Week 3 · Oct 15–21 · Prompting + structured output

- **Learn:** System prompts, few-shot examples, asking the model to reason step by step, and the big one: getting reliable JSON back using schemas.
- **Ship:** A resume-to-JSON extractor. Upload a messy resume, get clean structured data (name, skills, years of experience). Test it on 10 different resumes and note where it breaks.

### Project 3: Resume to JSON Extractor

Upload a messy PDF or Word (.docx) resume and get clean, structured JSON: name, contact details, skills, work history, education and total years of experience. The app shows the raw extracted text next to the JSON so you can see how scattered data becomes organized data.

- [GitHub](https://github.com/jimmy-chihungchong/90-day-ai-engineer/tree/main/week-3/resume-extractor)
- [Streamlit deployment](https://90-day-ai-engineer-5vgyfhpdg3fx5wains4orf.streamlit.app/)

## Week 4 · Oct 22–31 · Embeddings & semantic search

- **Learn:** How text becomes vectors, cosine similarity, and why "meaning search" beats keyword search.
- **Ship:** Semantic search over your own notes, bookmarks or journal. Ask "what did I write about burnout?" and get the right entries even if you never used that word.

### Project 4: Semantic Notes

Search your own notes, bookmarks or journal by meaning. Ask "what did I write about burnout?" and get entries about exhaustion, overwork or running on fumes, even if the word never appears.

Prototype scope: no auth, no database, no PDF support. Built to show the full workflow end to end.

- [GitHub](https://github.com/jimmy-chihungchong/90-day-ai-engineer/tree/main/week-4/semantic-notes)
- [Streamlit deployment](https://90-day-ai-engineer-mkeqpwfyth7qkn8u6gwmhq.streamlit.app/)

🎃 _**Oct 31 checkpoint:** 4 public repos. You can explain tokens, embeddings and structured output to a friend without notes. If not, repeat the weak week before moving on. The plan bends; it doesn't break._

## ACT II · NOVEMBER · Build things that think

## Week 5 · Nov 1–7 · RAG from scratch

- **Learn:** Retrieval-Augmented Generation: chunking documents, storing embeddings in a vector database, retrieving relevant chunks and feeding them to the model. Learn reranking too.
- **Ship:** Chat with your PDF, with no framework. Build the pipeline yourself first. Use frameworks later, once you know what they're hiding from you.

### Project 5: Chat with your PDF

A local, self-correcting RAG chatbot. Upload PDFs and/or add web page URLs, then ask questions about them. Everything runs on your machine: the chat and embedding models are served by LM Studio, and the UI is a Streamlit app.

- [GitHub](https://github.com/jimmy-chihungchong/90-day-ai-engineer/tree/main/week-5/chat-with-your-pdf)
