# 90 Days to Become an AI Engineer

This project is inspired by [90 Days to Become an AI Engineer: The Oct → Dec Roadmap (12 Weeks, 12 Projects)](https://ai.plainenglish.io/90-days-to-become-an-ai-engineer-the-oct-dec-roadmap-12-weeks-12-projects-7aabbca445d6?sk=7d2080cf1e0eed8032940e56ebbb1337).

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
