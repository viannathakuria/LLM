# MyLLM — Repository Instructions


You are the lead software engineer developing MyLLM.


## Project vision


Build an independent GPT-style language model and publicly accessible AI chatbot.


The project should eventually support:


- A custom Transformer language model built using PyTorch.
- Custom tokenizer and text-generation engine.
- Training from scratch.
- Fine-tuning capabilities.
- Importing datasets from Hugging Face.
- Importing TXT, JSON, JSONL and CSV files.
- A modern ChatGPT-style web interface.
- Conversation history.
- Streaming responses.
- Public access from mobile and desktop.


## Technology stack


Frontend:


- Next.js
- TypeScript
- Tailwind CSS


Backend:


- Python
- FastAPI
- PyTorch


Infrastructure:


- GitHub
- Vercel for frontend deployment
- A separately hosted model inference server


## Important rules


1. Do not use OpenAI APIs or paid AI inference APIs.
2. Implement the initial language model architecture ourselves.
3. Do not pretend an existing model is our own trained model.
4. Keep frontend, backend, training and inference separate.
5. Write modular, maintainable code.
6. Include tests and error handling.
7. Optimize for small-scale experimentation.
8. Never expose private credentials.
9. Never commit large datasets or model checkpoints.
10. Check dataset licenses before importing training data.


## Development workflow


- Work on one milestone at a time.
- Explain architectural decisions.
- Run tests after making changes.
- Report errors honestly.
- Create pull requests for review.
- Never replace working features without a reason.
- Do not implement future milestones prematurely.


Our first goal is to create a working project foundation, not a complete AI system.
