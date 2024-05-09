# Car Sales Chatbot

This is a Streamlit application that uses LangChain to create a conversational chatbot for assisting users in selecting a car based on their preferences and collecting their personal information for purchase.

## Features

- Multi-agent architecture with two agents: "Preferer" and "Data_collector"
- "Preferer" agent guides users through the car selection process and provides tailored recommendations
- "Data_collector" agent collects user's personal details (name, email) and finalizes the car purchase
- RAG using Chroma vector store

## Setup

1. Install the required dependencies: pip install -r requirements.txt
2. Set up the OpenAI API key by adding it in .env file
3. Run the Streamlit app: streamlit run multi_agent.py

