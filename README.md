Task 5: Auto Tagging Support Tickets Using LLM 📩🤖

This project demonstrates how to automatically tag support tickets into categories using Large Language Models (LLMs).
The goal is to apply Zero-Shot, Few-Shot, and Fine-Tuning techniques to classify tickets into relevant categories and output the top 3 most probable tags per ticket.

🎯 Objective

Automatically tag support tickets into categories using an LLM.

📂 Dataset

Free-text Support Ticket Dataset (contains customer support queries written in natural language).

📝 Instructions Followed

✅ Used prompt engineering and fine-tuning with an LLM

✅ Compared Zero-Shot vs Fine-Tuned performance

✅ Applied Few-Shot learning techniques to improve accuracy

✅ Output the Top 3 most probable tags per ticket

🚀 Features

Zero-Shot Classification (no training data required)

Few-Shot Learning (provide small labeled examples)

Fine-Tuning (train LLM on custom dataset for better accuracy)

Multi-class prediction and ranking (returns top 3 categories per ticket)

⚙️ Installation
1. Clone this repository
git clone https://github.com/your-username/support-ticket-tagging-llm.git
cd support-ticket-tagging-llm

2. Create & activate virtual environment
python -m venv env


Windows (cmd):

env\Scripts\activate


Linux / Mac:

source env/bin/activate

3. Install dependencies
pip install -r requirements.txt

▶️ Usage

Run the main script:

python main.py

Example Input
"My internet connection keeps dropping."

Example Output
Predicted Categories:
1. Network Issue
2. Technical Support
3. Account Problem

🛠 Requirements

Python 3.8+

transformers

torch

huggingface-hub

Install with:

pip install torch transformers huggingface_hub

🎓 Skills Gained

Prompt engineering

LLM-based text classification

Zero-Shot & Few-Shot Learning

Multi-class prediction and ranking

📌 Future Improvements

Train with a larger support ticket dataset
Deploy as an API for real-time classification
Add Gradio / Streamlit UI for better usability
