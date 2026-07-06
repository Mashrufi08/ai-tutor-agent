"""
AI Personalized Tutor Agent — SDG 4: Quality Education
--------------------------------------------------------
This app demonstrates an AI AGENT (not just a chatbot):
  PERCEIVE -> quiz answers from the student
  REASON   -> LLM analyzes wrong answers, finds weak topics
  ACT      -> generates a tailored explanation + new practice questions
  REMEMBER -> stores every attempt in progress.json and shows a trend

Run with:  streamlit run app.py
Requires:  pip install streamlit anthropic pandas
Set your API key as an environment variable before running:
  export ANTHROPIC_API_KEY="your-key-here"      (Mac/Linux)
  setx ANTHROPIC_API_KEY "your-key-here"         (Windows)
"""

import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
from anthropic import Anthropic

# ----------------------------
# CONFIG
# ----------------------------
PROGRESS_FILE = "progress.json"
MODEL = "claude-sonnet-4-5"

QUIZ = [
    {"q": "What is the capital of Australia?", "options": ["Sydney", "Canberra", "Melbourne", "Perth"], "answer": "Canberra", "topic": "Geography"},
    {"q": "Who wrote the play 'Romeo and Juliet'?", "options": ["Charles Dickens", "William Shakespeare", "Mark Twain", "Jane Austen"], "answer": "William Shakespeare", "topic": "Literature"},
    {"q": "Which planet is known as the Red Planet?", "options": ["Venus", "Jupiter", "Mars", "Saturn"], "answer": "Mars", "topic": "Science"},
    {"q": "In which year did World War II end?", "options": ["1943", "1945", "1947", "1950"], "answer": "1945", "topic": "History"},
    {"q": "What is the largest ocean on Earth?", "options": ["Atlantic", "Indian", "Arctic", "Pacific"], "answer": "Pacific", "topic": "Geography"},
    {"q": "What gas do plants absorb from the atmosphere for photosynthesis?", "options": ["Oxygen", "Nitrogen", "Carbon Dioxide", "Hydrogen"], "answer": "Carbon Dioxide", "topic": "Science"},
    {"q": "Who was the first Prime Minister of India?", "options": ["Mahatma Gandhi", "Jawaharlal Nehru", "Sardar Patel", "Indira Gandhi"], "answer": "Jawaharlal Nehru", "topic": "History"},
    {"q": "Which is the smallest country in the world by area?", "options": ["Monaco", "Vatican City", "San Marino", "Liechtenstein"], "answer": "Vatican City", "topic": "Geography"},
]


# ----------------------------
# MEMORY (simple file-based storage — the agent's "memory")
# ----------------------------
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return []


def save_attempt(score, total, weak_topics):
    history = load_progress()
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "score": score,
        "total": total,
        "weak_topics": weak_topics
    })
    with open(PROGRESS_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ----------------------------
# AGENT REASONING + ACTION (calls the LLM)
# ----------------------------
def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("No ANTHROPIC_API_KEY found. Set it as an environment variable and restart the app.")
        st.stop()
    return Anthropic(api_key=api_key)


def generate_feedback(wrong_items):
    """
    wrong_items: list of dicts with q, chosen, correct, topic
    Returns the agent's generated explanation + new practice questions.
    """
    client = get_client()

    topics = ", ".join(sorted(set(item["topic"] for item in wrong_items)))
    details = "\n".join(
        f"- Question: {item['q']} | Student answered: {item['chosen']} | Correct answer: {item['correct']} | Topic: {item['topic']}"
        for item in wrong_items
    )

    prompt = f"""You are an AI tutor agent helping a student improve.
The student got these questions wrong:
{details}

Weak topics detected: {topics}

Do the following:
1. Briefly explain WHY each correct answer is right, in simple, encouraging language (2-3 sentences per question).
2. Generate 3 NEW practice questions (multiple choice, 4 options each) covering the weak topics, to help the student practice. Do not repeat the same questions.
3. Keep the tone warm and motivating, like a supportive tutor.

Format your response in clear markdown with headers "### Explanations" and "### Practice Questions".
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="AI Tutor Agent — SDG 4", page_icon="🎓")
st.title("🎓 AI Personalized Tutor Agent")
st.caption("SDG 4: Quality Education — an AI agent that perceives, reasons, acts, and remembers.")

st.markdown("Answer the general knowledge quiz below. The agent will detect your weak topics and generate personalized help.")

with st.form("quiz_form"):
    responses = []
    for i, item in enumerate(QUIZ):
        choice = st.radio(f"**Q{i+1}. {item['q']}**", item["options"], index=None, key=f"q{i}")
        responses.append(choice)
    submitted = st.form_submit_button("Submit Quiz")

if submitted:
    if any(r is None for r in responses):
        st.warning("Please answer all questions before submitting.")
    else:
        wrong_items = []
        score = 0
        for item, chosen in zip(QUIZ, responses):
            if chosen == item["answer"]:
                score += 1
            else:
                wrong_items.append({"q": item["q"], "chosen": chosen, "correct": item["answer"], "topic": item["topic"]})

        total = len(QUIZ)
        st.subheader(f"Score: {score}/{total}")

        weak_topics = sorted(set(item["topic"] for item in wrong_items))
        save_attempt(score, total, weak_topics)

        if wrong_items:
            st.info(f"Weak topics detected: {', '.join(weak_topics)}")
            with st.spinner("🤖 Agent is analyzing your answers and generating personalized help..."):
                feedback = generate_feedback(wrong_items)
            st.markdown(feedback)
        else:
            st.success("Perfect score! No weak topics detected. 🎉")

# ----------------------------
# PROGRESS / MEMORY VIEW
# ----------------------------
st.divider()
st.subheader("📈 Your Progress Over Time")
history = load_progress()
if history:
    df = pd.DataFrame(history)
    df["percentage"] = (df["score"] / df["total"]) * 100
    st.line_chart(df.set_index("timestamp")["percentage"])
    st.dataframe(df[["timestamp", "score", "total", "weak_topics"]], use_container_width=True)
else:
    st.write("No attempts yet. Complete the quiz above to start tracking progress.")
