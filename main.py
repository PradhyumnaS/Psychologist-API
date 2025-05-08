import os
import pickle
import base64
import json
import redis
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import google.generativeai as genai
from reinforcement import PromptOptimizationRL
from dotenv import load_dotenv

# --- CONFIG ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash"
KB_PATH = "./kb.csv"
Q_TABLE_DIR = "./q_tables"
os.makedirs(Q_TABLE_DIR, exist_ok=True)
load_dotenv()

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
redis_conn = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)
MAX_HISTORY_LENGTH = 10

# --- FASTAPI APP ---
app = FastAPI(title="NeuroSphere Therapist API")

# --- MODELS ---
class ChatRequest(BaseModel):
    user_id: str
    message: str
    gender: Optional[str] = None
    age: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    audio_base64: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_id: str
    feedback: str

class FeedbackResponse(BaseModel):
    status: str

# --- KNOWLEDGE BASE ---
def load_knowledge_base():
    df = pd.read_csv(KB_PATH)
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise Exception("Knowledge base CSV missing required columns.")
    vectorizer = TfidfVectorizer(stop_words='english')
    question_vectors = vectorizer.fit_transform(df['question'])
    print("Knowledge base loaded and vectorized.")
    return {
        'df': df,
        'vectorizer': vectorizer,
        'question_vectors': question_vectors
    }

knowledge_base = load_knowledge_base()

def get_relevant_knowledge(user_message, knowledge_base, top_n=3):
    user_vector = knowledge_base['vectorizer'].transform([user_message])
    similarities = cosine_similarity(user_vector, knowledge_base['question_vectors']).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    relevant_entries = []
    for idx in top_indices:
        similarity = similarities[idx]
        if similarity > 0.2:
            relevant_entries.append({
                'question': knowledge_base['df']['question'].iloc[idx],
                'answer': knowledge_base['df']['answer'].iloc[idx],
                'similarity': similarity
            })
    print(f"Found {len(relevant_entries)} relevant entries.")
    return relevant_entries

def summarize_knowledge_entries(kb_entries):
    if not kb_entries:
        return ""
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    content_to_summarize = "Summarize these expert answers concisely within 100 words:\n\n"
    for entry in kb_entries:
        content_to_summarize += f"EXPERT ANSWER: {entry['answer']}\n\n"
    content_to_summarize += "Format as bullet points focusing on key advice and therapeutic approaches."
    response = model.generate_content(content_to_summarize)
    print(f"Summarized - {response.text}")
    return response.text

# --- RL AGENT PER USER ---
def get_q_table_path(user_id):
    return os.path.join(Q_TABLE_DIR, f"{user_id}_qtable.pkl")

def load_rl_agent(user_id):
    q_table_path = get_q_table_path(user_id)
    agent = PromptOptimizationRL()
    if os.path.exists(q_table_path) and os.path.getsize(q_table_path) > 0:
        with open(q_table_path, "rb") as f:
            agent.q_table = pickle.load(f)
    return agent

def save_rl_agent(user_id, agent):
    q_table_path = get_q_table_path(user_id)
    with open(q_table_path, "wb") as f:
        pickle.dump(agent.q_table, f)

# --- AUDIO ---
def text_to_base64_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_path = "response.mp3"
    tts.save(audio_path)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

# --- REDIS CONTEXT MANAGEMENT ---
def get_history_key(user_id):
    return f"chat_history:{user_id}"

def get_conversation_history(user_id):
    try:
        history_json = redis_conn.lrange(get_history_key(user_id), -MAX_HISTORY_LENGTH * 2, -1)
        history = [json.loads(msg) for msg in history_json]
        return history
    except Exception:
        return []

def add_to_conversation_history(user_id, role, content):
    try:
        redis_conn.rpush(get_history_key(user_id), json.dumps({"role": role, "content": content}))
        redis_conn.ltrim(get_history_key(user_id), -MAX_HISTORY_LENGTH * 2, -1)
    except Exception:
        pass

def get_cached_response(user_id, message):
    cache_key = f"response_cache:{user_id}:{hash(message)}"
    return redis_conn.get(cache_key)

def set_cached_response(user_id, message, response):
    cache_key = f"response_cache:{user_id}:{hash(message)}"
    redis_conn.set(cache_key, response, ex=3600)  

# --- ROOT ENDPOINT ---
@app.get("/")
def root():
    return {"Status": "NeuroSphere Therapist API is running."}

# --- CHAT ENDPOINT ---
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        # Check cache
        cached = get_cached_response(req.user_id, req.message)
        if cached:
            cached_data = json.loads(cached)
            return ChatResponse(**cached_data)

        # Load RL agent for user
        agent = load_rl_agent(req.user_id)
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        # Conversation context from Redis
        history = get_conversation_history(req.user_id)
        context_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        # Knowledge base context
        kb_entries = get_relevant_knowledge(req.message, knowledge_base)
        knowledge_context = ""
        if kb_entries:
            knowledge_summary = summarize_knowledge_entries(kb_entries)
            knowledge_context = "\n\nRELEVANT EXPERT INSIGHTS:\n" + knowledge_summary

        # User context
        user_context = ""
        if req.gender or req.age:
            user_context = "USER CONTEXT:\n"
            if req.gender:
                user_context += f"- Gender: {req.gender}\n"
            if req.age:
                user_context += f"- Age: {req.age}\n"

        # Compose prompt
        full_user_message = (
            f"{context_prompt}\n\n"
            f"{user_context}\n"
            f"{knowledge_context}\n\n"
            f"user: {req.message}\n\n"
        )

        print(f"Full user message: {full_user_message}")

        # RL prompt optimization
        optimized_prompt, action = agent.generate_optimized_prompt(full_user_message)
        agent.last_action = action

        # Get response from Gemini
        response = model.generate_content(optimized_prompt)
        response_text = response.text

        # Audio
        audio_base64 = text_to_base64_audio(response_text)

        redis_conn.set(f"rl_last_state:{req.user_id}", agent.last_state)
        redis_conn.set(f"rl_last_action:{req.user_id}", agent.last_action)

        # Save updated Q-table
        save_rl_agent(req.user_id, agent)

        # Save to Redis context
        add_to_conversation_history(req.user_id, "user", req.message)
        add_to_conversation_history(req.user_id, "assistant", response_text)

        # Cache response
        resp_obj = {"response": response_text, "audio_base64": audio_base64}
        set_cached_response(req.user_id, req.message, json.dumps(resp_obj))

        return ChatResponse(**resp_obj)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- FEEDBACK ENDPOINT ---
@app.post("/feedback", response_model=FeedbackResponse)
def feedback_endpoint(req: FeedbackRequest):
    try:
        agent = load_rl_agent(req.user_id)
        agent.last_state = redis_conn.get(f"rl_last_state:{req.user_id}")
        agent.last_action = redis_conn.get(f"rl_last_action:{req.user_id}")
        reward = agent.give_feedback(req.feedback)
        agent.process_feedback(reward)
        save_rl_agent(req.user_id, agent)
        q_table_printable = {k: dict(v) for k, v in agent.q_table.items()}
        print(f"Q-table for user {req.user_id}:\n{json.dumps(q_table_printable, indent=2)}")
        return FeedbackResponse(status="success")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- MAIN ---
# To run: uvicorn main:app --reload
