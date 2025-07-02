import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import tiktoken
from datetime import datetime

# ========== SETUP ==========
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== MODEL SELECTION & SETTINGS ==========
MODELS = {
    "gpt-4o": {"price_input": 0.0025, "price_output": 0.01},
    "gpt-3.5-turbo": {"price_input": 0.0005, "price_output": 0.0015},
}

st.sidebar.title("Assistant Settings")
model_choice = st.sidebar.selectbox("Model", list(MODELS.keys()), index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
top_p = st.sidebar.slider("Top-p (Nucleus Sampling)", 0.1, 1.0, 0.85, 0.01)
max_tokens = st.sidebar.slider("Max Tokens", 256, 2048, 1024, 32)

# ======= SYSTEM PROMPT (now with formatting) ==========
default_system_prompt = (
    "You are a senior IT support specialist. "
    "Always answer briefly and clearly, just like you would in a corporate chat. "
    "If the user's message already contains enough details, immediately give practical troubleshooting steps, not generic advice. "
    "If something is unclear or not enough info, ask ONLY one clarifying question, specific to the actual problem (not general, not formal). "
    "Never make up extra facts, error messages or scenarios. Never invent excuses or 'polite' phrases. "
    "Never write intros like 'How can I help you?' or 'Thank you for contacting support.' "
    "Never repeat previous cases or answer with examples. "
    "If a solution can be given immediately, provide it as a neat, easy-to-read numbered list or checklist, using markdown formatting (bold for step headers, code for commands/paths if needed). "
    "Add blank lines between steps for readability. "
    "If the problem is generic ('not working'), quickly ask what user already tried, or request the specific error message, but NEVER write more than two sentences in a row."
)

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = default_system_prompt

st.sidebar.markdown("**System prompt:**")
st.session_state.system_prompt = st.sidebar.text_area(
    "", st.session_state.system_prompt, height=160
)

# ============= TOKEN COUNTER =============
def count_tokens(messages, model):
    enc = tiktoken.encoding_for_model(model)
    return sum(len(enc.encode(m["content"])) for m in messages)

# ============= HISTORY + RESET ============
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "history" not in st.session_state:
    st.session_state.history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def reset_chat():
    st.session_state.chat_history = []
    st.session_state.history = []
    st.session_state.user_input = ""

# =========== PAGE UI ===============
st.set_page_config(page_title="IT Assistant Pro", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 IT Assistant Chat (Pro)</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Chat"):
        reset_chat()
        st.rerun()
with col2:
    if st.download_button("⬇️ Download History", data=str(st.session_state.history), file_name="chat_history.txt"):
        pass

# ===== INSTRUCTIONS FOR USERS =====
with st.expander("ℹ️ How to use this assistant? (click to show instructions)", expanded=False):
    st.markdown("""
**How to get the best support:**
- Describe your IT problem in one or two sentences.
- Mention the device, app or error message if possible (e.g. *'VPN not working on Windows 11, error 809'*).
- The bot will answer with a clear checklist or ask you only ONE clarifying question if needed.
- No unnecessary questions or chit-chat — just real IT support, fast!
    """)

# ====== Show chat (only real messages) ======
for msg in st.session_state.chat_history:
    role = "🧑‍💻 User" if msg["role"] == "user" else "🤖 Bot"
    # Render as markdown for assistant
    if msg["role"] == "assistant":
        st.markdown(
            f"<div style='margin-bottom: 10px;'><b>{role}:</b></div>", unsafe_allow_html=True,
        )
        st.markdown(msg["content"])
    else:
        st.markdown(
            f"<div style='margin-bottom: 10px;'><b>{role}:</b> <br/>{msg['content']}</div>",
            unsafe_allow_html=True,
        )

# ============= INPUT FORM =============
with st.form(key="chat_form"):
    user_input = st.text_area("Type your message:", height=80, key="user_input")
    submit = st.form_submit_button("Send")

# =========== HANDLING USER MESSAGE ============
if submit and user_input.strip():
    prompt_messages = [
        {"role": "system", "content": st.session_state.system_prompt}
    ]
    prompt_messages += st.session_state.chat_history

    try:
        response = client.chat.completions.create(
            model=model_choice,
            messages=prompt_messages + [{"role": "user", "content": user_input}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        assistant_reply = response.choices[0].message.content

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

        prompt_tokens = count_tokens(prompt_messages[:-1], model_choice)
        completion_tokens = count_tokens([{"role": "assistant", "content": assistant_reply}], model_choice)
        price = (
            prompt_tokens * MODELS[model_choice]["price_input"] / 1000
            + completion_tokens * MODELS[model_choice]["price_output"] / 1000
        )

        st.session_state.history.append({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model": model_choice,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "price_usd": round(price, 6),
            "user_message": user_input,
            "assistant_reply": assistant_reply
        })

        st.markdown(
            f"<div style='color:gray;font-size:0.92em;'>Tokens: {prompt_tokens+completion_tokens} | Cost: ${price:.6f}</div>",
            unsafe_allow_html=True,
        )
        if len(st.session_state.chat_history) > 40:
            st.session_state.chat_history = st.session_state.chat_history[-40:]
        st.session_state.user_input = ""
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
    st.rerun()

# ============ FOOTER ============
st.markdown(
    "<hr><div style='text-align:center;color:gray;font-size:0.95em;'>Created by [Margarita]. Demo for AI Engineer / Prompt Engineer portfolio, 2025.</div>",
    unsafe_allow_html=True,
)
