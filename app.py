from __future__ import annotations

from typing import List

import streamlit as st

from backend import RateLimitError, available_models, call_model
from db import (
    clear_messages,
    export_messages,
    get_conversation_title,
    init_db,
    load_messages,
    save_message,
    set_conversation_title,
)
from utils import format_human_timestamp, secure_session_id, serialize_messages, utc_now_iso


init_db()
st.set_page_config(page_title="Secure Multi-LLM Chat", page_icon="🤖", layout="centered")

APP_PASSWORD = st.secrets.get("APP_PASSWORD")
if not APP_PASSWORD:
    st.error(
        "APP_PASSWORD is missing from Streamlit secrets. "
        "Add it via `.streamlit/secrets.toml` locally or Streamlit Cloud settings."
    )
    st.stop()


def _inject_custom_css(dark_mode: bool) -> None:
    base_bg = "#0E1117" if dark_mode else "#F4F5FB"
    panel_bg = "#1F232B" if dark_mode else "#FFFFFF"
    user_color = "#5E63FF"
    assistant_color = "#303446" if dark_mode else "#EEF0FF"
    text_color = "#F4F6FF" if dark_mode else "#1F232B"
    st.markdown(
        f"""
        <style>
        body {{
            background-color: {base_bg};
        }}
        section[data-testid="stSidebar"] {{ background-color: {panel_bg}; }}
        div[data-testid="stChatMessage"] {{
            padding: 0.5rem 1rem;
        }}
        div[data-testid="stChatInput"] {{
            position: sticky;
            bottom: 0;
            background-color: {panel_bg};
            padding-bottom: 1rem;
            border-top: 1px solid rgba(94,99,255,0.1);
        }}
        .chat-wrapper {{
            max-width: 900px;
            margin: 0 auto;
        }}
        .chat-bubble.user {{
            background: linear-gradient(135deg, {user_color}, #8891FF);
            color: #FFFFFF;
            border-radius: 1rem;
            margin-left: auto;
        }}
        .chat-bubble.assistant {{
            background: {assistant_color};
            color: {text_color};
            border-radius: 1rem;
            margin-right: auto;
        }}
        .chat-meta {{
            font-size: 0.75rem;
            opacity: 0.0;
            transition: opacity 0.2s ease;
            text-align: right;
            color: #8891FF;
        }}
        .chat-bubble:hover + .chat-meta {{
            opacity: 1;
        }}
        button[kind="primary"] {{
            border-radius: 999px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _require_auth() -> bool:
    if st.session_state.get("authenticated"):
        return True
    st.title("🔐 LLM Control Center Login")
    with st.form("login_form", clear_on_submit=True):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.user_id = secure_session_id()
                st.success("Authenticated — welcome!")
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


if not _require_auth():
    st.stop()


session_id = st.session_state.get("user_id") or secure_session_id()
st.session_state["user_id"] = session_id

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True
if "stream_response" not in st.session_state:
    st.session_state["stream_response"] = True

_inject_custom_css(st.session_state["dark_mode"])

models = available_models()
model_options = []
default_index = 0
for idx, config in enumerate(models):
    has_secret = config.secret in st.secrets
    label = config.label if has_secret else f"{config.label} (configure {config.secret})"
    model_options.append({"label": label, "key": config.key, "enabled": has_secret})
    if has_secret and default_index == 0:
        default_index = idx

if not model_options:
    st.error("No models configured. Add at least one API key to Streamlit secrets.")
    st.stop()

selected_idx = next(
    (idx for idx, item in enumerate(model_options) if item["enabled"]),
    0,
)

with st.container():
    col_left, col_mid, col_right = st.columns([3, 1, 1])
    with col_left:
        selected_model_label = st.selectbox(
            "Model",
            [item["label"] for item in model_options],
            index=selected_idx,
        )
        selected_model_key = model_options[
            [item["label"] for item in model_options].index(selected_model_label)
        ]["key"]
    with col_mid:
        st.session_state["dark_mode"] = st.toggle(
            "Dark mode", value=st.session_state["dark_mode"], key="dark-mode-toggle"
        )
    with col_right:
        st.session_state["stream_response"] = st.toggle(
            "Streaming", value=st.session_state["stream_response"], key="stream-toggle"
        )

_inject_custom_css(st.session_state["dark_mode"])

title = get_conversation_title(session_id)

with st.expander("Conversation settings", expanded=True):
    new_title = st.text_input("Rename chat", value=title, max_chars=80)
    cols = st.columns(3)
    with cols[0]:
        if st.button("Save title", use_container_width=True):
            set_conversation_title(session_id, new_title or title)
            st.success("Title updated")
            st.rerun()
    with cols[1]:
        if st.button("Clear chat", use_container_width=True):
            clear_messages(session_id)
            st.success("Chat cleared")
            st.rerun()
    with cols[2]:
        if st.button("Log out", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

history = load_messages(session_id)

json_export = serialize_messages(history)
txt_export = "\n\n".join(f"{row['role'].upper()}: {row['content']}" for row in history)

col_export_a, col_export_b = st.columns(2)
with col_export_a:
    st.download_button(
        "⬇️ Export JSON",
        data=json_export.encode("utf-8"),
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True,
    )
with col_export_b:
    st.download_button(
        "⬇️ Export TXT",
        data=txt_export.encode("utf-8"),
        file_name="chat_history.txt",
        mime="text/plain",
        use_container_width=True,
    )

retry_payload = st.session_state.get("retry_payload")
message_to_send = None
forced_model_key = selected_model_key
if retry_payload and st.button("Retry last message", type="secondary"):
    message_to_send = retry_payload["prompt"]
    forced_model_key = retry_payload.get("model_key", selected_model_key)
    st.session_state["retry_payload"] = None

with st.container():
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in history:
        bubble_class = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(msg["role"]):
            st.markdown(
                f'<div class="chat-bubble {bubble_class}">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="chat-meta">{msg["model"]} · {format_human_timestamp(msg["timestamp"])}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

user_prompt = st.chat_input("Ask anything...", disabled=not any(opt["enabled"] for opt in model_options))
if user_prompt:
    message_to_send = user_prompt.strip()
    forced_model_key = selected_model_key

def _is_failure(text: str) -> bool:
    lowered = text.lower()
    failure_keywords = ["provider error", "unavailable", "requires", "sdk missing", "unknown model"]
    return any(key in lowered for key in failure_keywords)


if message_to_send:
    if not message_to_send.strip():
        st.warning("Cannot send an empty message.")
        st.stop()

    active_model_key = forced_model_key
    selected_config = next((cfg for cfg in models if cfg.key == active_model_key), None)
    if not selected_config:
        st.error("Model not found.")
        st.stop()

    chat_history_for_llm = [
        {"role": row["role"], "content": row["content"]} for row in history
    ] + [{"role": "user", "content": message_to_send}]

    with st.chat_message("user"):
        st.markdown(
            f'<div class="chat-bubble user">{message_to_send}</div>',
            unsafe_allow_html=True,
        )

    assistant_placeholder = st.chat_message("assistant")
    stream_area = assistant_placeholder.empty()
    collected_chunks: List[str] = []

    def on_token(chunk: str) -> None:
        collected_chunks.append(chunk)
        stream_area.markdown(
            f'<div class="chat-bubble assistant">{"".join(collected_chunks)}</div>',
            unsafe_allow_html=True,
        )

    try:
        response_text = call_model(
            active_model_key,
            chat_history_for_llm,
            session_id=session_id,
            stream=st.session_state["stream_response"],
            on_token=on_token if st.session_state["stream_response"] else None,
        )
    except RateLimitError as err:
        st.error(str(err))
        st.session_state["retry_payload"] = {
            "prompt": message_to_send,
            "model_key": active_model_key,
        }
        st.stop()

    if not st.session_state["stream_response"]:
        stream_area.markdown(
            f'<div class="chat-bubble assistant">{response_text}</div>',
            unsafe_allow_html=True,
        )

    timestamp = utc_now_iso()
    save_message(session_id, "user", message_to_send, selected_config.label, timestamp)

    if _is_failure(response_text):
        st.warning(response_text)
        st.session_state["retry_payload"] = {
            "prompt": message_to_send,
            "model_key": active_model_key,
        }
    else:
        save_message(
            session_id,
            "assistant",
            response_text,
            selected_config.label,
            utc_now_iso(),
        )
        st.session_state["retry_payload"] = None

    st.rerun()

