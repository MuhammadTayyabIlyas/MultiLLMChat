from __future__ import annotations

from typing import List

import streamlit as st

from backend import RateLimitError, available_models, call_model, evaluate_responses
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
st.set_page_config(
    page_title="AI Chat Studio | Multi-LLM Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_PASSWORD = st.secrets.get("APP_PASSWORD")
if not APP_PASSWORD:
    st.error(
        "APP_PASSWORD is missing from Streamlit secrets. "
        "Add it via `.streamlit/secrets.toml` locally or Streamlit Cloud settings."
    )
    st.stop()


def _inject_custom_css(dark_mode: bool) -> None:
    base_bg = "#0E1117" if dark_mode else "#F0F2F6"
    panel_bg = "#1E1E1E" if dark_mode else "#FFFFFF"
    user_color = "#10A37F" if dark_mode else "#10A37F"
    assistant_color = "#2A2A2A" if dark_mode else "#F7F7F8"
    text_color = "#ECECF1" if dark_mode else "#1F1F1F"
    border_color = "#404040" if dark_mode else "#E5E5E5"

    st.markdown(
        f"""
        <style>
        /* Main Container */
        .main {{
            background-color: {base_bg};
            padding: 0;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background: {panel_bg};
            border-right: 1px solid {border_color};
            padding: 2rem 1rem;
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
        }}

        /* Header */
        .app-header {{
            text-align: center;
            padding: 2rem 0 1rem 0;
            border-bottom: 1px solid {border_color};
            margin-bottom: 2rem;
        }}

        .app-title {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #10A37F, #5E63FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .app-subtitle {{
            color: #8E8EA0;
            font-size: 0.9rem;
        }}

        /* Chat Messages */
        .stChatMessage {{
            background-color: transparent !important;
            padding: 1.5rem 1rem !important;
            margin-bottom: 0.5rem;
        }}

        .stChatMessage[data-testid*="user"] {{
            background: linear-gradient(90deg, rgba(16,163,127,0.1), rgba(16,163,127,0.05)) !important;
            border-left: 3px solid {user_color};
        }}

        .stChatMessage[data-testid*="assistant"] {{
            background: {assistant_color} !important;
            border-left: 3px solid #5E63FF;
        }}

        /* Chat Input */
        .stChatInput {{
            border-top: 1px solid {border_color};
            padding: 1.5rem 0 1rem 0;
            background: {panel_bg};
        }}

        .stChatInput > div {{
            border-radius: 1rem !important;
            border: 2px solid {border_color} !important;
            background: {base_bg} !important;
        }}

        .stChatInput > div:focus-within {{
            border-color: #10A37F !important;
            box-shadow: 0 0 0 3px rgba(16,163,127,0.1);
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 0.5rem;
            font-weight: 500;
            transition: all 0.2s;
        }}

        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        button[kind="primary"] {{
            background: linear-gradient(135deg, #10A37F, #0D8A6A) !important;
            border: none !important;
            color: white !important;
        }}

        button[kind="secondary"] {{
            background: transparent !important;
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
        }}

        /* Selectbox */
        .stSelectbox > div > div {{
            border-radius: 0.5rem;
            border: 1px solid {border_color};
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background: {assistant_color};
            border-radius: 0.5rem;
            border: 1px solid {border_color};
            font-weight: 600;
        }}

        /* Toggle */
        .stCheckbox {{
            padding: 0.5rem 0;
        }}

        /* Download Buttons */
        .stDownloadButton > button {{
            width: 100%;
            border-radius: 0.5rem;
            border: 1px solid {border_color};
            background: {assistant_color};
        }}

        /* Model Badge */
        .model-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #5E63FF, #8891FF);
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }}

        /* Timestamp */
        .timestamp {{
            color: #8E8EA0;
            font-size: 0.75rem;
            margin-top: 0.5rem;
        }}

        /* Footer */
        .app-footer {{
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
            border-top: 1px solid {border_color};
            color: #8E8EA0;
            font-size: 0.85rem;
        }}

        .author-link {{
            color: #10A37F;
            text-decoration: none;
            font-weight: 600;
        }}

        .author-link:hover {{
            text-decoration: underline;
        }}

        /* Comparison Mode Styling */
        .stColumn > div {{
            padding: 0.5rem;
        }}

        .stColumn h3 {{
            color: {text_color};
            font-size: 1rem;
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background: {assistant_color};
            border-radius: 0.5rem;
            border-left: 3px solid #10A37F;
        }}

        /* Evaluation Box */
        .stAlert {{
            border-radius: 0.5rem;
            border-left: 4px solid #10A37F;
        }}

        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _require_auth() -> bool:
    if st.session_state.get("authenticated"):
        return True

    # Modern login screen
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🤖</h1>
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">AI Chat Studio</h1>
            <p style="color: #8E8EA0; font-size: 1rem;">Multi-LLM Platform by Muhammad Tayyab ILYAS</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form", clear_on_submit=True):
            st.markdown("### 🔐 Welcome Back")
            password = st.text_input("Enter your password", type="password", placeholder="Password")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submitted:
                if password == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.session_state.user_id = secure_session_id()
                    st.success("✅ Authenticated successfully!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Please try again.")

        st.markdown(
            """
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(94,99,255,0.1); border-radius: 0.5rem;">
                <p style="color: #8E8EA0; font-size: 0.85rem; margin: 0;">
                    🌟 Access 11 AI providers in one unified interface
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
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

# App Header
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">🤖 AI Chat Studio</div>
        <div class="app-subtitle">Powered by 11 Leading AI Models | Built by Muhammad Tayyab ILYAS</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Model Selection
    models = available_models()
    model_options = []
    default_index = 0
    for idx, config in enumerate(models):
        has_secret = config.secret in st.secrets
        label = config.label if has_secret else f"{config.label} (🔒 locked)"
        model_options.append({"label": label, "key": config.key, "enabled": has_secret, "provider": config.provider})
        if has_secret and default_index == 0:
            default_index = idx

    if not model_options:
        st.error("No models configured. Add at least one API key to Streamlit secrets.")
        st.stop()

    # Comparison Mode Toggle
    if "comparison_mode" not in st.session_state:
        st.session_state["comparison_mode"] = False

    st.session_state["comparison_mode"] = st.toggle(
        "⚖️ Comparison Mode",
        value=st.session_state["comparison_mode"],
        key="comparison-toggle",
        help="Compare two different AI models side-by-side"
    )

    # Initialize variables
    model_a_key = None
    model_b_key = None
    selected_model_key = None

    if st.session_state["comparison_mode"]:
        st.markdown("#### Model A")
        enabled_options_a = [item for item in model_options if item["enabled"]]
        if not enabled_options_a:
            st.error("No models available")
            st.stop()

        model_a_label = st.selectbox(
            "First Model",
            [item["label"] for item in enabled_options_a],
            index=0,
            key="model_a_select"
        )
        model_a_key = next(item["key"] for item in enabled_options_a if item["label"] == model_a_label)
        model_a_provider = next(item["provider"] for item in enabled_options_a if item["label"] == model_a_label)

        st.markdown("#### Model B")
        # Filter out Model A's provider to prevent same provider comparison
        enabled_options_b = [item for item in enabled_options_a if item["provider"] != model_a_provider]

        if not enabled_options_b:
            st.warning("⚠️ Please enable at least one model from a different provider")
            model_b_key = None
        else:
            model_b_label = st.selectbox(
                "Second Model",
                [item["label"] for item in enabled_options_b],
                index=0,
                key="model_b_select"
            )
            model_b_key = next(item["key"] for item in enabled_options_b if item["label"] == model_b_label)
    else:
        # SINGLE MODEL MODE
        selected_idx = next(
            (idx for idx, item in enumerate(model_options) if item["enabled"]),
            0,
        )

        selected_model_label = st.selectbox(
            "🤖 Select AI Model",
            [item["label"] for item in model_options],
            index=selected_idx,
            help="Choose from different AI providers"
        )
        selected_model_key = model_options[
            [item["label"] for item in model_options].index(selected_model_label)
        ]["key"]

    st.divider()

    # Display Options
    st.markdown("### 🎨 Display")
    st.session_state["dark_mode"] = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state["dark_mode"],
        key="dark-mode-toggle"
    )

    st.session_state["stream_response"] = st.toggle(
        "⚡ Streaming",
        value=st.session_state["stream_response"],
        key="stream-toggle",
        help="Enable real-time response streaming"
    )

    st.divider()

    # Conversation Management
    st.markdown("### 💬 Conversation")
    title = get_conversation_title(session_id)
    new_title = st.text_input("💬 Chat Title", value=title, max_chars=80, placeholder="Enter chat title")

    if st.button("💾 Save Title", use_container_width=True):
        set_conversation_title(session_id, new_title or title)
        st.success("✅ Title saved!")
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        clear_messages(session_id)
        st.success("✅ Chat cleared!")
        st.rerun()

    st.divider()

    # Export Options
    st.markdown("### 📥 Export")
    history = load_messages(session_id)
    json_export = serialize_messages(history)
    txt_export = "\n\n".join(f"{row['role'].upper()}: {row['content']}" for row in history)

    st.download_button(
        "📄 Download JSON",
        data=json_export.encode("utf-8"),
        file_name=f"chat_{session_id[:8]}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.download_button(
        "📝 Download TXT",
        data=txt_export.encode("utf-8"),
        file_name=f"chat_{session_id[:8]}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()

    # Logout
    if st.button("🚪 Sign Out", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Sidebar Footer
    st.markdown(
        """
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); text-align: center;">
            <p style="font-size: 0.75rem; color: #8E8EA0; margin: 0;">
                Built with ❤️ by<br/>
                <a href="https://github.com/MuhammadTayyabIlyas" target="_blank" style="color: #10A37F; text-decoration: none; font-weight: 600;">
                    Muhammad Tayyab ILYAS
                </a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

_inject_custom_css(st.session_state["dark_mode"])

history = load_messages(session_id)

retry_payload = st.session_state.get("retry_payload")
message_to_send = None
forced_model_key = selected_model_key

# Retry button if needed
if retry_payload:
    if st.button("🔄 Retry Last Message", type="secondary"):
        message_to_send = retry_payload["prompt"]
        forced_model_key = retry_payload.get("model_key", selected_model_key)
        st.session_state["retry_payload"] = None

# Chat Messages
for msg in history:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        st.markdown(
            f'<div class="timestamp">💬 <span class="model-badge">{msg["model"]}</span>{format_human_timestamp(msg["timestamp"])}</div>',
            unsafe_allow_html=True,
        )

user_prompt = st.chat_input(
    "💬 Message AI Chat Studio...",
    disabled=not any(opt["enabled"] for opt in model_options),
    key="chat_input"
)
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

    chat_history_for_llm = [
        {"role": row["role"], "content": row["content"]} for row in history
    ] + [{"role": "user", "content": message_to_send}]

    with st.chat_message("user", avatar="🧑"):
        st.markdown(message_to_send)

    # COMPARISON MODE
    if st.session_state.get("comparison_mode") and model_a_key and model_b_key:
        config_a = next((cfg for cfg in models if cfg.key == model_a_key), None)
        config_b = next((cfg for cfg in models if cfg.key == model_b_key), None)

        if not config_a or not config_b:
            st.error("One or both models not found.")
            st.stop()

        # Call both models (no streaming in comparison mode for better table display)
        response_a = ""
        response_b = ""

        with st.spinner(f"🤖 {config_a.label} is thinking..."):
            try:
                response_a = call_model(
                    model_a_key,
                    chat_history_for_llm,
                    session_id=session_id,
                    stream=False,
                    on_token=None,
                )
            except RateLimitError as err:
                st.error(str(err))
                st.stop()

        with st.spinner(f"🤖 {config_b.label} is thinking..."):
            try:
                response_b = call_model(
                    model_b_key,
                    chat_history_for_llm,
                    session_id=session_id,
                    stream=False,
                    on_token=None,
                )
            except RateLimitError as err:
                st.error(str(err))
                st.stop()

        # GPT-5 nano Evaluation
        with st.spinner("🎯 Evaluating responses..."):
            evaluation = evaluate_responses(
                message_to_send,
                response_a,
                response_b,
                config_a.label,
                config_b.label,
            )

        # Display in Table Format with Color Coding
        dark_mode = st.session_state.get("dark_mode", True)

        if dark_mode:
            color_a = "#10A37F"  # Green for Model A
            color_b = "#5E63FF"  # Blue for Model B
            color_eval = "#FF6B6B"  # Red for Evaluator
            bg_color = "#1E1E1E"
            text_color = "#ECECF1"
        else:
            color_a = "#0D8A6A"
            color_b = "#4A4FCC"
            color_eval = "#CC5555"
            bg_color = "#FFFFFF"
            text_color = "#1F1F1F"

        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <table style="width: 100%; border-collapse: collapse; background: {bg_color}; border-radius: 0.5rem; overflow: hidden;">
                <thead>
                    <tr style="background: linear-gradient(135deg, {color_a}20, {color_b}20);">
                        <th style="padding: 1rem; text-align: left; color: {text_color}; font-weight: 600; border-bottom: 2px solid {color_a};">Model</th>
                        <th style="padding: 1rem; text-align: left; color: {text_color}; font-weight: 600; border-bottom: 2px solid {color_b};">Response</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 1rem; vertical-align: top; border-left: 4px solid {color_a}; color: {color_a}; font-weight: 600;">
                            🤖 {config_a.label}
                        </td>
                        <td style="padding: 1rem; color: {text_color}; border-bottom: 1px solid #404040;">
                            {response_a}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 1rem; vertical-align: top; border-left: 4px solid {color_b}; color: {color_b}; font-weight: 600;">
                            🤖 {config_b.label}
                        </td>
                        <td style="padding: 1rem; color: {text_color}; border-bottom: 1px solid #404040;">
                            {response_b}
                        </td>
                    </tr>
                    <tr style="background: {color_eval}10;">
                        <td style="padding: 1rem; vertical-align: top; border-left: 4px solid {color_eval}; color: {color_eval}; font-weight: 600;">
                            🎯 GPT-5 Nano
                        </td>
                        <td style="padding: 1rem; color: {color_eval}; font-weight: 500;">
                            {evaluation}
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Save messages
        timestamp = utc_now_iso()
        save_message(session_id, "user", message_to_send, "Comparison", timestamp)

        comparison_result = f"**{config_a.label}:**\n{response_a}\n\n**{config_b.label}:**\n{response_b}\n\n**Evaluation:** {evaluation}"
        save_message(session_id, "assistant", comparison_result, "Comparison", utc_now_iso())

        st.rerun()

    # SINGLE MODEL MODE
    else:
        active_model_key = forced_model_key
        selected_config = next((cfg for cfg in models if cfg.key == active_model_key), None)
        if not selected_config:
            st.error("Model not found.")
            st.stop()

        assistant_placeholder = st.chat_message("assistant", avatar="🤖")
        stream_area = assistant_placeholder.empty()
        collected_chunks: List[str] = []

        def on_token(chunk: str) -> None:
            collected_chunks.append(chunk)
            stream_area.markdown("".join(collected_chunks))

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
            stream_area.markdown(response_text)

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

