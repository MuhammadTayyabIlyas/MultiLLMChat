Never commit any secrets to GitHub. Use Streamlit Cloud secrets or environment variables.

# Secure Multi-LLM Streamlit Chat

This project is a production-ready Streamlit chat interface that mirrors the ChatGPT experience while letting you swap between multiple LLM providers (OpenAI, Anthropic, Gemini, Qwen, Groq, DeepSeek). Chats are stored server-side in SQLite, streaming is supported where provider SDKs allow it, and access is protected by a password sourced from `st.secrets`.

## Features
- Password gate backed by `APP_PASSWORD` in `st.secrets`.
- Model switcher with provider availability hints and streaming toggle.
- Persistent chat history in `data/chat_history.db`, with rename, clear, and export (JSON/TXT) controls.
- Per-session rate limiting (fast in-memory guard + durable SQLite counter).
- Dark/light mode toggle, sticky chat input, retry-on-failure workflow.
- Modular `backend.py` functions (`call_openai`, `call_anthropic`, etc.) modeled after `Testing.py` to simplify adding new providers.

## Required secrets
Create `.streamlit/secrets.toml` locally and set the same keys in Streamlit Cloud.

```
APP_PASSWORD = "example-strong-password"
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "anthropic-..."
GEMINI_API_KEY = "ya29...."
QWEN_API_KEY = "dashscope-..."
GROQ_API_KEY = "gsk-..."
DEEPSEEK_API_KEY = "deepseek-..."
```

> The app automatically hides unavailable models when the matching key is missing, and displays a friendly message instead of leaking raw errors.

## Local development
1. **Install Python 3.10+** and create a virtualenv.
2. **Install deps**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Secrets for local testing** (do **not** commit):
   ```
   # .streamlit/secrets.toml
   APP_PASSWORD = "dev-password"
   OPENAI_API_KEY = "..."
   # add whichever providers you want
   ```
   You can also keep a local `.env` file for your own reference (ignored by git):
   ```
   # .env (copy these into .streamlit/secrets.toml before running)
   APP_PASSWORD=dev-password
   OPENAI_API_KEY=sk-...
   ```
4. **Run**:
   ```bash
   streamlit run app.py
   ```
5. The SQLite file is created at `data/chat_history.db`. Optional exports saved via the UI can also be written to a local `Out/` folder (ignored by git).

## Deploy on Streamlit Cloud
1. Push this repository to GitHub (avoid committing anything inside `data/`, `Out/`, or secrets).
2. In Streamlit Cloud, create a new app pointing to `app.py`.
3. Under **App settings → Secrets**, paste the TOML snippet above with only the providers you need.
4. Click **Deploy**. The Cloud runtime uses the same SQLite file path (stored on the ephemeral filesystem) and automatically respects rate limits + password gate.

## Database & storage
- `db.py` manages all persistence: `messages`, `conversations`, and `rate_limits` tables.
- Data lives under `data/chat_history.db`; this path works locally and on Streamlit Cloud.
- Chats can be renamed, cleared, or exported. Exports are streamed to the browser and may optionally be saved manually to the ignored `Out/` directory.

## Rate limiting
- **Burst limit**: 8 requests per session per rolling minute (memory-only guard).
- **Daily limit**: 200 requests persisted in SQLite per session.
- When a limit is reached the user sees a friendly notice and can retry after the timer elapses.

## Troubleshooting
| Symptom | Fix |
| --- | --- |
| Login screen rejects password | Ensure `APP_PASSWORD` is defined in `st.secrets`. |
| Model shows “configure secret” | Add the referenced key (`OPENAI_API_KEY`, etc.) to secrets. |
| “Provider error — try again” | Transient upstream issue; use the Retry button or switch models. |
| “Daily limit reached” | Wait until the 24h window resets or clear the SQLite `rate_limits` table for local testing. |
| No chat history saved | Confirm the app can write to `data/` (not on read-only storage). |

## File structure
```
.
├── app.py                # Streamlit UI, auth, chat UX
├── backend.py            # Provider-specific callers + rate limiting
├── db.py                 # SQLite helpers (messages, titles, rate limits)
├── utils.py              # Shared helpers (timestamps, session IDs, streaming chunks)
├── requirements.txt      # Pip dependencies
├── README.md             # Deployment + usage guide
├── .gitignore            # Excludes data/, Out/, secrets, caches
└── .streamlit/config.toml
```

## Requirements snapshot
```
streamlit>=1.39.0
requests>=2.32.0
openai>=1.50.0
anthropic>=0.74.0
google-generativeai>=0.7.2
groq>=0.11.0
python-dotenv>=1.0.1
```

