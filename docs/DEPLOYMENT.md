# Deployment Guide

## GitHub upload

Upload the files from the extracted project folder to the root of a GitHub repository. Do not upload only the outer ZIP file because Streamlit must read `app.py` and `requirements.txt` directly from the repository.

## Streamlit Community Cloud settings

- Repository: your GitHub repository
- Branch: `main`
- Main file path: `app.py`
- Python: 3.11 or 3.12
- Secrets: none required for the default demo

## Updating the app

Push changes to the selected branch. Streamlit normally detects the new commit and rebuilds the application. When requirements change, use the app management console to reboot the app.

## Future secrets

Never commit service tokens. For Zoom, Teams, Google Meet, an external LLM, or a database, place credentials in Streamlit secrets or another managed secret store, then read them at runtime.

Example local-only `.streamlit/secrets.toml` structure:

```toml
[meeting_provider]
client_id = "replace-me"
client_secret = "replace-me"

[llm]
api_key = "replace-me"
```

The repository `.gitignore` excludes `.streamlit/secrets.toml`.
