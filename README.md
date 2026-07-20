# QubitPath AI

**Author:** Suman Poola  
**Mentor:** Dr. Qingyang Xiao

QubitPath AI is a GitHub- and Streamlit-ready prototype for an AI-assisted quantum-computing education platform. It supports entry, intermediate, and advanced learners through structured lessons, quizzes, mini-games, quantum-circuit simulations, progress analytics, adaptive recommendations, and an AI tutor prototype.

## Main capabilities

- Three learning levels: Entry, Intermediate, and Advanced
- Recorded-learning curriculum and curated references
- Prototype live-tutoring reservation workflow
- Quantum circuit lab powered by Qiskit statevector simulation
- Quizzes, mini-games, XP, streaks, and learner progress tracking
- Random-forest progress forecasting
- Neural-network learner-support classification
- Reinforcement-learning-style feedback adaptation
- Retrieval-based Professor Qubit tutor
- Interactive Plotly dashboards and downloadable reports
- Responsible-AI guardrails and production roadmap

## Repository structure

```text
qubitpath-ai-quantum-education/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── docs/
│   ├── ARCHITECTURE.md
│   └── DEPLOYMENT.md
├── notebooks/
│   └── AI_Quantum_Computing_Education_Platform_Colab.ipynb
└── sample_data/
    └── learner_progress_example.csv
```

## Run locally

Python 3.11 or 3.12 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Extract this ZIP.
2. Create a new GitHub repository.
3. Upload **the contents inside this folder** to the root of the repository. `app.py` and `requirements.txt` must be visible at the repository root.
4. Commit the files to the `main` branch.
5. Open Streamlit Community Cloud and create a new app.
6. Select the GitHub repository and branch `main`.
7. Set the main file path to `app.py`.
8. In advanced settings, select Python 3.12 when available.
9. Deploy the app.

No secrets or API keys are required for the default prototype.

## Colab workflow

Open `notebooks/AI_Quantum_Computing_Education_Platform_Colab.ipynb` in Google Colab to inspect the model logic, regenerate the application, or launch a temporary development tunnel.

## Data and model notice

The demonstration models are trained on synthetic data generated inside the application. Their predictions illustrate the workflow only and must not be used as real grades, admissions decisions, disciplinary actions, or high-stakes learner evaluations. A production deployment requires consented and de-identified data, model validation, accessibility testing, privacy controls, human oversight, and secure persistent storage.

## Production integrations

The live-session panel currently demonstrates booking logic. Production video meetings require authenticated Zoom, Microsoft Teams, or Google Meet APIs. A production generative-AI tutor should use grounded retrieval, citations, moderation, rate limits, evaluation, and secret management rather than placing credentials directly in the repository.

## Troubleshooting

- **Dependency installation takes several minutes:** Qiskit visualization dependencies are larger than a basic Streamlit deployment.
- **Circuit drawing error:** Verify that all requirements installed successfully and reboot the Streamlit app.
- **App cannot find `app.py`:** Confirm `app.py` is in the GitHub repository root and that Streamlit's main file path is exactly `app.py`.
- **Blank or stale deployment:** Reboot the app from the Streamlit management console after dependency changes.
- **Python incompatibility:** Redeploy with Python 3.11 or 3.12.
