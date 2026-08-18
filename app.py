# QubitPath AI - Streamlit application
# Integrated with the new animated UI from app_1.py.
# No API keys are required for the default demo.

from __future__ import annotations

import json
import math
import random
from datetime import date, datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


APP_TITLE = "QubitPath AI"
AUTHOR_NAME = "Suman Poola"
MENTOR_NAME = "Dr. Qingyang Xiao"
LEVELS = ["Entry", "Intermediate", "Advanced"]
LEVEL_TO_NUM = {"Entry": 1, "Intermediate": 2, "Advanced": 3}
IBM_COURSES_URL = "https://quantum.cloud.ibm.com/learning/en/courses"
QISKIT_DOCS_URL = "https://qiskit.qotlabs.org/docs/"

st.set_page_config(
    page_title=f"{APP_TITLE} | Quantum Learning",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root { --qp-purple:#6f4bf2; --qp-cyan:#13c8c8; --qp-ink:#172033; }
      .stApp { background: linear-gradient(180deg,#f8f7ff 0%,#f4fbff 100%); }
      .block-container { padding-top: 0; padding-bottom: 0; max-width: 100%; padding-left: 0; padding-right: 0;}
      .hero {
        padding: 1.45rem 1.6rem; border-radius: 24px; color: white;
        background: radial-gradient(circle at 10% 20%,#8a6cff 0,#6f4bf2 35%,#27357f 100%);
        box-shadow: 0 18px 50px rgba(61,54,140,.20); margin-bottom: 1rem;
      }
      .hero h1 { margin:0; font-size:2.35rem; }
      .hero p { margin:.45rem 0 0; opacity:.94; font-size:1.04rem; }
      .glass-card {
        background:rgba(255,255,255,.92); border:1px solid rgba(111,75,242,.12);
        padding:1rem 1.1rem; border-radius:18px; box-shadow:0 8px 26px rgba(30,38,90,.07);
        min-height:145px;
      }
      [data-testid="stHeader"] {
        background: transparent;}
      .pill { display:inline-block; padding:.22rem .62rem; border-radius:999px;
        background:#eee9ff; color:#4b35b4; font-weight:700; font-size:.78rem; margin-right:.3rem; }
      .success-box { background:#ecfff7; border-left:5px solid #20a977; padding:.8rem 1rem; border-radius:12px; }
      .info-box { background:#eef8ff; border-left:5px solid #368bea; padding:.8rem 1rem; border-radius:12px; }
      .warning-box { background:#fff8e7; border-left:5px solid #e4a11b; padding:.8rem 1rem; border-radius:12px; }
      .small-muted { color:#667085; font-size:.86rem; }
      div[data-testid="stMetric"] { background:white; border:1px solid #e8e8f5; padding:.7rem; border-radius:16px; }
      section[data-testid="stSidebar"] { background:linear-gradient(180deg,#171b36,#202958); color:white; }
      section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color:#f4f5ff !important; }
      section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color:white; }
      .project-credits-sidebar {
        margin:.45rem 0 1rem; padding:.72rem .78rem; border-radius:14px;
        background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.16);
        line-height:1.42; font-size:.86rem;
      }
      .project-credits-sidebar .credit-label { color:#c9cdef; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
      .project-credits-sidebar .mentor-credit { margin-top:.52rem; padding-top:.52rem; border-top:1px solid rgba(255,255,255,.13); }
      .project-credits-page { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.8rem; margin:.2rem 0 1.25rem; }
      .credit-card { background:rgba(255,255,255,.94); border:1px solid rgba(111,75,242,.16); padding:1rem 1.1rem; border-radius:16px; box-shadow:0 7px 22px rgba(30,38,90,.06); }
      .credit-role { color:#6043d8; font-size:.76rem; font-weight:800; text-transform:uppercase; letter-spacing:.06em; margin-bottom:.2rem; }
      .credit-name { color:#172033; font-size:1.05rem; font-weight:750; }
      @media (max-width:700px) { .project-credits-page { grid-template-columns:1fr; } }
      .professor { display:flex; gap:1rem; align-items:center; background:white; padding:1rem; border-radius:18px; border:1px solid #e9e7ff; }
      .avatar { width:68px; height:68px; border-radius:50%; display:flex; align-items:center; justify-content:center;
        font-size:2rem; background:linear-gradient(135deg,#6f4bf2,#13c8c8); color:white; }
    .hometitle {
            color: #FFFDD0;
            font-family: "Impact", "Arial Black", sans-serif;
            text-align: center;
            position: relative;
            z-index: 2;
            margin-top: 75vh;
        }
    .testhome-title {
        display: flex;
        flex-direction: column;
        height: 100vh;
        size: 20px;
        font-size: 20vh;
        justify-content: center;
        background: radial-gradient(circle at 50% 35%, #202958 0%, #111936 52%, #0c1127 100%);
        width: 100%;
        overflow: hidden;
    }
    .testhome-path-options{
        display: flex;
        flex-direction: row;
        height: 100vh;
        justify-content: center;
        align-items: center;
        background:#0c1127;
        max-width: 100%;
        border-radius: 2vh;
        gap: 3%;
    }
    .testhome-about{
    height:30vh;
    background:#0c1127;
    padding-top:3rem;
    }
    .testhome-about h4 {
    text-align: center;
    color:#FFFDD0;
    size: 5vw;
    }
    .testhome-about h1{
    text-align: center;
    color:#5dcaa5;
    size: 10vw;
    }
    .testhome-path{
    flex: 1;
    border: 5px ;
    text-align: center;
    max-width: 30vw;
    border-radius: 30px;
    height: 80vh;
    background-color: #4fab8c;
    }
    .testhome-path h2{
    color: #FFFDD0;
    margin-top: 15px;
    }
    .landing-cta-note {
      background:#0c1127; color:#dfe6ff; text-align:center; margin:0; padding:.35rem 1rem 1.1rem;
    }
    div[data-testid="stHorizontalBlock"]:has(button[kind="primary"]) {
      gap:1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


CURRICULUM: dict[str, list[dict[str, Any]]] = {
    "Entry": [
        {
            "id": "E1",
            "title": "Bits, Qubits, and the Quantum Mindset",
            "duration": 35,
            "difficulty": 1,
            "objectives": ["Compare a classical bit with a qubit", "Read ket notation", "Identify realistic quantum-computing use cases"],
            "summary": "A bit is either 0 or 1. A qubit is described by a normalized complex state vector and can be measured to produce a classical result.",
            "activity": "Classify five everyday problems as classical, quantum-inspired, or potentially quantum-relevant.",
            "recording": "Recorded lesson: From bits to qubits",
            "resource": "https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/single-systems/introduction",
        },
        {
            "id": "E2",
            "title": "Superposition and Measurement",
            "duration": 45,
            "difficulty": 1,
            "objectives": ["Explain amplitudes and probabilities", "Apply normalization", "Interpret repeated measurement counts"],
            "summary": "Superposition is a linear combination of basis states. Measurement converts amplitudes into outcome probabilities through the Born rule.",
            "activity": "Use the Quantum Coin mini-game to compare predicted and sampled outcomes.",
            "recording": "Recorded lesson: Why quantum measurement is probabilistic",
            "resource": "https://qiskit.qotlabs.org/learning/courses/basics-of-quantum-information/single-systems/quantum-information",
        },
        {
            "id": "E3",
            "title": "Quantum Gates and Circuit Diagrams",
            "duration": 50,
            "difficulty": 1,
            "objectives": ["Recognize X, H, Z, and rotation gates", "Read a circuit left to right", "Predict basic single-qubit outcomes"],
            "summary": "Quantum gates are reversible linear operations. The X gate flips basis states, while the H gate creates and removes equal superpositions.",
            "activity": "Build a single-qubit circuit and compare your prediction with Qiskit simulation.",
            "recording": "Recorded lesson: Gate-by-gate circuit reading",
            "resource": "https://qiskit.qotlabs.org/docs/guides/circuit-library",
        },
        {
            "id": "E4",
            "title": "Your First Qiskit Program",
            "duration": 55,
            "difficulty": 1,
            "objectives": ["Create a QuantumCircuit", "Simulate a statevector", "Sample measurement counts"],
            "summary": "Qiskit represents quantum programs as circuits. A local statevector simulator is ideal for learning before using real quantum hardware.",
            "activity": "Create a Bell state and explain why 01 and 10 should not appear in the ideal simulator.",
            "recording": "Recorded lab: Build and run a Bell circuit",
            "resource": "https://qiskit.qotlabs.org/docs/guides/quick-start",
        },
    ],
    "Intermediate": [
        {
            "id": "I1",
            "title": "Multiple Qubits and Tensor Products",
            "duration": 60,
            "difficulty": 2,
            "objectives": ["Determine multi-qubit state dimensions", "Interpret basis ordering", "Construct product states"],
            "summary": "The state space grows as 2^n for n qubits. Tensor products combine individual systems into a joint mathematical description.",
            "activity": "Calculate the eight basis states for a three-qubit register.",
            "recording": "Recorded lesson: Scaling from one qubit to many",
            "resource": "https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/multiple-systems/introduction",
        },
        {
            "id": "I2",
            "title": "Entanglement and Bell States",
            "duration": 65,
            "difficulty": 2,
            "objectives": ["Distinguish correlation from entanglement", "Prepare a Bell state", "Interpret joint measurements"],
            "summary": "Entangled states cannot be written as a product of independent subsystem states. Bell states are the standard two-qubit examples.",
            "activity": "Run the Bell-state lab for several shot counts and compare statistical variation.",
            "recording": "Recorded lesson: Entanglement without faster-than-light messaging",
            "resource": "https://qiskit.qotlabs.org/learning/courses/basics-of-quantum-information/entanglement-in-action/introduction",
        },
        {
            "id": "I3",
            "title": "Teleportation and Superdense Coding",
            "duration": 75,
            "difficulty": 2,
            "objectives": ["Describe the role of shared entanglement", "Track classical communication", "Separate state transfer from matter transfer"],
            "summary": "Quantum teleportation transfers an unknown quantum state using shared entanglement plus two classical bits; it does not transmit matter or violate causality.",
            "activity": "Draw the information flow and label quantum versus classical channels.",
            "recording": "Recorded lesson: Quantum communication protocols",
            "resource": "https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/entanglement-in-action/quantum-teleportation",
        },
        {
            "id": "I4",
            "title": "Quantum Algorithms: Deutsch–Jozsa and Grover",
            "duration": 85,
            "difficulty": 2,
            "objectives": ["Explain oracle-based algorithms", "Describe amplitude amplification", "Compare query complexity"],
            "summary": "Early quantum algorithms reveal how interference can suppress wrong answers and amplify useful ones. Grover search gives a quadratic query improvement for unstructured search.",
            "activity": "Trace one Grover iteration for a two-qubit search space.",
            "recording": "Recorded lesson: Interference as an algorithmic resource",
            "resource": "https://quantum.cloud.ibm.com/learning/en/courses/fundamentals-of-quantum-algorithms",
        },
    ],
    "Advanced": [
        {
            "id": "A1",
            "title": "Density Matrices, Noise, and Channels",
            "duration": 90,
            "difficulty": 3,
            "objectives": ["Represent mixed states", "Calculate purity conceptually", "Describe common noise channels"],
            "summary": "Density matrices describe pure and mixed states. Quantum channels model physically allowed transformations, including noise and decoherence.",
            "activity": "Compare ideal and noisy state descriptions and identify which information is lost.",
            "recording": "Recorded seminar: From statevectors to open systems",
            "resource": "https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information",
        },
        {
            "id": "A2",
            "title": "Variational Algorithms and VQE",
            "duration": 100,
            "difficulty": 3,
            "objectives": ["Describe hybrid quantum-classical loops", "Identify ansatz and optimizer roles", "Interpret expectation-value objectives"],
            "summary": "Variational algorithms use a parameterized quantum circuit and a classical optimizer. VQE estimates low-energy states of a Hamiltonian.",
            "activity": "Sketch a VQE workflow and identify possible sources of optimization failure.",
            "recording": "Recorded lab: Anatomy of a VQE experiment",
            "resource": "https://quantum.cloud.ibm.com/learning/en/courses/quantum-chem-with-vqe",
        },
        {
            "id": "A3",
            "title": "QAOA and Combinatorial Optimization",
            "duration": 100,
            "difficulty": 3,
            "objectives": ["Map a cost function to a Hamiltonian", "Explain alternating operators", "Evaluate approximation quality"],
            "summary": "QAOA alternates problem and mixer operators. It is a framework for experimenting with approximate solutions to combinatorial problems.",
            "activity": "Encode a small Max-Cut instance and discuss measurement-to-solution post-processing.",
            "recording": "Recorded workshop: Hybrid optimization with QAOA",
            "resource": "https://qiskit.qotlabs.org/learning/courses/variational-algorithm-design",
        },
        {
            "id": "A4",
            "title": "Quantum Error Correction and Fault Tolerance",
            "duration": 110,
            "difficulty": 3,
            "objectives": ["Explain logical versus physical qubits", "Describe syndrome measurement", "Identify the purpose of fault tolerance"],
            "summary": "Quantum error correction protects logical information by encoding it across many physical qubits and extracting error syndromes without directly reading the logical state.",
            "activity": "Work through a three-qubit repetition-code example and identify its limitations.",
            "recording": "Recorded seminar: Protecting fragile quantum information",
            "resource": "https://quantum.cloud.ibm.com/learning/en/courses/foundations-of-quantum-error-correction",
        },
    ],
}

QUIZZES: dict[str, list[dict[str, Any]]] = {
    "Entry": [
        {"q": "Which statement best describes a qubit before measurement?", "options": ["It is always secretly 0 or 1", "It can be represented by normalized complex amplitudes", "It stores two classical bits", "It violates probability rules"], "answer": 1, "why": "A pure qubit state is represented by two normalized complex amplitudes."},
        {"q": "What does an ideal Hadamard gate do to |0>?", "options": ["Creates an equal superposition of |0> and |1>", "Always produces |1>", "Measures the qubit", "Copies the qubit"], "answer": 0, "why": "H|0> = (|0> + |1>)/sqrt(2)."},
        {"q": "Why do repeated measurements of the same prepared superposition vary?", "options": ["The computer is broken", "Quantum outcomes are sampled from a probability distribution", "The circuit changes itself", "Qubits are classical random-number generators"], "answer": 1, "why": "Measurement samples outcomes according to the state's probability amplitudes."},
        {"q": "Which gate acts like a classical NOT on basis states?", "options": ["Z", "H", "X", "S"], "answer": 2, "why": "The Pauli-X gate maps |0> to |1> and |1> to |0>."},
    ],
    "Intermediate": [
        {"q": "How many computational basis states does a three-qubit register have?", "options": ["3", "6", "8", "9"], "answer": 2, "why": "An n-qubit system has 2^n basis states, so 2^3 = 8."},
        {"q": "An entangled two-qubit state can always be written as…", "options": ["A product of two single-qubit states", "A classical probability table only", "A joint state that may not factor into subsystem states", "Two copied unknown qubits"], "answer": 2, "why": "Non-factorability is a defining feature of pure-state entanglement."},
        {"q": "Quantum teleportation requires shared entanglement and…", "options": ["No communication", "Two classical bits", "Faster-than-light signaling", "A copy of the unknown state"], "answer": 1, "why": "The protocol requires two classical bits, preserving causality."},
        {"q": "Grover's algorithm is primarily associated with…", "options": ["Unstructured search", "Sorting in constant time", "Copying quantum states", "Perfect error correction"], "answer": 0, "why": "Grover search provides quadratic query improvement for unstructured search."},
    ],
    "Advanced": [
        {"q": "A density matrix is especially useful for representing…", "options": ["Only classical bits", "Mixed states and subsystems", "Only deterministic algorithms", "Copied unknown states"], "answer": 1, "why": "Density matrices represent both pure and mixed quantum states."},
        {"q": "In VQE, the classical optimizer updates…", "options": ["Hardware temperature", "Parameters of an ansatz circuit", "The number of physical laws", "Measurement postulates"], "answer": 1, "why": "VQE iteratively adjusts parameterized circuit values to reduce an energy objective."},
        {"q": "The purpose of a quantum error syndrome is to…", "options": ["Read the logical state directly", "Identify information about errors without revealing the logical data", "Eliminate all physical noise", "Clone the logical qubit"], "answer": 1, "why": "Syndromes reveal error information while preserving encoded logical information."},
        {"q": "A fault-tolerant protocol is designed so that…", "options": ["One fault does not uncontrollably spread into many logical errors", "Every gate is noiseless", "No redundancy is needed", "Classical control is forbidden"], "answer": 0, "why": "Fault tolerance constrains error propagation and supports reliable logical operations."},
    ],
}

KNOWLEDGE_BASE = [
    {"title": "Qubit", "level": "Entry", "text": "A qubit is a two-level quantum information unit. A pure state can be written alpha|0> + beta|1>, where alpha and beta are complex amplitudes and their squared magnitudes sum to one."},
    {"title": "Measurement", "level": "Entry", "text": "Computational-basis measurement returns 0 or 1. The probabilities are the squared magnitudes of the relevant amplitudes. Measurement also changes the post-measurement state."},
    {"title": "Hadamard gate", "level": "Entry", "text": "The Hadamard gate creates equal superpositions from computational-basis states and can also recombine amplitudes, making interference visible."},
    {"title": "No-cloning theorem", "level": "Intermediate", "text": "An arbitrary unknown quantum state cannot be perfectly copied by a universal physical operation. This is consistent with linear quantum evolution."},
    {"title": "Entanglement", "level": "Intermediate", "text": "Entanglement describes joint quantum states whose correlations cannot be explained by assigning independent pure states to each subsystem. It does not enable faster-than-light communication."},
    {"title": "Quantum teleportation", "level": "Intermediate", "text": "Teleportation transfers an unknown quantum state using one shared entangled pair, a joint measurement, and two classical bits. The sender's original state is not retained."},
    {"title": "Grover search", "level": "Intermediate", "text": "Grover's algorithm alternates an oracle phase operation and diffusion-like amplitude amplification. It uses on the order of the square root of N oracle queries for an unstructured search space of size N."},
    {"title": "Density matrix", "level": "Advanced", "text": "A density matrix is positive semidefinite with trace one. It represents pure states, probabilistic mixtures, and reduced states of larger entangled systems."},
    {"title": "VQE", "level": "Advanced", "text": "The variational quantum eigensolver evaluates expectation values from a parameterized ansatz and uses a classical optimizer to search for a low-energy parameter setting."},
    {"title": "Quantum error correction", "level": "Advanced", "text": "Quantum error correction encodes logical information into a larger Hilbert space. Syndrome measurements identify error information while avoiding direct measurement of the logical state."},
    {"title": "QAOA", "level": "Advanced", "text": "QAOA alternates parameterized cost and mixer unitaries, then samples candidate solutions. Performance depends on encoding, depth, parameter optimization, noise, and post-processing."},
]

TEACHING_ARMS = ["Visual analogy", "Worked example", "Guided practice", "Short knowledge check", "Tutor session"]


def init_state() -> None:
    defaults = {
        "page": "Test Home",
        "learner_name": "Quantum Explorer",
        "learner_level": "Entry",
        "weekly_hours": 4.0,
        "math_comfort": 3,
        "learner_goal": "Understand foundations",
        "completed_lessons": [],
        "quiz_history": [],
        "xp": 120,
        "streak": 4,
        "bookings": [],
        "chat_history": [
            {"role": "assistant", "content": "Welcome! I am Professor Qubit. Ask me about qubits, gates, entanglement, algorithms, noise, or error correction."}
        ],
        "bandit_values": {arm: 0.5 for arm in TEACHING_ARMS},
        "bandit_counts": {arm: 0 for arm in TEACHING_ARMS},
        "current_strategy": "Visual analogy",
        "last_lab_success": None,
        "weekly_activity": pd.DataFrame(
            {
                "week": [f"W{i}" for i in range(1, 9)],
                "minutes": [70, 95, 80, 135, 120, 155, 145, 180],
                "quiz_accuracy": [55, 62, 65, 71, 74, 78, 80, 84],
            }
        ),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    


@st.cache_resource
def train_ai_models() -> tuple[RandomForestRegressor, Pipeline, pd.DataFrame, float]:
    rng = np.random.default_rng(42)
    n = 2400
    df = pd.DataFrame(
        {
            "weekly_hours": rng.uniform(0.5, 12.0, n),
            "lessons_completed": rng.integers(0, 13, n),
            "quiz_average": rng.uniform(25, 100, n),
            "streak_days": rng.integers(0, 31, n),
            "positive_feedback": rng.uniform(0.25, 1.0, n),
            "math_comfort": rng.integers(1, 6, n),
            "live_sessions": rng.integers(0, 7, n),
            "difficulty_level": rng.integers(1, 4, n),
        }
    )
    raw_progress = (
        1.8 * df["weekly_hours"]
        + 2.6 * df["lessons_completed"]
        + 0.22 * df["quiz_average"]
        + 0.38 * df["streak_days"]
        + 8.0 * df["positive_feedback"]
        + 2.0 * df["math_comfort"]
        + 2.8 * df["live_sessions"]
        - 3.0 * df["difficulty_level"]
        + rng.normal(0, 5.5, n)
    )
    df["four_week_progress"] = np.clip(raw_progress, 3, 100)

    risk_score = (
        1.2
        - 0.22 * df["weekly_hours"]
        - 0.045 * df["quiz_average"]
        - 0.08 * df["streak_days"]
        - 0.7 * df["positive_feedback"]
        - 0.25 * df["live_sessions"]
        + 0.35 * df["difficulty_level"]
        + rng.normal(0, 0.6, n)
    )
    df["support_risk"] = (risk_score > -2.55).astype(int)

    features = [
        "weekly_hours",
        "lessons_completed",
        "quiz_average",
        "streak_days",
        "positive_feedback",
        "math_comfort",
        "live_sessions",
        "difficulty_level",
    ]
    progress_model = RandomForestRegressor(
        n_estimators=220,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    progress_model.fit(df[features], df["four_week_progress"])

    risk_model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "dnn",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    alpha=0.001,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ]
    )
    risk_model.fit(df[features], df["support_risk"])
    training_accuracy = float(risk_model.score(df[features], df["support_risk"]))
    importance = pd.DataFrame(
        {"feature": features, "importance": progress_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return progress_model, risk_model, importance, training_accuracy


@st.cache_resource
def build_retriever() -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([item["title"] + " " + item["text"] for item in KNOWLEDGE_BASE])
    return vectorizer, matrix


def get_quiz_average() -> float:
    if not st.session_state.quiz_history:
        return 65.0
    return float(np.mean([x["score_pct"] for x in st.session_state.quiz_history]))


def positive_feedback_rate() -> float:
    total = sum(st.session_state.bandit_counts.values())
    if total == 0:
        return 0.65
    weighted = sum(
        st.session_state.bandit_values[a] * st.session_state.bandit_counts[a]
        for a in TEACHING_ARMS
    )
    return float(weighted / total)


def learner_features(level: str, weekly_hours: float, math_comfort: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "weekly_hours": weekly_hours,
                "lessons_completed": len(st.session_state.completed_lessons),
                "quiz_average": get_quiz_average(),
                "streak_days": st.session_state.streak,
                "positive_feedback": positive_feedback_rate(),
                "math_comfort": math_comfort,
                "live_sessions": len(st.session_state.bookings),
                "difficulty_level": LEVEL_TO_NUM[level],
            }
        ]
    )


def choose_teaching_strategy(epsilon: float = 0.16) -> str:
    if random.random() < epsilon:
        return random.choice(TEACHING_ARMS)
    values = st.session_state.bandit_values
    max_value = max(values.values())
    best = [arm for arm, value in values.items() if math.isclose(value, max_value, rel_tol=1e-9)]
    return random.choice(best)


def update_bandit(arm: str, reward: int) -> None:
    st.session_state.bandit_counts[arm] += 1
    n = st.session_state.bandit_counts[arm]
    old = st.session_state.bandit_values[arm]
    st.session_state.bandit_values[arm] = old + (reward - old) / n
    st.session_state.current_strategy = choose_teaching_strategy()


def professor_answer(question: str, learner_level: str, strategy: str) -> str:
    vectorizer, matrix = build_retriever()
    query_vec = vectorizer.transform([question])
    scores = cosine_similarity(query_vec, matrix).ravel()
    top_idx = scores.argsort()[::-1][:3]
    top = [KNOWLEDGE_BASE[i] for i in top_idx]
    confidence = float(scores[top_idx[0]]) if len(top_idx) else 0.0
    core = top[0]["text"]
    related = ", ".join(item["title"] for item in top[1:])

    strategy_text = {
        "Visual analogy": "Analogy: Think of amplitudes as arrows whose directions matter; when arrows combine, they can reinforce or cancel.",
        "Worked example": "Worked example: Start with |0>, apply one gate at a time, and write the state after each operation before predicting measurement outcomes.",
        "Guided practice": "Guided practice: First state the number of qubits, then list basis states, then identify gates, and only then calculate or simulate.",
        "Short knowledge check": "Knowledge check: What result would change if the final measurement were removed? Explain your answer in one sentence.",
        "Tutor session": "Tutor-session suggestion: Bring one circuit and one specific point of confusion to a 25-minute live session for targeted feedback.",
    }[strategy]

    level_note = {
        "Entry": "I will keep the explanation conceptual and use minimal linear algebra.",
        "Intermediate": "I will connect the concept to circuit behavior and basic state-vector reasoning.",
        "Advanced": "I will connect the concept to formal representations, assumptions, and implementation tradeoffs.",
    }[learner_level]

    if confidence < 0.08:
        caveat = "I found only a weak match in the built-in knowledge base, so treat this as a starting point and verify it with the linked course materials or a tutor."
    else:
        caveat = "This answer is grounded in the platform's curated concept notes."

    return (
        f"**Core idea — {top[0]['title']}**\n\n{core}\n\n"
        f"**Adaptive teaching move**\n\n{strategy_text}\n\n"
        f"**For your level**\n\n{level_note}\n\n"
        f"**Related concepts:** {related}.\n\n"
        f"*{caveat}*"
    )


def find_recommended_lesson(level: str) -> dict[str, Any]:
    for lesson in CURRICULUM[level]:
        if lesson["id"] not in st.session_state.completed_lessons:
            return lesson
    for fallback in LEVELS:
        for lesson in CURRICULUM[fallback]:
            if lesson["id"] not in st.session_state.completed_lessons:
                return lesson
    return CURRICULUM["Advanced"][-1]


def make_circuit(experiment: str, theta: float) -> tuple[QuantumCircuit, str]:
    if experiment == "Quantum coin (H gate)":
        qc = QuantumCircuit(1)
        qc.h(0)
        expectation = "0 and 1 should be approximately balanced."
    elif experiment == "Bell entanglement":
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        expectation = "Only 00 and 11 should appear in the ideal simulator."
    elif experiment == "Three-qubit GHZ":
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        expectation = "Only 000 and 111 should appear in the ideal simulator."
    else:
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        expectation = "The probability of 1 is sin²(theta/2)."
    return qc, expectation


def sample_statevector(qc: QuantumCircuit, shots: int, seed: int = 7) -> tuple[dict[str, int], Statevector]:
    state = Statevector.from_instruction(qc)
    state.seed(seed)
    counts = dict(state.sample_counts(shots=shots))
    return counts, state


def score_quiz(level: str, selected: list[str]) -> tuple[int, list[dict[str, Any]]]:
    questions = QUIZZES[level]
    correct = 0
    details = []
    for item, answer_text in zip(questions, selected):
        selected_idx = item["options"].index(answer_text)
        is_correct = selected_idx == item["answer"]
        correct += int(is_correct)
        details.append({"question": item["q"], "correct": is_correct, "explanation": item["why"]})
    return correct, details


def header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


init_state()
progress_model, risk_model, feature_importance, dnn_training_accuracy = train_ai_models()

# These values always exist, including on the full-screen landing page where the
# sidebar is intentionally hidden.
learner_name = st.session_state["learner_name"]
learner_level = st.session_state["learner_level"]
weekly_hours = float(st.session_state["weekly_hours"])
math_comfort = int(st.session_state["math_comfort"])
learner_goal = st.session_state["learner_goal"]

if st.session_state["page"] != "Test Home":
    st.markdown(
        """
        <style>
          .block-container { padding-top:1.2rem; padding-bottom:3rem; max-width:1400px; padding-left:1rem; padding-right:1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("# ⚛️ QubitPath AI")
        st.caption("Adaptive quantum-computing education prototype")
        st.markdown(
            f"""
            <div class="project-credits-sidebar">
              <div><span class="credit-label">Author</span><br><strong>{AUTHOR_NAME}</strong></div>
              <div class="mentor-credit"><span class="credit-label">Mentor</span><br><strong>{MENTOR_NAME}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        learner_name = st.text_input("Learner display name", key="learner_name")
        learner_level = st.selectbox("Current pathway", LEVELS, key="learner_level")
        weekly_hours = st.slider("Planned study hours per week", min_value=1.0, max_value=12.0, step=0.5, key="weekly_hours")
        math_comfort = st.slider("Math comfort", 1, 5, key="math_comfort", help="1 = developing; 5 = very comfortable")
        learner_goal = st.selectbox(
            "Primary goal",
            ["Understand foundations", "Build Qiskit projects", "Prepare for research", "Explore quantum careers"],
            key="learner_goal",
        )
        st.markdown("---")
        st.radio(
            "Navigate",
            ["Home", "Recorded Learning", "Live Tutoring", "Quantum Lab", "Quiz & Games", "AI Professor", "Analytics", "Responsible AI", "Test Home"],
            key="page",
            format_func=lambda item: "Landing" if item == "Test Home" else item,
        )
        page = st.session_state["page"]
        st.markdown("---")
        completion_pct = 100 * len(st.session_state.completed_lessons) / sum(len(v) for v in CURRICULUM.values())
        st.progress(completion_pct / 100, text=f"Overall curriculum: {completion_pct:.0f}%")
        st.caption(f"🔥 {st.session_state.streak}-day streak · ⭐ {st.session_state.xp} XP")
else:
    page = "Test Home"

if st.session_state["page"] == "Test Home":
    page = st.session_state["page"]
    components.html(
        """
        <script>
        (function(){
        function startAnimation() {
            var topDoc = window.parent.document;
            var targetContainer = topDoc.querySelector('.testhome-title');
            
            // Wait for Streamlit to render the container
            if (!targetContainer) {
                requestAnimationFrame(startAnimation);
                return;
            }

            // Check if already injected
            if (targetContainer.querySelector('#blochOverlayRoot')){
              return;
            }

            // Anchor the container so our absolute-positioned canvas stays inside it
            targetContainer.style.position = 'relative';

            var CELL = 100;
            var COLS, ROWS, N, cellW, cellH;
            
            var root = topDoc.createElement('div');
            root.id = 'blochOverlayRoot';
            // Position it absolutely to cover ONLY the target container
            root.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;overflow:hidden;';
            root.innerHTML = '<canvas id="blochCanvas" style="display:block;width:100%;height:100%"></canvas>' +
            //  '<button id="measureBtn2" style="pointer-events:auto;position:absolute;top:16px;left:50%;transform:translateX(-50%);background:#7f77dd;color:#fff;border:none;padding:10px 22px;border-radius:8px;font-size:14px;cursor:pointer;z-index:2">Measure</button>';
            
            // Append to the flexbox, not topDoc.body
            targetContainer.appendChild(root);

            var canvas = topDoc.getElementById('blochCanvas');
            var ctx = canvas.getContext('2d');
            var btn = topDoc.getElementById('measureBtn2');

            var spheres = [];

            function buildSpheres(){
              // Base the width and height on the container, not the window
              canvas.width = targetContainer.clientWidth;
              canvas.height = targetContainer.clientHeight;
              COLS = Math.max(3, Math.round(canvas.width / CELL));
              ROWS = Math.max(2, Math.round(canvas.height / CELL));
              cellW = canvas.width / COLS;
              cellH = canvas.height / ROWS;
              N = COLS * ROWS;
              spheres = [];
              for (var i=0;i<N;i++){
                spheres.push({
                  ax: Math.random()*Math.PI*2, ay: Math.random()*Math.PI*2, az: Math.random()*Math.PI*2,
                  wx: (Math.random()*2-1)*3.2, wy: (Math.random()*2-1)*3.2, wz: (Math.random()*2-1)*3.2,
                  spinning: true, tweenStart: 0, tweening: false, ax0: 0, axT: 0, outcome: null, col: '#5dcaa5'
                });
              }
            }
            buildSpheres();
            
            // Listen for changes to the container size instead of window resizing
            const resizeObserver = new ResizeObserver(() => buildSpheres());
            resizeObserver.observe(targetContainer);

            var ringPts = [];
            var STEPS = 20;
            for (var k=0;k<STEPS;k++){
              var a = k/STEPS*Math.PI*2;
              ringPts.push(Math.cos(a));
              ringPts.push(Math.sin(a));
            }
            function rot(p, ax, ay, az){
              var x=p[0], y=p[1], z=p[2];
              var y1 = y*Math.cos(ax)-z*Math.sin(ax);
              var z1 = y*Math.sin(ax)+z*Math.cos(ax);
              var x1 = x;
              var x2 = x1*Math.cos(ay)+z1*Math.sin(ay);
              var z2 = -x1*Math.sin(ay)+z1*Math.cos(ay);
              var y2 = y1;
              var x3 = x2*Math.cos(az)-y2*Math.sin(az);
              var y3 = x2*Math.sin(az)+y2*Math.cos(az);
              return [x3,y3,z2];
            }
            function rotXY(p, ax, ay){
              var x=p[0], y=p[1], z=p[2];
              var y1 = y*Math.cos(ax)-z*Math.sin(ax);
              var z1 = y*Math.sin(ax)+z*Math.cos(ax);
              var x2 = x*Math.cos(ay)+z1*Math.sin(ay);
              return [x2, y1];
            }

            var last = performance.now();
            function draw(now){
              var dt = Math.min(0.033, (now-last)/1000);
              last = now;
              ctx.clearRect(0,0,canvas.width,canvas.height);
              var r = Math.min(cellW, cellH)*0.28;

              for (var i=0;i<N;i++){
                var s = spheres[i];
                if (!s) continue;
                if (s.spinning){
                  s.ax += s.wx*dt; s.ay += s.wy*dt; s.az += s.wz*dt;
                }
                if (s.tweening){
                  var t = Math.min(1, (now - s.tweenStart)/650);
                  var e = 1 - Math.pow(1-t, 3);
                  s.ax = s.ax0 + (s.axT - s.ax0)*e;
                  if (t>=1) s.tweening = false;
                }
                var col = (i % COLS) + 0.5;
                var row = Math.floor(i / COLS) + 0.5;
                var cx = col*cellW, cy = row*cellH;

                ctx.strokeStyle = 'rgba(127,119,221,0.35)';
                ctx.lineWidth = 1.3;
                for (var pl=0; pl<3; pl++){
                  ctx.beginPath();
                  for (var k=0;k<STEPS;k++){
                    var u = ringPts[k*2], v = ringPts[k*2+1];
                    var p3;
                    if (pl===0) p3=[u,v,0];
                    else if (pl===1) p3=[u,0,v];
                    else p3=[0,u,v];
                    var pr = rot(p3, s.ax, s.ay, s.az);
                    var px = cx + pr[0]*r, py = cy + pr[1]*r;
                    if (k===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
                  }
                  ctx.closePath();
                  ctx.stroke();
                }

                var polePt = rotXY([0,0,1], s.ax, s.ay);
                var tipx = cx + polePt[0]*r*1.15, tipy = cy + polePt[1]*r*1.15;
                ctx.strokeStyle = s.col;
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.85;
                ctx.beginPath();
                ctx.moveTo(cx,cy);
                ctx.lineTo(tipx,tipy);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(tipx,tipy, r*0.13, 0, Math.PI*2);
                ctx.fillStyle = s.col;
                ctx.fill();
                ctx.globalAlpha = 1;
              }
              requestAnimationFrame(draw);
            }
            requestAnimationFrame(draw);

            if (btn) {
            btn.addEventListener('click', function(){
              var isReset = btn.textContent === 'Reset';
              if (!isReset){
                btn.disabled = true;
                spheres.forEach(function(s, idx){
                  setTimeout(function(){
                    s.spinning = false;
                    var outcome = Math.random() < 0.5 ? 0 : 1;
                    s.outcome = outcome;
                    s.col = outcome === 0 ? '#5dcaa5' : '#f0997b';
                    var target = outcome === 0 ? Math.PI/2 : -Math.PI/2;
                    var diff = target - (s.ax % (Math.PI*2));
                    while (diff > Math.PI) diff -= Math.PI*2;
                    while (diff < -Math.PI) diff += Math.PI*2;
                    s.ax0 = s.ax;
                    s.axT = s.ax + diff;
                    s.tweenStart = performance.now();
                    s.tweening = true;
                  }, idx*6);
                });
                setTimeout(function(){ btn.disabled = false; btn.textContent = 'Reset'; }, 1500);
              } else {
                spheres.forEach(function(s){
                  s.spinning = true;
                  s.outcome = null;
                  s.col = '#5dcaa5';
                  s.wx=(Math.random()*2-1)*3.2; s.wy=(Math.random()*2-1)*3.2; s.wz=(Math.random()*2-1)*3.2;
                });
                btn.textContent = 'Measure';
              }
            });
            }
        }
        
        // Start the polling and initialization
        startAnimation();
        
        })();
        </script>
        """,
        height=0,
    )

    st.markdown(
        '<div class="testhome-title"> '
        '<p class = "hometitle"><span style="color: #5dcaa5;">Qubit</span>Path</p> '
        '</div>', unsafe_allow_html=True
        )
    # Scroll animation script
    components.html(
        """
        <script>
        (function() {
            var topDoc = window.parent.document;
            
            function initScrollAnimation() {
                // Target your specific text class
                var textElement = topDoc.querySelector('.hometitle');
                // Target Streamlit's main scrolling container
                var scrollContainer = topDoc.querySelector('section[data-testid="stMain"]');
                
                // Wait for both elements to render
                if (!textElement || !scrollContainer) {
                    requestAnimationFrame(initScrollAnimation);
                    return;
                }

                // Apply styles for smooth scaling
                textElement.style.transition = 'transform 0.1s ease-out';
                textElement.style.transformOrigin = 'center top'; 
                textElement.style.willChange = 'transform';

                // Listen to the scroll event
                scrollContainer.addEventListener('scroll', function() {
                    var scrollY = scrollContainer.scrollTop;
                    
                    // Calculate scale: starts at 1, grows as you scroll down.
                    // The higher the divisor (e.g., 400), the slower it grows.
                    var scale = 1 + (scrollY / 400);
                    
                    // Cap the maximum scale so it doesn't break the layout
                    scale = Math.min(scale, 4.0);

                    // Apply the scale
                    textElement.style.transform = 'scale(' + scale + ')';
                });
            }
            
            initScrollAnimation();
        })();
        </script>
        """,
        height=0,
    )
    #st.markdown("Choose what path you wish to take")
    #learner_name = st.text_input("Learner display name", value="Quantum Explorer")
    #learner_level = st.selectbox("Current pathway", LEVELS)
    #weekly_hours = st.slider("Planned study hours per week", 1.0, 12.0, 4.0, 0.5)
    #math_comfort = st.slider("Math comfort", 1, 5, 3, help="1 = developing; 5 = very comfortable")
    #learner_goal = st.selectbox(
    #            "Primary goal",
    #            ["Understand foundations", "Build Qiskit projects", "Prepare for research", "Explore quantum careers"],
    #        )

    st.markdown(
        '<div class = "testhome-about"> <h4>Learn about</h4> <h1> Quantum Computing</h1></div>'
        '<div class = "testhome-path-options">'
        '<div class = "testhome-path"><h2>Elementary</h2></div>'
        '<div class = "testhome-path"><h2>Intermediate </h2></div>'
        '<div class = "testhome-path"><h2>Advanced</h2></div>'
        '</div>',unsafe_allow_html=True
    )

    st.markdown('<p class="landing-cta-note">Choose a pathway to enter the learning platform.</p>', unsafe_allow_html=True)

    def enter_learning_platform(level: str) -> None:
        st.session_state["learner_level"] = level
        st.session_state["page"] = "Home"

    c_entry, c_intermediate, c_advanced = st.columns(3)
    c_entry.button("Start Elementary", type="primary", width="stretch", key="start_entry", on_click=enter_learning_platform, args=("Entry",))
    c_intermediate.button("Start Intermediate", type="primary", width="stretch", key="start_intermediate", on_click=enter_learning_platform, args=("Intermediate",))
    c_advanced.button("Start Advanced", type="primary", width="stretch", key="start_advanced", on_click=enter_learning_platform, args=("Advanced",))
features = learner_features(learner_level, weekly_hours, math_comfort)
predicted_progress = float(progress_model.predict(features)[0])
risk_probability = float(risk_model.predict_proba(features)[0, 1])
risk_label = "Needs support" if risk_probability >= 0.58 else "On track"
recommended = find_recommended_lesson(learner_level)
if page == "Test Home":
    pass
elif page == "Home":
    header("Learn quantum computing with an adaptive guide", "Structured pathways, hands-on Qiskit labs, live tutoring, and transparent AI recommendations.")
    st.markdown(f"### Welcome, {learner_name} 👋")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current pathway", learner_level)
    c2.metric("4-week progress forecast", f"{predicted_progress:.0f}%")
    c3.metric("Learning status", risk_label, f"{risk_probability:.0%} support probability")
    c4.metric("Experience points", st.session_state.xp)

    st.markdown("### Your next best action")
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown(
            f"""
            <div class="glass-card">
              <span class="pill">{recommended['id']}</span><span class="pill">{learner_level}</span>
              <h3>{recommended['title']}</h3>
              <p>{recommended['summary']}</p>
              <p class="small-muted">Estimated time: {recommended['duration']} minutes · Goal: {learner_goal}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="glass-card">
              <h3>🧠 Adaptive strategy</h3>
              <p><b>{st.session_state.current_strategy}</b></p>
              <p>The contextual bandit updates this recommendation when you rate Professor Qubit's help.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Explore the platform")
    cols = st.columns(4)
    cards = [
        ("🎬", "Recorded learning", "Study leveled modules, lesson notes, activities, and official references."),
        ("🧑‍🏫", "Live tutoring", "Book a prototype Zoom, Teams, or Google Meet session with a tutor."),
        ("🧪", "Quantum lab", "Build and simulate Qiskit circuits with sampled quantum measurements."),
        ("📊", "Analytics", "Review progress, skill coverage, quiz history, and AI forecasts."),
    ]
    for col, (icon, title, text) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="glass-card"><h2>{icon}</h2><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown("### How the AI components work")
    st.info(
        "Machine learning forecasts progress; a multilayer neural network estimates whether additional support may be helpful; "
        "a transparent retrieval tutor answers from curated notes; and a bandit learns which teaching format receives positive feedback."
    )
    st.warning("All predictions are demonstrations trained on synthetic data. A human tutor should review any learner-support decision.")

elif page == "Recorded Learning":
    header("Recorded learning library", "Choose a pathway, open a lesson, complete its activity, and track your progress.")
    tabs = st.tabs(LEVELS)
    for tab, level in zip(tabs, LEVELS):
        with tab:
            st.markdown(f"### {level} pathway")
            lessons = CURRICULUM[level]
            selected_title = st.selectbox(
                f"Select a {level.lower()} lesson",
                [f"{x['id']} — {x['title']}" for x in lessons],
                key=f"lesson_select_{level}",
            )
            lesson = lessons[[f"{x['id']} — {x['title']}" for x in lessons].index(selected_title)]
            a, b = st.columns([1.6, 1])
            with a:
                st.subheader(lesson["title"])
                st.caption(f"{lesson['duration']} minutes · Difficulty {lesson['difficulty']}/3")
                st.markdown("**Learning objectives**")
                for objective in lesson["objectives"]:
                    st.markdown(f"- {objective}")
                st.markdown("**Lesson notes**")
                st.write(lesson["summary"])
                st.markdown("**Hands-on activity**")
                st.write(lesson["activity"])
                st.markdown("**Recording panel**")
                st.info(f"🎬 {lesson['recording']} — prototype slot. Replace the external resource link with your own hosted recording when available.")
                st.link_button("Open official learning resource", lesson["resource"], width="stretch")
            with b:
                completed = lesson["id"] in st.session_state.completed_lessons
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Lesson checkpoint")
                st.write("Explain the key concept in your own words, complete the activity, then mark the lesson complete.")
                if completed:
                    st.success("Completed ✓")
                elif st.button("Mark lesson complete", key=f"complete_{lesson['id']}", width="stretch"):
                    st.session_state.completed_lessons.append(lesson["id"])
                    st.session_state.xp += 40
                    st.success("Lesson completed. +40 XP")
                    st.rerun()
                st.markdown("#### Recommended references")
                st.markdown("- IBM Quantum Learning course catalog")
                st.markdown("- Qiskit documentation and tutorials")
                st.markdown("- *Quantum Computing for Everyone* — Chris Bernhardt")
                st.markdown("- *Introduction to Classical and Quantum Computing* — Thomas Wong")
                st.markdown('</div>', unsafe_allow_html=True)

elif page == "Live Tutoring":
    header("Live tutoring studio", "Schedule guided practice with a tutor through Zoom, Microsoft Teams, or Google Meet.")
    today = date.today()
    schedule = pd.DataFrame(
        [
            {"Tutor": "Dr. Maya Chen", "Focus": "Foundations and Qiskit", "Date": today + timedelta(days=1), "Time": "5:00 PM", "Seats": 3},
            {"Tutor": "Alex Rivera", "Focus": "Entanglement and algorithms", "Date": today + timedelta(days=2), "Time": "7:00 PM", "Seats": 2},
            {"Tutor": "Dr. Samir Patel", "Focus": "VQE, QAOA, and research", "Date": today + timedelta(days=4), "Time": "6:30 PM", "Seats": 1},
            {"Tutor": "Jordan Lee", "Focus": "Quantum math clinic", "Date": today + timedelta(days=5), "Time": "4:30 PM", "Seats": 4},
        ]
    )
    st.dataframe(schedule, hide_index=True, width="stretch")

    st.markdown("### Book a prototype session")
    with st.form("booking_form"):
        c1, c2 = st.columns(2)
        tutor = c1.selectbox("Tutor", schedule["Tutor"].tolist())
        platform = c2.selectbox("Meeting platform", ["Zoom", "Microsoft Teams", "Google Meet"])
        session_date = c1.date_input("Preferred date", value=today + timedelta(days=2), min_value=today)
        session_time = c2.time_input("Preferred time", value=datetime.strptime("18:00", "%H:%M").time())
        topic = st.text_area("Topic or circuit you want help with", placeholder="Example: I understand H and CNOT separately, but I need help interpreting Bell-state measurement results.")
        submitted = st.form_submit_button("Book demo session", width="stretch")
    if submitted:
        booking = {
            "Tutor": tutor,
            "Platform": platform,
            "Date": str(session_date),
            "Time": session_time.strftime("%I:%M %p"),
            "Topic": topic or "General quantum-computing guidance",
            "Status": "Requested",
        }
        st.session_state.bookings.append(booking)
        st.session_state.xp += 15
        st.success("Session request saved in this browser session. +15 XP")

    if st.session_state.bookings:
        st.markdown("### Your session requests")
        st.dataframe(pd.DataFrame(st.session_state.bookings), hide_index=True, width="stretch")
    st.markdown(
        '<div class="warning-box"><b>Production note:</b> Real meeting creation requires OAuth and the provider APIs. '
        'Store credentials in a secret manager, collect the minimum data needed, and obtain consent before recording sessions.</div>',
        unsafe_allow_html=True,
    )

elif page == "Quantum Lab":
    header("Qiskit quantum laboratory", "Create an ideal circuit, inspect its state, and compare your prediction with sampled measurement counts.")
    left, right = st.columns([1, 1.25])
    with left:
        experiment = st.selectbox("Experiment", ["Quantum coin (H gate)", "Bell entanglement", "Three-qubit GHZ", "Custom Ry rotation"])
        theta = st.slider("Rotation angle θ", 0.0, float(2 * np.pi), float(np.pi / 2), 0.05, disabled=experiment != "Custom Ry rotation")
        shots = st.select_slider("Measurement shots", options=[100, 256, 512, 1024, 2048, 4096], value=1024)
        prediction = st.selectbox(
            "Predict the dominant ideal outcome pattern",
            ["Mostly 0", "Mostly 1", "Balanced outcomes", "Only correlated all-zero/all-one outcomes"],
        )
        run_lab = st.button("Run quantum experiment", type="primary", width="stretch")
        st.caption("The simulator is ideal and does not include hardware noise.")

    qc, expectation = make_circuit(experiment, theta)
    with right:
        st.markdown("#### Circuit")
        fig = qc.draw(output="mpl", fold=-1)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    if run_lab:
        counts, state = sample_statevector(qc, shots)
        probability_df = pd.DataFrame({"State": list(counts.keys()), "Counts": list(counts.values())}).sort_values("State")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            bar = px.bar(probability_df, x="State", y="Counts", text="Counts", title="Sampled measurement counts")
            bar.update_layout(yaxis_title="Counts", xaxis_title="Computational basis state")
            st.plotly_chart(bar, width="stretch")
        with c2:
            st.markdown("#### Statevector amplitudes")
            amplitude_rows = []
            n_qubits = qc.num_qubits
            for idx, amp in enumerate(state.data):
                amplitude_rows.append(
                    {
                        "basis": format(idx, f"0{n_qubits}b"),
                        "real": float(np.real(amp)),
                        "imag": float(np.imag(amp)),
                        "probability": float(abs(amp) ** 2),
                    }
                )
            st.dataframe(pd.DataFrame(amplitude_rows), hide_index=True, width="stretch")
            st.info(expectation)

        expected_prediction = {
            "Quantum coin (H gate)": "Balanced outcomes",
            "Bell entanglement": "Only correlated all-zero/all-one outcomes",
            "Three-qubit GHZ": "Only correlated all-zero/all-one outcomes",
            "Custom Ry rotation": "Mostly 0" if theta < np.pi / 2 or theta > 3 * np.pi / 2 else ("Mostly 1" if np.pi / 2 < theta < 3 * np.pi / 2 else "Balanced outcomes"),
        }[experiment]
        success = prediction == expected_prediction
        if success and st.session_state.last_lab_success != (experiment, prediction, round(theta, 2)):
            st.session_state.xp += 25
            st.session_state.last_lab_success = (experiment, prediction, round(theta, 2))
            st.success("Your prediction matched the ideal pattern. +25 XP")
        elif success:
            st.success("Your prediction matched the ideal pattern.")
        else:
            st.warning(f"Compare your prediction with this expectation: {expectation}")

elif page == "Quiz & Games":
    header("Quizzes and mini-games", "Check understanding immediately and turn practice into an active learning loop.")
    quiz_level = st.selectbox("Quiz level", LEVELS, index=LEVELS.index(learner_level))
    with st.form(f"quiz_{quiz_level}"):
        selected_answers = []
        for idx, item in enumerate(QUIZZES[quiz_level], start=1):
            selected_answers.append(st.radio(f"{idx}. {item['q']}", item["options"], key=f"{quiz_level}_{idx}"))
        quiz_submit = st.form_submit_button("Submit quiz", width="stretch")
    if quiz_submit:
        correct, details = score_quiz(quiz_level, selected_answers)
        score_pct = 100 * correct / len(QUIZZES[quiz_level])
        st.session_state.quiz_history.append(
            {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "level": quiz_level, "score_pct": score_pct}
        )
        gained = int(10 * correct)
        st.session_state.xp += gained
        st.metric("Quiz score", f"{correct}/{len(QUIZZES[quiz_level])}", f"+{gained} XP")
        for detail in details:
            if detail["correct"]:
                st.success(f"✓ {detail['question']} — {detail['explanation']}")
            else:
                st.error(f"Review: {detail['question']} — {detail['explanation']}")

    st.markdown("### Gate-matching mini-game")
    game_col1, game_col2 = st.columns(2)
    challenge = game_col1.selectbox("Target transformation", ["Turn |0> into |1>", "Create equal probabilities from |0>", "Add a phase flip to |1>"])
    chosen_gate = game_col2.selectbox("Choose a gate", ["X", "H", "Z"])
    solution = {"Turn |0> into |1>": "X", "Create equal probabilities from |0>": "H", "Add a phase flip to |1>": "Z"}[challenge]
    if st.button("Check gate", width="stretch"):
        if chosen_gate == solution:
            st.success("Correct. The selected gate matches the target transformation.")
        else:
            st.warning(f"Try again. The best match is the {solution} gate.")

elif page == "AI Professor":
    header("Professor Qubit", "A transparent AI tutor that retrieves curated concepts and adapts its teaching format from feedback.")
    st.markdown(
        '<div class="professor"><div class="avatar">⚛️</div><div><h3>Professor Qubit</h3>'
        '<p>Ask a quantum question. I will cite the closest built-in concept, tailor the depth to your pathway, and show the teaching strategy being tested.</p></div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Current adaptive strategy: {st.session_state.current_strategy}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask Professor Qubit a question…")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        answer = professor_answer(prompt, learner_level, st.session_state.current_strategy)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if len(st.session_state.chat_history) > 1:
        st.markdown("#### Was the most recent teaching approach useful?")
        f1, f2, f3 = st.columns([1, 1, 3])
        if f1.button("👍 Helpful", width="stretch"):
            update_bandit(st.session_state.current_strategy, 1)
            st.success("Feedback recorded. The adaptive policy has been updated.")
        if f2.button("👎 Not helpful", width="stretch"):
            update_bandit(st.session_state.current_strategy, 0)
            st.info("Feedback recorded. The platform will explore another teaching format.")
        with f3:
            st.caption("This is an epsilon-greedy contextual-bandit demonstration, not autonomous self-evolution.")

    with st.expander("See the current bandit learning table"):
        bandit_df = pd.DataFrame(
            {
                "Teaching strategy": TEACHING_ARMS,
                "Estimated reward": [st.session_state.bandit_values[a] for a in TEACHING_ARMS],
                "Feedback count": [st.session_state.bandit_counts[a] for a in TEACHING_ARMS],
            }
        )
        st.dataframe(bandit_df, hide_index=True, width="stretch")

elif page == "Analytics":
    header("Learning analytics dashboard", "Review behavior, outcomes, forecasts, and model inputs without treating AI output as a final decision.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lessons completed", len(st.session_state.completed_lessons), f"of {sum(len(v) for v in CURRICULUM.values())}")
    m2.metric("Quiz average", f"{get_quiz_average():.0f}%")
    m3.metric("4-week forecast", f"{predicted_progress:.0f}%")
    m4.metric("Support status", risk_label, f"{risk_probability:.0%}")

    c1, c2 = st.columns(2)
    with c1:
        activity_fig = px.line(
            st.session_state.weekly_activity,
            x="week",
            y=["minutes", "quiz_accuracy"],
            markers=True,
            title="Weekly activity and quiz accuracy",
        )
        activity_fig.update_layout(legend_title_text="Metric")
        st.plotly_chart(activity_fig, width="stretch")
    with c2:
        forecast_weeks = np.arange(0, 5)
        current_completion = 100 * len(st.session_state.completed_lessons) / sum(len(v) for v in CURRICULUM.values())
        projected = np.clip(current_completion + forecast_weeks * predicted_progress / 4.0, 0, 100)
        forecast_df = pd.DataFrame({"Week": forecast_weeks, "Projected curriculum completion": projected})
        forecast_fig = px.area(forecast_df, x="Week", y="Projected curriculum completion", title="Illustrative completion trajectory")
        forecast_fig.update_yaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(forecast_fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        completed_set = set(st.session_state.completed_lessons)
        skill_scores = {
            "Foundations": min(100, 25 + 15 * len(completed_set.intersection({"E1", "E2", "E3", "E4"}))),
            "Circuits": min(100, 20 + 18 * len(completed_set.intersection({"E3", "E4", "I2"}))),
            "Algorithms": min(100, 15 + 22 * len(completed_set.intersection({"I4", "A2", "A3"}))),
            "Quantum math": min(100, 20 + 16 * len(completed_set.intersection({"I1", "A1"}))),
            "Error correction": min(100, 10 + 55 * int("A4" in completed_set)),
        }
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=list(skill_scores.values()), theta=list(skill_scores.keys()), fill="toself", name="Skill score"))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Skill coverage profile")
        st.plotly_chart(radar, width="stretch")
    with c4:
        if st.session_state.quiz_history:
            quiz_df = pd.DataFrame(st.session_state.quiz_history)
            quiz_fig = px.bar(quiz_df, x="date", y="score_pct", color="level", title="Quiz history", range_y=[0, 100])
            st.plotly_chart(quiz_fig, width="stretch")
        else:
            st.info("Complete a quiz to populate the quiz-history chart.")

    st.markdown("### Why the progress model made its forecast")
    importance_fig = px.bar(feature_importance.sort_values("importance"), x="importance", y="feature", orientation="h", title="Random-forest global feature importance on synthetic training data")
    st.plotly_chart(importance_fig, width="stretch")
    st.caption(f"The support-risk neural network uses hidden layers (64, 32, 16). Synthetic-data training accuracy: {dnn_training_accuracy:.1%}. This number does not validate real-world performance.")

    report = {
        "learner": learner_name,
        "pathway": learner_level,
        "goal": learner_goal,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lessons_completed": st.session_state.completed_lessons,
        "quiz_average": get_quiz_average(),
        "xp": st.session_state.xp,
        "four_week_progress_forecast": predicted_progress,
        "support_risk_probability": risk_probability,
        "support_status": risk_label,
        "recommended_next_lesson": recommended["id"],
    }
    r1, r2 = st.columns(2)
    r1.download_button("Download learner report (JSON)", json.dumps(report, indent=2), file_name="qubitpath_learner_report.json", mime="application/json", width="stretch")
    csv_report = pd.DataFrame([report | {"lessons_completed": ",".join(report["lessons_completed"])}]).to_csv(index=False)
    r2.download_button("Download learner report (CSV)", csv_report, file_name="qubitpath_learner_report.csv", mime="text/csv", width="stretch")

else:
    header("Responsible AI and production roadmap", "Build learner trust through transparency, privacy, accessibility, evaluation, and human oversight.")
    st.markdown("### Project credits")
    st.markdown(
        f"""
        <div class="project-credits-page">
          <div class="credit-card">
            <div class="credit-role">Author</div>
            <div class="credit-name">{AUTHOR_NAME}</div>
          </div>
          <div class="credit-card">
            <div class="credit-role">Mentor</div>
            <div class="credit-name">{MENTOR_NAME}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### What this prototype does")
    st.markdown(
        """
        - Uses **synthetic data** to demonstrate progress forecasting and learner-support risk prediction.
        - Uses a **curated retrieval tutor** rather than pretending that a generic language model is always correct.
        - Uses a small **epsilon-greedy bandit** to learn which teaching format receives positive feedback.
        - Keeps all session data in Streamlit's temporary browser/server session state.
        """
    )
    st.markdown("### What a production platform still needs")
    roadmap = pd.DataFrame(
        [
            ["Identity and roles", "Student, tutor, parent/guardian, instructor, and administrator permissions", "High"],
            ["Persistent database", "Encrypted learner profiles, progress events, content, and audit logs", "High"],
            ["Live-session APIs", "OAuth integration for Zoom, Teams, or Google Meet", "High"],
            ["Content management", "Instructor upload, captioning, transcripts, versioning, and moderation", "High"],
            ["Model evaluation", "Real holdout data, subgroup analysis, calibration, drift monitoring, and tutor review", "High"],
            ["Accessibility", "Keyboard navigation, captions, transcripts, readable contrast, and screen-reader testing", "High"],
            ["Security", "Secret management, least privilege, rate limiting, backups, and incident response", "High"],
            ["Optional LLM tutor", "Grounded generation, citation checks, refusal behavior, and age-appropriate controls", "Medium"],
            ["Real quantum hardware", "IBM Quantum credentials, job limits, cost controls, and queue feedback", "Medium"],
        ],
        columns=["Capability", "Production requirement", "Priority"],
    )
    st.dataframe(roadmap, hide_index=True, width="stretch")

    st.markdown("### Guardrails")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown('<div class="glass-card"><h3>🔒 Privacy</h3><p>Minimize data collection, encrypt records, define retention limits, and obtain consent for recordings.</p></div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="glass-card"><h3>🧑‍🏫 Human oversight</h3><p>Never use a risk score as an automatic grade or penalty. Let tutors review context and learner preferences.</p></div>', unsafe_allow_html=True)
    with g3:
        st.markdown('<div class="glass-card"><h3>📐 Evaluation</h3><p>Measure learning gains, calibration, false alerts, accessibility, and performance across relevant learner groups.</p></div>', unsafe_allow_html=True)

    st.markdown("### Official learning references")
    ref1, ref2 = st.columns(2)
    ref1.link_button("IBM Quantum Learning catalog", IBM_COURSES_URL, width="stretch")
    ref2.link_button("Qiskit documentation", QISKIT_DOCS_URL, width="stretch")
