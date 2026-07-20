# Architecture

## Prototype flow

```text
Learner profile and activity
        |
        v
Curriculum + quizzes + quantum lab
        |
        +------------------------------+
        |                              |
        v                              v
Progress event history          Tutor question
        |                              |
        v                              v
Random forest forecast     TF-IDF retrieval tutor
MLP support signal                   |
        |                              v
        +------> Dashboard and recommended next action
                         |
                         v
                 Like/dislike feedback
                         |
                         v
             Epsilon-greedy teaching policy
```

## Application layers

1. **Presentation layer:** Streamlit navigation, learner controls, lesson pages, circuit lab, tutor, and dashboard.
2. **Learning-content layer:** Leveled curriculum, objectives, activities, quizzes, references, and mini-games.
3. **Analytics layer:** Synthetic event generation, progress forecasting, learner-support classification, and visualizations.
4. **Adaptive layer:** Feedback-based teaching-format selection using an epsilon-greedy multi-armed bandit prototype.
5. **Quantum layer:** Qiskit `QuantumCircuit` definitions and ideal `Statevector` sampling.
6. **Session layer:** Streamlit session state for prototype learner progress, bookings, feedback, and reports.

## Production target architecture

Replace in-memory session state with an authenticated application and encrypted database. Separate model training from online inference, add an auditable feature store, place meeting-provider and LLM credentials in managed secrets, and log model versions and tutor citations. Use role-based access for students, tutors, administrators, and guardians where applicable.
