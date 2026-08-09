
**PRD = Product Requirements Document**

# For AstraMed, I want the PRD to look like this:

1. Project Overview

2. Problem Statement

3. Vision

4. Objectives

5. Target Users

6. Scope

7. Functional Requirements

8. Non-Functional Requirements

9. Success Metrics

10. Dataset

11. Technology Stack

12. Constraints

13. Risks

14. Future Roadmap

---

# Section 1 — Project Overview

This is the elevator pitch.

Example:

> AstraMed v1.0 is an AI-powered medical imaging platform that detects pneumonia from chest X-rays using deep learning while providing visual explanations of its predictions. The goal is to create an explainable and production-ready system that can later evolve into a multimodal clinical decision support platform.

---

# Section 2 — Problem Statement

What problem are we solving?

Example:

> Pneumonia remains one of the leading causes of respiratory illness. Manual interpretation of chest X-rays requires experienced radiologists and may take time in resource-constrained settings. Existing AI solutions often lack transparency. AstraMed aims to assist healthcare professionals by providing accurate pneumonia detection along with explainable visual evidence.

---

# Section 3 — Vision

Think 2–3 years ahead.

Example:

> Build a modular healthcare AI platform capable of analyzing multiple medical modalities such as X-rays, CT scans, ECGs, blood reports, and clinical notes to assist healthcare professionals in diagnosis and decision making.

Notice:

We're not saying v1 will do this.

We're saying this is the destination.

---

# Section 4 — Objectives

Keep it measurable.

Example:

### Primary Objectives

- Detect pneumonia from chest X-rays.
- Generate confidence scores.
- Produce Grad-CAM explanations.
- Build a production-ready API.

### Secondary Objectives

- Learn end-to-end ML Engineering.
- Follow industry best practices.
- Build a scalable architecture.

---

# Section 5 — Target Users

Who will use AstraMed?

Version 1

- Medical students
- Researchers
- AI Engineers
- Developers

Later

- Hospitals
- Clinics
- Doctors

---

# Section 6 — Scope

This is VERY important.

Write two sections.

### In Scope

✅ Chest X-ray classification

✅ Pneumonia detection

✅ Explainability

✅ REST API

---

### Out of Scope

❌ Medical diagnosis

❌ Prescription generation

❌ Doctor replacement

❌ CT scan analysis

❌ ECG analysis

This prevents "feature creep."

---

# Section 7 — Functional Requirements

What should the software do?

Example

The system shall

- Upload chest X-ray images.
- Predict pneumonia probability.
- Display confidence score.
- Generate Grad-CAM visualization.
- Return predictions through API.

---

# Section 8 — Non-Functional Requirements

These are quality requirements.

Example

- Response time < 3 seconds
- Modular architecture
- Docker support
- Scalable codebase
- Reproducible experiments

---

# Section 9 — Success Metrics

Not just accuracy.

Example

Technical

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Engineering

- Dockerized
- API deployed
- CI/CD pipeline
- Documentation complete

Learning

- Understand CNNs
- Learn MLOps
- Deploy successfully

---

# Section 10 — Dataset

Mention

Primary

- RSNA Pneumonia Detection Challenge

Future

- NIH ChestXray14
- African Multimodal Dataset

---

# Section 11 — Technology Stack

- Initial

Python

PyTorch

OpenCV

Pandas

NumPy

Matplotlib

FastAPI

Docker

Git

- Future

MLflow

DVC

GitHub Actions

AWS

---

# Section 12 — Constraints

Example

- Limited GPU
- Student budget
- Public datasets only
- Educational use

---

# Section 13 — Risks

Example

- Class imbalance
- Overfitting
- Dataset bias
- Domain shift
- Explainability limitations

Every engineering project has risks.

---

# Section 14 — Future Roadmap

v1.0

Pneumonia Detection

↓

v2.0

Vision + Symptoms

↓

v3.0

Clinical Decision Support

↓

v4.0

Medical Report Generation

↓

v5.0

Complete Multimodal Healthcare AI Platform
