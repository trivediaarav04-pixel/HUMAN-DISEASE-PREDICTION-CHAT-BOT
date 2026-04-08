# 🩺 Human Disease Prediction Chat Bot

> An AI-powered chatbot that predicts possible diseases based on symptoms you describe. Simply type your symptoms in natural language, and the bot analyzes them to suggest likely conditions, severity levels, and when to seek medical help — making early health awareness faster, smarter, and more accessible for everyone.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [App Pages](#app-pages)
- [Dataset](#dataset)
- [Model](#model)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Overview

The **Human Disease Prediction Chat Bot** is a machine learning web application built with **Streamlit**. It takes up to 17 symptoms as input — either typed in natural language or selected from a dropdown — and uses a trained **Random Forest Classifier** to predict the most likely disease, along with confidence scores for the top predictions.

---

## Features

- **Symptom input via typing or dropdowns** — supports up to 17 symptoms at once
- **Natural language symptom entry** — comma-separated typed symptoms are auto-normalized
- **Top-3 disease predictions** displayed as metric cards with confidence percentages
- **Bar chart** of the top 5 predicted diseases by probability
- **"Why this prediction?"** panel showing the top contributing symptoms
- **Add Data page** — contribute new symptom-disease rows to expand the dataset
- **Train Model page** — retrain the model in-browser with a progress bar and accuracy display
- **Feature importance table** after training
- **Download buttons** for the dataset and trained model
- **Auto-add new symptoms** to the severity database with a configurable default weight
- Fully self-contained — model and datasets are bundled in the repository

---

## Project Structure

```
HUMAN-DISEASE-PREDICTION-CHAT-BOT/
│
├── app.py                        # Main Streamlit application
├── model_RFC.joblib              # Pre-trained Random Forest model
├── Disease_Prediction.ipynb      # Jupyter notebook for exploration & training
│
├── Dataset/
│   ├── dataset.csv               # Training dataset (disease + 17 symptom columns)
│   └── Symptom-severity.csv      # Symptom-to-severity weight mapping
│
└── README.md
```

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core language |
| Streamlit | Web UI framework |
| scikit-learn | Random Forest Classifier |
| pandas | Data manipulation |
| numpy | Numerical operations |
| joblib | Model serialization |
| streamlit-option-menu | Sidebar navigation |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/trivediaarav04-pixel/HUMAN-DISEASE-PREDICTION-CHAT-BOT.git
cd HUMAN-DISEASE-PREDICTION-CHAT-BOT
```

### 2. Install dependencies

```bash
pip install streamlit scikit-learn pandas numpy joblib streamlit-option-menu
```

### 3. Run the app

```bash
streamlit run app.py
```

> The app will open automatically at `http://localhost:8501`

---

## How It Works

1. **Symptom encoding** — each symptom is looked up in `Symptom-severity.csv` and replaced with its severity weight (an integer score reflecting clinical severity).
2. **Feature vector** — a fixed-length vector of 17 weights is built from your input (missing slots are zero-padded).
3. **Prediction** — the pre-trained Random Forest model predicts the disease class and outputs class probabilities.
4. **Explainability** — the top contributing symptoms are computed by multiplying each symptom's severity weight by the model's global feature importance score.

---

## App Pages

### Prediction
- Enter symptoms by typing (comma-separated) or selecting from dropdowns
- Click **Make Prediction** to get results
- View the predicted disease, top-3 confidence cards, probability bar chart, and contributing symptoms
- Download your symptom input as a CSV

### Add Data
- Contribute a new disease label + symptom combination to the training dataset
- Unknown typed symptoms are auto-added to the severity database
- Download or replace the dataset CSV

### Train Model
- Retrain the Random Forest Classifier on the current dataset
- View training accuracy and feature importance
- Download the newly trained `model_RFC.joblib`

---

## Dataset

The `Dataset/` folder contains two files:

| File | Description |
|------|-------------|
| `dataset.csv` | Each row has a `Disease` label followed by up to 17 symptom columns |
| `Symptom-severity.csv` | Maps each symptom name to a numeric severity weight used for encoding |

You can expand the dataset directly from the **Add Data** page within the app.

---

## Model

- **Algorithm:** Random Forest Classifier (`n_estimators=200`, `random_state=42`)
- **Train/test split:** 80/20 with stratification
- **Serialization:** saved as `model_RFC.joblib` using `joblib`
- A pre-trained model is included in the repository so you can predict immediately without retraining.

---

## Disclaimer

> **This application is for educational and informational purposes only.**
> It is not a substitute for professional medical advice, diagnosis, or treatment.
> Always consult a qualified healthcare provider for any medical concerns.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*Made with ❤️ by [Aarav Trivedi](https://github.com/trivediaarav04-pixel)*
