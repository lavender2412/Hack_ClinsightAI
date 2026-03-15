# ClinsightAI

> A multi-task clinical intelligence pipeline that transforms raw hospital patient reviews into structured insights, theme-level diagnostics, and a priority-ranked operational action roadmap.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hackclinsightai.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![ML](https://img.shields.io/badge/ML-LDA%20%7C%20Ridge%20Regression%20%7C%20TF--IDF-green)

---

## What It Does

ClinsightAI processes hospital patient reviews and outputs four layers of analysis:

| Output | Description |
|---|---|
| **Review-level** | Per-review dominant theme, predicted rating, contribution scores, and recommended action |
| **Theme-level** | Rating impact per theme with bootstrap confidence intervals and risk scores |
| **Systemic issues** | Themes classified as Isolated, Recurring, or Systemic based on frequency and impact |
| **Action roadmap** | Priority-ranked improvement initiatives with effort estimates and KPI targets per theme |

---

## How It Works

### 1. Topic Modeling — LDA
Five clinical themes are extracted from patient review text using **Latent Dirichlet Allocation**:
- Wait Time & Operational Efficiency
- Overall Service & Facility Cleanliness
- Emergency Service Failures
- Clinical Care & Treatment Quality
- Consultation & Positive Patient Experience

### 2. Rating Prediction — Ridge Regression
A **Ridge Regression** model is trained on combined **TF-IDF** unigram/bigram features and LDA topic probabilities to predict patient ratings, improving R² from **0.046 to 0.302**.

### 3. Per-Review Explainability
Each review receives:
- Dominant theme label
- Local theme contribution scores
- Most positive and most negative driving theme
- Predicted rating and residual

### 4. Systemic Issue Classification
Themes are classified by frequency and impact:
- **Isolated** — appears in < 5% of reviews
- **Recurring** — moderate frequency, moderate impact
- **Systemic** — high frequency and high impact coefficient

### 5. Action Roadmap
Each theme is assigned a priority rank, effort estimate (Quick Win or High Effort), KPI targets, and operational recommendations — exported as a structured CSV and surfaced in the Streamlit dashboard.

---

## Pipeline Architecture
```
hospital.csv
     │
     ▼
Text Preprocessing (NLTK, lemmatization, stopword removal)
     │
     ▼
LDA Topic Modeling (CountVectorizer → 5 themes)
     │
     ├──► Theme probabilities per review
     │
TF-IDF (unigram + bigram) + Topic features → Ridge Regression
     │
     ├──► Predicted ratings
     ├──► Per-review contribution scores
     ├──► Bootstrap coefficient CIs
     │
     ▼
Systemic Issue Classification + Risk Scoring
     │
     ▼
Priority-Ranked Action Roadmap
     │
     ▼
Streamlit Dashboard + CSV/JSON Exports
```

---

## Output Files

| File | Contents |
|---|---|
| `review_level_outputs.csv` | Per-review themes, predictions, contributions, actions |
| `theme_level_outputs.csv` | Theme-level impact, risk scores, bootstrap CIs |
| `task3_recurring_systemic.csv` | Issue classification by frequency and impact |
| `task4_action_roadmap.csv` | Priority-ranked roadmap with KPIs and effort estimates |
| `clinsight_output.json` | Structured JSON of all outputs |

---

## Getting Started
```bash
git clone https://github.com/lavender2412/Code_Blooded_ClinsightAI.git
cd Code_Blooded_ClinsightAI
pip install -r requirements.txt
```

**Run the pipeline:**
```bash
python run_pipeline.py
```

**Launch the dashboard:**
```bash
streamlit run app.py
```

---

## Project Structure
```
Code_Blooded_ClinsightAI/
├── pipeline/                  # Core analysis modules
├── app.py                     # Streamlit dashboard
├── run_pipeline.py            # Pipeline entry point
├── hospital.csv               # Input data
├── review_level_outputs.csv   # Output: per-review analysis
├── theme_level_outputs.csv    # Output: theme diagnostics
├── task3_recurring_systemic.csv  # Output: systemic classification
├── task4_action_roadmap.csv   # Output: action roadmap
├── clinsight_output.json      # Output: structured JSON
└── requirements.txt
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy, NLTK |
| ML & NLP | Scikit-learn, LDA, TF-IDF, Ridge Regression, SciPy |
| Visualization | Streamlit |
| Output Formats | CSV, JSON |
