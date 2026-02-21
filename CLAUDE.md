# CLAUDE.md — AI Assistant Guide for Hack_ClinsightAI

This file provides context for AI assistants (Claude, Copilot, etc.) working in this repository.

---

## Project Overview

**ClinSight AI** is a healthcare review analytics system that ingests raw patient feedback, runs an ML pipeline (text preprocessing → LDA topic modeling → regression impact analysis → risk scoring), and exposes results through an interactive Streamlit dashboard.

The system answers three practical questions for hospital administrators:
1. What operational themes appear in patient reviews?
2. Which themes most strongly impact ratings (positive or negative)?
3. Which themes represent systemic risks requiring urgent action?

---

## Repository Structure

```
Hack_ClinsightAI/
├── app.py                    # Streamlit dashboard (7-tab UI)
├── run_pipeline.py           # End-to-end ML pipeline executor
├── requirements.txt          # Python dependencies
├── README.md                 # Minimal project readme
├── hospital.csv              # Input data: 996 patient reviews
├── clinsight_output.json     # Structured JSON summary (pipeline output)
├── review_level_outputs.csv  # Per-review predictions and analysis
├── theme_level_outputs.csv   # Theme-level aggregated metrics
├── task3_recurring_systemic.csv  # Recurring/systemic issue table
├── task4_action_roadmap.csv      # Business action roadmap
└── pipeline/                 # Python package — ML pipeline modules
    ├── __init__.py
    ├── preprocess.py         # Text cleaning and lemmatization
    ├── topic_model.py        # LDA topic modeling
    ├── impact.py             # Regression impact analysis
    ├── risk.py               # Risk scoring and classification
    ├── json_output.py        # JSON output generation
    └── recommendations.py    # Placeholder (not yet implemented)
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Dashboard | Streamlit |
| Data manipulation | pandas, numpy |
| ML / NLP | scikit-learn, nltk |
| Visualizations | plotly, matplotlib, seaborn |
| Network analysis | networkx (imported, minimal use) |
| Optional LLM | OpenAI API (gated behind `USE_GPT=False`) |

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

NLTK corpora are downloaded at runtime inside `preprocess.py`:
```python
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

### 2. Run the ML pipeline

```bash
python run_pipeline.py
```

This reads `hospital.csv`, runs preprocessing → LDA → regression → risk scoring, and writes all output CSVs and `clinsight_output.json`.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard reads the pre-computed CSVs. Always run the pipeline first if the data or model logic has changed.

---

## Data Flow

```
hospital.csv  (raw patient reviews)
    │
    ▼
preprocess.py  →  clean text, lemmatize, remove stopwords
    │
    ▼
topic_model.py →  CountVectorizer + LDA (5 topics)
                  produces topic probability vectors per review
    │
    ▼
impact.py      →  TF-IDF + topic probs → Ridge regression
                  bootstrap CIs (200 iter), 5-fold cross-validation
    │
    ▼
risk.py        →  severity scores, risk scores, issue classification
                  (Isolated / Recurring / Systemic)
    │
    ▼
json_output.py →  clinsight_output.json  +  *.csv files
    │
    ▼
app.py         →  Streamlit dashboard
```

---

## Key Configuration Constants

All knobs live at the top of `run_pipeline.py`:

| Constant | Default | Purpose |
|---|---|---|
| `DATA_PATH` | `"hospital.csv"` | Input CSV file path |
| `TEXT_COL` | `"Feedback"` | Column name for review text |
| `RATING_COL` | `"Ratings"` | Column name for numeric ratings |
| `N_TOPICS` | `5` | Number of LDA topics |
| `PROB_THRESHOLD` | `0.30` | Minimum topic probability to assign a theme |
| `BOOTSTRAP_B` | `200` | Bootstrap iterations for confidence intervals |
| `USE_GPT` | `False` | Toggle OpenAI GPT enrichment |
| `OPENAI_MODEL` | `"gpt-4o-mini"` | GPT model to use when `USE_GPT=True` |

When `USE_GPT=True`, set the environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

---

## LDA Topic Labels (Hardcoded)

Topic labels are defined in `topic_model.py` and must be updated manually if `N_TOPICS` changes:

| Topic # | Label |
|---|---|
| 1 | Clinical Care & Treatment Quality |
| 2 | Overall Service & Facility Cleanliness |
| 3 | Emergency Service Failures |
| 4 | Consultation & Positive Patient Experience |
| 5 | Wait Time & Operational Efficiency |

**Important:** If you change `N_TOPICS`, you must also update the labels list in `topic_model.py` and re-run the full pipeline.

---

## Pipeline Module API

### `pipeline/preprocess.py`

```python
preprocess(text: str) -> str
    # Lowercases, strips special chars, lemmatizes, removes stopwords

clean_dataframe(df, text_col, rating_col) -> pd.DataFrame
    # Applies preprocess() to the text column; drops rows with missing ratings
```

### `pipeline/topic_model.py`

```python
run_lda(df, text_col, n_topics) -> dict
    # Fits LDA, returns topic probabilities, labels, similarity matrix
```

### `pipeline/impact.py`

```python
run_impact(df, topic_probs, ratings) -> pd.DataFrame
    # Fits Ridge regression, runs bootstrap, returns impact_df with:
    # coefficient, CI_lower, CI_upper, stability_score per theme
```

### `pipeline/risk.py`

```python
run_risk(theme_df, impact_df) -> pd.DataFrame
    # Adds severity_score, risk_score, issue_class
    # Classification rules:
    #   Isolated  : frequency < 0.05
    #   Systemic  : frequency >= 0.20 AND |impact| >= median(|impacts|)
    #   Recurring : everything else
```

### `pipeline/json_output.py`

```python
generate_json_output(theme_df, review_df) -> dict
save_json(data, path) -> None
```

---

## Dashboard Structure (`app.py`)

The Streamlit app has 7 tabs and a sidebar with global filters.

**Global sidebar filters:**
- Theme selection (All or specific theme)
- Rating range (1–5)
- Topic probability threshold (0.05–0.80)
- Risk classification thresholds
- Impact filter (Negative / Positive / All)

**Tabs:**

| Tab | Content |
|---|---|
| Operational Themes | Frequency bar and pie charts |
| Rating Impact | Lollipop chart with bootstrap confidence intervals |
| Risk & Systemic Issues | Heatmap + classification distribution |
| Recurring (Interactive) | Live threshold controls with drilldown |
| Action Roadmap | Priority-ranked business actions with KPIs |
| Review Explorer | Row-level review inspection |
| JSON Output | Machine-readable output + download button |

**Performance note:** Data loading functions are decorated with `@st.cache_data`. Avoid mutating cached DataFrames directly.

---

## Output Files

| File | Description |
|---|---|
| `review_level_outputs.csv` | One row per review: clean text, topic probs, contributions, predicted rating, residual, theme, recommendations, KPI urgency |
| `theme_level_outputs.csv` | One row per theme: impact coeff, CI, stability, risk score, severity, issue class |
| `task3_recurring_systemic.csv` | Subset of themes classified as Recurring or Systemic |
| `task4_action_roadmap.csv` | Prioritized action items with effort and KPI tags |
| `clinsight_output.json` | Structured summary: clinic_summary, theme_analysis, improvement_roadmap |

---

## Code Conventions

- **Naming:** `snake_case` for functions and variables; `UPPER_CASE` for module-level constants.
- **Dataframes:** All intermediate results passed as pandas DataFrames between pipeline stages.
- **Sparse matrices:** TF-IDF features use `scipy.sparse` and are combined with dense topic probability arrays via `scipy.sparse.hstack`.
- **Section separators:** Long files use `# ════════════════════════════════════════` as visual separators.
- **Progress logging:** `print()` statements are used throughout the pipeline for progress tracking (no logging framework).
- **No global state in modules:** Each pipeline function accepts inputs and returns outputs; side effects are limited to file I/O at the end of `run_pipeline.py`.

---

## Known Issues & Gaps

- **No tests:** There are no unit or integration tests. When modifying pipeline modules, manually verify CSV outputs for shape and value sanity.
- **Hardcoded paths:** `DATA_PATH`, output filenames, and topic labels are hardcoded. Keep input/output files in the repository root.
- **SSL override:** `run_pipeline.py` includes an SSL context override (`ssl._create_default_https_context = ssl._create_unverified_context`) for NLTK downloads. This is a known security trade-off for development environments.
- **`recommendations.py` is empty:** The module file exists but contains no implementation. Do not import from it.
- **No input validation:** The pipeline assumes `hospital.csv` contains `Feedback` and `Ratings` columns with no missing structure validation.
- **GPT path untested:** `USE_GPT=True` code path exists but is not the primary development focus.

---

## Development Workflow

1. Make changes to pipeline modules (`pipeline/*.py`) or `run_pipeline.py`.
2. Run `python run_pipeline.py` to regenerate all output files.
3. Run `streamlit run app.py` to verify dashboard correctness.
4. Commit both code changes and updated output CSVs/JSON together so the repo remains self-contained.

**Branch convention:** Feature branches follow `claude/<description>-<session-id>` for AI-assisted work.

---

## Adding a New Pipeline Stage

1. Create `pipeline/new_stage.py` with a single public function following the pattern:
   ```python
   def run_new_stage(input_df: pd.DataFrame, ...) -> pd.DataFrame:
       ...
       return output_df
   ```
2. Import and call it in `run_pipeline.py` after the appropriate existing stage.
3. Add any new output columns to the CSV export logic at the bottom of `run_pipeline.py`.
4. If the new stage produces a new output file, document it in this file under **Output Files**.

---

## Extending the Dashboard

- Add new tabs by appending to the `tabs = st.tabs([...])` list in `app.py`.
- Use `@st.cache_data` on any new data loading or computation functions.
- All Plotly figures should use `fig.update_layout(template="plotly_white")` for visual consistency.
- Interactive controls that affect multiple tabs should be placed in the sidebar, not inside a tab.
