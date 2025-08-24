# Autiscan-Pro


# Autism Spectrum Assessment Tool

A privacy‑first web app that screens for autism spectrum traits using two inputs:
- AQ‑20 questionnaire responses (converted from 0–3 Likert scale to binary 0/1)
- Optional free‑text responses that are analyzed for theme alignment

The system has a simple Flask frontend (UI) and a Flask backend (API with ML and text analysis). No user data is stored.

## What this project does
- Collects AQ responses and basic demographics in the browser
- Sends them to a backend API that validates, preprocesses, analyzes text, and predicts
- Returns a result (ASD / No ASD) with confidence, risk level, recommendations, and text insights
- Renders a results page; data is processed in memory only and discarded after use

---

## Repository structure (what each file/folder is for)

- `frontend/`
  - `app.py` — The UI server. Renders pages, collects form data, calls the backend API, and shows results.
  - `templates/`
    - `landing.html` — Welcome/entry page.
    - `index.html` — The assessment form (AQ questions, demographics, and optional text prompts).
    - `result.html` — Results view. Recommendations and text analysis are dynamic; summary cards currently have placeholder values unless wired to backend variables.
    - `privacy.html` — Privacy information page.
  - `static/`
    - `style.css` — Styles for the UI.
    - `script.js` — Client‑side helpers: progress tracking, character counts, API status indicator, submit UX.

- `backend/`
  - `app.py` — The REST API. Endpoints for health, questions, validation, text analysis, prediction, and model info.
  - `requirements.txt` — Backend Python dependencies.
  - `models/`
    - `autism_classifier.py` — Loads/trains the RandomForest model; handles encoders, scaler, preprocessing, predict; tracks training metrics.
  - `utils/`
    - `data_processor.py` — Converts raw form input into ML‑ready features (AQ mapping, demographics normalization) and produces recommendations/risk labels.
    - `validator.py` — Validates the incoming payload (AQ responses, age/gender/etc.).
    - `text_analyzer.py` — TF‑IDF + cosine similarity against reference themes; returns category alignment and insights.
  - `ml_models/` — Saved artifacts used at inference: `autism_model.pkl`, `scaler.pkl`, `label_encoders.pkl` (and optionally `training_info.pkl`).

- `data/` (expected)
  - `expanded.csv` — Training data used by the classifier (AQ‑20 + demographics + label). Not committed here.
  - `synthetic_autism_text_data.csv` — Reference text dataset used by the text analyzer.

- Top‑level helpers
  - `check_model_accuracy.py` — Reads `training_info.pkl` and prints train/test/CV accuracy and top feature importances.

---

## How the pieces work together

1) Frontend flow
- The user opens `landing.html` → starts the assessment at `/index`.
- `index.html` shows AQ questions and demographic fields; optional text boxes add qualitative context.
- On submit, the form posts to the frontend route `/result`.
- The frontend immediately pings backend `/api/health` (for visibility), then forwards the form data to backend `/api/predict`.

2) Backend flow
- `validator.py` checks the payload (AQ present and typed correctly; reasonable age; basic sanity checks).
- `data_processor.py` converts AQ from 0–3 → binary 0/1 per question (threshold 2+ → 1) and normalizes demographics to stable strings. It also totals the AQ score (0–20).
- `text_analyzer.py` cleans optional free text, vectorizes with TF‑IDF (uni/bi‑grams), and compares to reference category texts using cosine similarity. It returns a top category and confidence plus readable insights.
- `autism_classifier.py` preprocesses features to match the model’s training format and uses the RandomForest to predict label and probabilities.
- `data_processor.py` combines the model output and AQ score to derive a risk label and tailored recommendations.
- The API returns a JSON with: prediction, confidence, AQ total, risk, explanation, recommendations, optional text insights, and model info.

3) Results rendering
- The frontend receives the API JSON and renders `result.html`.
- Today, the recommendations list and text insights are dynamic.
- Note: The summary cards (prediction text, confidence ring, AQ score dial, risk chip) are currently placeholders in `result.html`. To make them live, bind `results.prediction`, `results.confidence`, `results.aq_score`, and `results.risk_class` into the template or via a small script hook.

---

## Backend API endpoints

- `GET /api/health`
  - Returns status, UTC timestamp, and whether the ML model is loaded.

- `GET /api/questions`
  - Returns the AQ questions (falls back if JSON not present).

- `POST /api/validate`
  - Validates a full payload; returns `valid`, `errors`, and `warnings`.

- `POST /api/analyze-text`
  - Accepts `{ text_responses: { id: text } }` and returns per‑text analysis and overall insights.

- `POST /api/predict`
  - Main endpoint. Accepts AQ+demographics (+optional text). Returns prediction, confidence, AQ total, risk, explanation, recommendations, text analysis (if any), and model info.

- `GET /api/model-info`
  - Returns model type, feature count, training metrics snapshot (if available), and flags for privacy.

---

## Machine learning details

- Model
  - RandomForestClassifier with `class_weight='balanced'` to handle label imbalance.
  - Features: 20 AQ binary items + demographics (age, gender, ethnicity, jaundice, autism family history, country, used_app_before, relation).
  - Preprocessing: LabelEncoder for categoricals; StandardScaler for numerics (`age` and AQ binary features).

- Training
  - See `AutismClassifier.train_model()`: split 80/20, train RF, evaluate test accuracy, and 5‑fold CV; save model/encoders/scaler; store `training_info.pkl` with metrics and feature importances.
  - Expects `data/expanded.csv` with columns: `A1_Score`..`A20_Score`, demographics, and target `Class/ASD`.

- Accuracy
  - Run `check_model_accuracy.py` to print training/test/CV accuracy and top features if `training_info.pkl` exists.

- Known caveats
  - Feature alignment: scaling requires the feature vector at inference to match the scaler’s fit time. If columns drift, you might see errors like “X has N features, but StandardScaler is expecting M.” The fix is to lock and persist the training feature order and enforce it during preprocessing, or retrain pipeline artifacts together and load them as a unit.

---

## Text analysis details

- Reference dataset: `data/synthetic_autism_text_data.csv` with labeled categories (e.g., social, sensory, routine, interests, emotional).
- Vectorization: TF‑IDF with English stop words and n‑grams (1,2), limited vocabulary size for speed.
- Scoring: For each category, compute cosine similarity of the user vector to all samples in that category and keep the max as the category score.
- Output: top category, a percentage‑style confidence, plus readable insights (length/complexity notes and theme‑specific observations).
- Purpose: Adds context to the quantitative AQ score; not a diagnostic tool.

---

## How to run (local)

Optional commands for Windows PowerShell. Use two terminals: one for backend, one for frontend.

```powershell
# (Optional) Create a virtual environment at project root
py -m venv .venv ; .\.venv\Scripts\Activate.ps1

# Install backend deps
pip install -r backend/requirements.txt

# Start backend API (port 5001)
python backend/app.py
```

In a second terminal:

```powershell
# Activate the same venv if you created it
.\.venv\Scripts\Activate.ps1

# Start the frontend server (defaults to port 5000)
python frontend/app.py
```

Then open http://localhost:5000 in your browser.

---

## Retraining and checking accuracy

- Ensure `data/expanded.csv` is present with the expected schema.
- Start the backend once; it will try to load existing artifacts from `backend/ml_models/`. If not found, you can programmatically train via the classifier.
- Use the helper to view metrics:
  - `python check_model_accuracy.py`

Notes:
- When training, run from the project root or `backend/` so relative save paths match `backend/ml_models/`.
- Keep `autism_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, and `training_info.pkl` together.

---

## Privacy

- No inputs are stored in files or databases.
- The only persisted artifacts are model files generated during training.
- Logs avoid personal data and are focused on system health and debugging.

---

## Troubleshooting

- Backend health is red in the UI
  - Check `http://localhost:5001/api/health` directly. If it fails, ensure the backend is running and the port matches the frontend’s `BACKEND_API_URL`.

- Feature count error in scaler
  - Indicates a mismatch between training and inference feature order/count. Align the preprocessing or retrain the whole pipeline.

- Text analysis unavailable
  - Ensure `data/synthetic_autism_text_data.csv` exists and has `text` and `category` columns.

- Results page shows placeholder values
  - `result.html` has example numbers in the summary cards. Bind real values (prediction/confidence/AQ/risk) from `results` passed by the frontend.

---

## Roadmap (quick wins)

- Bind dynamic prediction/confidence/AQ/risk into `result.html` summary cards.
- Persist and enforce `feature_columns.pkl` for 1:1 training/inference alignment.
- Replace `datetime.utcnow()` with timezone‑aware timestamps.
- Add API and unit tests for validator, data_processor, and classifier.
- Add a small `/api/ping` returning model version and active feature count for sanity checks.
