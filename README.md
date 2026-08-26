# Diabetes ML - Web Deployment

## 1. Files

- `app.py`: Flask backend.
- `templates/index.html`: web interface.
- `requirements.txt`: required Python packages.
- `diabetes_model.joblib`: THE TRAINED MODEL FROM YOUR NOTEBOOK (must be copied into this folder).
- `README.md`: run instructions.

## 2. Important

The web application does NOT train the model again.

Training must be done in the Assignment 01 notebook:

```python
best_pipeline.fit(X_train, y_train)
```

Then save the complete preprocessing + model pipeline:

```python
import joblib
joblib.dump(best_pipeline, "diabetes_model.joblib")
```

Copy `diabetes_model.joblib` into the same folder as `app.py`.

The pipeline should contain the same preprocessing/representation used during training. Assignment 01 explicitly requires the same representation for new application inputs.

## 3. Install

Open VS Code Terminal in this folder:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

## 4. Run

```bash
python app.py
```

Open:

http://127.0.0.1:5000

## 5. Expected model input columns

The current web form sends:

- gender
- age
- hypertension
- heart_disease
- smoking_history
- bmi
- HbA1c_level
- blood_glucose_level

These names MUST match the columns used by your trained pipeline.

## 6. Troubleshooting

If you get:

`FileNotFoundError: diabetes_model.joblib`

copy the trained model file from your notebook into this folder.

If you get a feature/column error, check that the column names and preprocessing in the saved pipeline match the notebook.

If `predict_proba()` is unavailable, the app still displays the class prediction but not probability.

## 7. Assignment demonstration

Prepare at least three input cases and show the complete path:

User Input -> Feature Representation -> Preprocessing -> ML Model -> Prediction -> Output
