import os
from functools import lru_cache

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

HOUSE_NUMERIC_COLS = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
HOUSE_CATEGORICAL_COLS = ["House direction", "Balcony direction", "Legal status", "Furniture state"]
HOUSE_CATEGORY_OPTIONS = {
    "House direction": ["Bắc", "Nam", "Đông", "Tây", "Đông - Bắc", "Đông - Nam", "Tây - Bắc", "Tây - Nam"],
    "Balcony direction": ["Bắc", "Nam", "Đông", "Tây", "Đông - Bắc", "Đông - Nam", "Tây - Bắc", "Tây - Nam"],
    "Legal status": ["Have certificate", "Sale contract"],
    "Furniture state": ["Basic", "Full"],
}


def env_flag(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# On Render free tier (512MB), disable house model by default to avoid worker restarts.
HOUSE_MODE_ENABLED = env_flag("ENABLE_HOUSE_MODE", default=(os.environ.get("RENDER") is None))
HOUSE_MODE_DISABLED_MSG = (
    "Mode Giá Nhà đang tắt trên bản cloud free để tránh vượt RAM (model quá lớn). "
    "Bạn có thể dùng mode Tiểu Đường hoặc bật ENABLE_HOUSE_MODE=true khi nâng gói RAM."
)


def load_house_resources():
    artifact = joblib.load("vietnam_house_price_model.joblib")
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    return {
        "model": model,
        "feature_cols": feature_cols,
        "numeric_cols": HOUSE_NUMERIC_COLS,
        "categorical_cols": HOUSE_CATEGORICAL_COLS,
        "options": HOUSE_CATEGORY_OPTIONS,
    }


@lru_cache(maxsize=1)
def get_house_resources():
    return load_house_resources()


@lru_cache(maxsize=1)
def get_diabetes_model():
    return joblib.load("diabetes_prediction_final_model.joblib")


MODE_META = {
    "house": {
        "title": "Dự Đoán Giá Nhà Việt Nam",
        "subtitle": "Input nhà đất -> mô hình ML -> giá dự đoán",
        "hint": "Mô hình dự đoán theo đơn vị tỷ VND, hệ thống quy đổi thêm sang VND.",
        "note": "Kết quả phục vụ mục đích học tập và tham khảo mô hình, không thay thế định giá thực tế.",
        "button": "DỰ ĐOÁN GIÁ",
        "result_title": "Giá Dự Đoán",
    },
    "diabetes": {
        "title": "Dự Đoán Nguy Cơ Tiểu Đường",
        "subtitle": "Input sức khỏe -> mô hình ML -> mức nguy cơ",
        "hint": "Nhập đúng thông tin sức khỏe để mô hình ước lượng chính xác hơn.",
        "note": "Kết quả chỉ phục vụ mục đích học tập, không thay thế chẩn đoán y khoa.",
        "button": "DỰ ĐOÁN NGUY CƠ",
        "result_title": "Kết Quả Dự Đoán",
    },
}


def to_key(name):
    return name.lower().replace(" ", "_")


def format_vnd_from_billion(value):
    value_vnd = int(round(float(value) * 1_000_000_000))
    return f"{value_vnd:,}".replace(",", ".") + " VND"


def format_billion_vnd(value):
    return f"{float(value):.2f} tỷ VND"


def get_mode(raw_mode):
    return raw_mode if raw_mode in MODE_META else "house"


def build_house_fields():
    fields = []
    for col in HOUSE_NUMERIC_COLS:
        fields.append(
            {
                "name": to_key(col),
                "label": col,
                "type": "number",
                "min": "0",
                "step": "0.1",
                "placeholder": f"Nhập {col.lower()}",
            }
        )

    for col in HOUSE_CATEGORICAL_COLS:
        fields.append(
            {
                "name": to_key(col),
                "label": col,
                "type": "select",
                "options": HOUSE_CATEGORY_OPTIONS.get(col, []),
            }
        )

    return fields


def build_diabetes_fields():
    return [
        {"name": "gender", "label": "Gender", "type": "select", "options": ["Female", "Male", "Other"]},
        {"name": "age", "label": "Age", "type": "number", "min": "0", "step": "0.1", "placeholder": "Ví dụ: 50"},
        {"name": "hypertension", "label": "Hypertension", "type": "select", "options": ["0", "1"], "option_labels": {"0": "Không", "1": "Có"}},
        {"name": "heart_disease", "label": "Heart Disease", "type": "select", "options": ["0", "1"], "option_labels": {"0": "Không", "1": "Có"}},
        {"name": "smoking_history", "label": "Smoking History", "type": "select", "options": ["never", "former", "current", "not current", "ever", "No Info"]},
        {"name": "bmi", "label": "BMI", "type": "number", "min": "0", "step": "0.1", "placeholder": "Ví dụ: 27.5"},
        {"name": "hba1c", "label": "HbA1c Level", "type": "number", "min": "0", "step": "0.1", "placeholder": "Ví dụ: 6.2"},
        {"name": "glucose", "label": "Blood Glucose Level", "type": "number", "min": "0", "step": "1", "placeholder": "Ví dụ: 140"},
    ]


def default_form_values(fields):
    return {field["name"]: "" for field in fields}


def predict_house(form_values):
    house = get_house_resources()
    sample = {}
    for col in house["numeric_cols"]:
        key = to_key(col)
        raw = form_values.get(key, "").strip()
        if raw == "":
            raise ValueError(f"Vui lòng nhập trường số: {col}")
        sample[col] = float(raw)

    for col in house["categorical_cols"]:
        key = to_key(col)
        raw = form_values.get(key, "").strip()
        if raw == "":
            raise ValueError(f"Vui lòng chọn trường phân loại: {col}")
        sample[col] = raw

    data = pd.DataFrame([sample], columns=house["feature_cols"])
    raw_pred = float(house["model"].predict(data)[0])

    return {
        "primary": format_vnd_from_billion(raw_pred),
        "lines": [
            f"Tương đương: {format_billion_vnd(raw_pred)}",
            f"Giá trị thô từ mô hình: {raw_pred:.2f} (tỷ VND)",
        ],
    }


def predict_diabetes(form_values):
    diabetes_model = get_diabetes_model()
    payload = pd.DataFrame(
        [
            {
                "gender": form_values.get("gender", "").strip(),
                "age": float(form_values.get("age", "")),
                "hypertension": int(form_values.get("hypertension", "")),
                "heart_disease": int(form_values.get("heart_disease", "")),
                "smoking_history": form_values.get("smoking_history", "").strip(),
                "bmi": float(form_values.get("bmi", "")),
                "HbA1c_level": float(form_values.get("hba1c", "")),
                "blood_glucose_level": float(form_values.get("glucose", "")),
            }
        ]
    )

    pred = int(diabetes_model.predict(payload)[0])
    primary = "Có nguy cơ mắc tiểu đường" if pred == 1 else "Không có nguy cơ mắc tiểu đường theo mô hình"

    lines = []
    if hasattr(diabetes_model, "predict_proba"):
        prob = float(diabetes_model.predict_proba(payload)[0][1] * 100)
        lines.append(f"Xác suất dự đoán: {prob:.2f}%")

    return {"primary": primary, "lines": lines}


@app.route("/", methods=["GET", "POST"])
def home():
    mode = get_mode(request.values.get("mode", "house"))
    mode_info = dict(MODE_META[mode])
    fields = build_house_fields() if mode == "house" else build_diabetes_fields()
    form_values = default_form_values(fields)
    result = None
    error = None
    warning = None

    house_mode_available = not (mode == "house" and not HOUSE_MODE_ENABLED)
    if mode == "house" and not HOUSE_MODE_ENABLED:
        warning = HOUSE_MODE_DISABLED_MSG
        mode_info["hint"] = HOUSE_MODE_DISABLED_MSG

    if request.method == "POST":
        form_values = {field["name"]: request.form.get(field["name"], "").strip() for field in fields}
        try:
            if mode == "house" and not HOUSE_MODE_ENABLED:
                raise ValueError(HOUSE_MODE_DISABLED_MSG)
            if mode == "house":
                pred = predict_house(form_values)
            else:
                pred = predict_diabetes(form_values)

            result = {
                "title": mode_info["result_title"],
                "primary": pred["primary"],
                "lines": pred["lines"],
            }
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        mode=mode,
        mode_info=mode_info,
        fields=fields,
        form_values=form_values,
        house_mode_available=house_mode_available,
        warning=warning,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
