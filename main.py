from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
from xgboost import XGBClassifier, Booster

app = FastAPI()

# Load preprocessor and label encoder
preprocessor = joblib.load("final_preprocessor.pkl")
label_encoder = joblib.load("final_label_encoder.pkl")

# Load XGBoost Booster from JSON and wrap it
booster = Booster()
booster.load_model("final_model.json")
model = XGBClassifier()
model._Booster = booster
# Match the feature count (important for XGBClassifier wrapper)

 pd.DataFrame([{
    "Age": 0,
    "Gender": "Male",
    "Occupation": "Engineer",
    "Sleep Duration": 0.0,
    "Quality of Sleep": 0,
    "Physical Activity Level": 0,
    "Stress Level": 0,
    "BMI Category": "Normal",
    "Heart Rate": 0,
    "Daily Steps": 0,
    "Blood Pressure": "Normal"
}])
).shape[1]

# HTML form
html_form = """
<!DOCTYPE html>
<html>
<head><title>Sleep Disorder Predictor</title></head>
<body style="font-family: Arial; background: #f9f9f9; padding: 30px;">
    <h2>🛌 Sleep Disorder Prediction</h2>
    <form method="post" action="/predict">
        Age: <input type="number" name="Age" required><br><br>
        Gender: <input type="text" name="Gender" required><br><br>
        Occupation: <input type="text" name="Occupation" required><br><br>
        Sleep Duration: <input type="number" step="0.1" name="Sleep_Duration" required><br><br>
        Quality of Sleep: <input type="number" name="Quality_of_Sleep" required><br><br>
        Physical Activity Level: <input type="number" name="Physical_Activity_Level" required><br><br>
        Stress Level: <input type="number" name="Stress_Level" required><br><br>
        BMI Category: <input type="text" name="BMI_Category" required><br><br>
        Heart Rate: <input type="number" name="Heart_Rate" required><br><br>
        Daily Steps: <input type="number" name="Daily_Steps" required><br><br>
        <input type="submit" value="Predict">
    </form>
    <div>{result}</div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def form_get():
    return HTMLResponse(content=html_form.format(result=""), status_code=200)

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Age: int = Form(...),
    Gender: str = Form(...),
    Occupation: str = Form(...),
    Sleep_Duration: float = Form(...),
    Quality_of_Sleep: int = Form(...),
    Physical_Activity_Level: int = Form(...),
    Stress_Level: int = Form(...),
    BMI_Category: str = Form(...),
    Heart_Rate: int = Form(...),
    Daily_Steps: int = Form(...)
):
    try:
        # Match column names to original training data
        input_data = pd.DataFrame([{
            "Age": Age,
            "Gender": Gender,
            "Occupation": Occupation,
            "Sleep Duration": Sleep_Duration,
            "Quality of Sleep": Quality_of_Sleep,
            "Physical Activity Level": Physical_Activity_Level,
            "Stress Level": Stress_Level,
            "BMI Category": BMI_Category,
            "Heart Rate": Heart_Rate,
            "Daily Steps": Daily_Steps
        }])

        # Predict
        transformed = preprocessor.transform(input_data)
        pred_encoded = model.predict(transformed)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        result_html = f"<h3 style='color:green;'>Prediction: {pred_label}</h3>"

    except Exception as e:
        result_html = f"<h3 style='color:red;'>Error: {str(e)}</h3>"

    return HTMLResponse(content=html_form.format(result=result_html), status_code=200)
