from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from Scripts.data_clean_utils import perform_data_cleaning  
import os

set_config(transform_output="pandas")

from dotenv import load_dotenv
load_dotenv()

# credentials
dagshub_token = os.getenv("DAGSHUB_PAT")
dagshub_username = os.getenv("DAGSHUB_USERNAME")
dagshub_repo = os.getenv("DAGSHUB_REPO")

os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI()
model_pipe = None  # global pipeline

class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str


def load_model_information(file_path):
    with open(file_path) as f:
        return json.load(f)

@app.on_event("startup")
def load_model():
    global model_pipe
    print("🔄 Loading model and preprocessor from MLflow...")

    model_name = load_model_information("run_information.json")["model_name"]
    stage = "Production"
    model_path = f"models:/{model_name}/{stage}"

    model = mlflow.sklearn.load_model(model_path)
    preprocessor = joblib.load("models/preprocessor.joblib")

    model_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", model),
    ])
    print("✅ Model pipeline ready!")


@app.get("/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

@app.post("/predict")
async def do_predictions(data: Data):
    global model_pipe
    if model_pipe is None:
        return {"error": "Model not loaded yet"}
    
    pred_data = pd.DataFrame({
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
    }, index=[0])

    cleaned_data = perform_data_cleaning(pred_data)
    prediction = model_pipe.predict(cleaned_data)[0]
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
    print('swiggy data loaded successfuly')