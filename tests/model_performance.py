import pytest
import mlflow
import dagshub
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
import os
import dagshub
import mlflow.client

from dotenv import load_dotenv
load_dotenv()
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
dagshub_username = os.getenv("DAGSHUB_USERNAME")
dagshub_repo=os.getenv("DAGSHUB_REPO")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")
if not dagshub_username:
    raise EnvironmentError("DAGSHUB_USERNAME environment variable is not set")
if not dagshub_repo:
    raise EnvironmentError("DAGSHUB_REPO environment variable is not set")



os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

# set model name
model_name = load_model_information("run_information.json")["model_name"]
stage = "Staging"

# load the model
model_path = f"models:/{model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# set the root path
root_path = Path(__file__).parent.parent

# load the preprocessor
preprocessor_path = root_path / "models" / "preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)


# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
])

test_data_path = root_path / "data" / "interim" / "test.csv"

@pytest.mark.parametrize(argnames="model_pipe, test_data_path, threshold_error",
                        argvalues=[(model_pipe, test_data_path, 5)])
def test_model_performance(model_pipe,test_data_path,threshold_error):
    # load test data
    df = pd.read_csv(test_data_path)
    
    # drop the missing values
    df.dropna(inplace=True)
    
    # make X and y
    X = df.drop(columns=["time_taken"])
    y = df['time_taken']
    
    # get the predictions
    y_pred = model_pipe.predict(X)
    
    # calculate the mean error
    mean_error = mean_absolute_error(y,y_pred)
    
    # check for performance
    assert mean_error <= threshold_error, f"The model does not pass the performance threshold of {threshold_error} minutes"
    print("The avg error is", mean_error)
    
    print(f"The {model_name} model passed the performance test")
    