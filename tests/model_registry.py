import pytest
import mlflow
from mlflow import MlflowClient
import dagshub
import json
import os 
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

# set model name
model_name = load_model_information("run_information.json")["model_name"]


@pytest.mark.parametrize(argnames="model_name, stage",
                         argvalues=[(model_name, "Staging")])
def test_load_model_from_registry(model_name,stage):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name,stages=[stage])
    latest_version = latest_versions[0].version if latest_versions else None
    
    assert latest_version is not None, f"No model at {stage} stage"
    
    # load the model
    model_path = f"models:/{model_name}/{stage}"

    # load the latest model from model registry
    model = mlflow.sklearn.load_model(model_path)
    
    assert model is not None, "Failed to load model from registry"
    print(f"The {model_name} model with version {latest_version} was loaded successfully")
    