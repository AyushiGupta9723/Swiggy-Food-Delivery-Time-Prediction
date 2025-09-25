import mlflow
import dagshub
import json
from mlflow import MlflowClient
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


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


# get model name
model_name = load_model_information("run_information.json")["model_name"]
stage = "Staging"

# get the latest version from staging stage
client = MlflowClient()

# get the latest version of model in staging
latest_versions = client.get_latest_versions(name=model_name,stages=[stage])

latest_model_version_staging = latest_versions[0].version

# promotion stage
promotion_stage = "Production"

client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version_staging,
    stage=promotion_stage,
    archive_existing_versions=True
)