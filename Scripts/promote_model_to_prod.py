import mlflow
import json
from mlflow import MlflowClient
import os
from dotenv import load_dotenv

load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
dagshub_username = os.getenv("DAGSHUB_USERNAME")
dagshub_repo = os.getenv("DAGSHUB_REPO")

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


# Get model name
model_name = load_model_information("run_information.json")["model_name"]
client = MlflowClient()

# Get latest staging model
latest_staging_versions = client.get_latest_versions(name=model_name, stages=["Staging"])

if not latest_staging_versions:
    print("ℹ️ No model found in Staging. Skipping promotion.")
else:
    latest_model_version_staging = latest_staging_versions[0].version
    print(f"🚀 Found Staging model v{latest_model_version_staging} for {model_name}")

    # Get latest production model (optional check)
    latest_production_versions = client.get_latest_versions(name=model_name, stages=["Production"])
    if latest_production_versions:
        latest_model_version_production = latest_production_versions[0].version
        if latest_model_version_staging == latest_model_version_production:
            print("⚠️ Staging model is the same as current Production. Skipping promotion.")
        else:
            # Promote to production and archive old versions
            client.transition_model_version_stage(
                name=model_name,
                version=latest_model_version_staging,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"🎯 Model {model_name} v{latest_model_version_staging} promoted to Production")
    else:
        # No production model yet → promote directly
        client.transition_model_version_stage(
            name=model_name,
            version=latest_model_version_staging,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"🎯 First Production model: {model_name} v{latest_model_version_staging}")
