# Swiggy Delivery Time Prediction

A full end-to-end **Machine Learning + MLOps project** to predict delivery times for Swiggy orders.
This repo integrates **DVC**, **MLflow**, **DagsHub**, **GitHub Actions CI/CD**, **Docker**, and **AWS EC2/ECR** for a complete production-ready pipeline.

## 📂 Project Structure

```
├── LICENSE
├── Makefile           <- Commands for automation (data, train, test, deploy)
├── README.md          <- Project documentation
├── data
│   ├── raw            <- Raw data files
│   ├── processed      <- Cleaned and feature-engineered data
│   ├── interim        <- Intermediate datasets
│   └── external       <- External datasets
│
├── models             <- Trained and serialized ML models
├── notebooks          <- Jupyter notebooks for EDA & experimentation
├── reports            <- Generated reports and visualizations
├── src                <- Source code for data processing and ML pipeline
├── tests              <- Unit and integration tests
├── Scripts            <- Deployment and utility scripts (e.g., promote model)
├── app.py             <- Main application entry point
├── requirements.txt   <- Python dependencies
├── setup.py           <- Package setup
├── Dockerfile         <- Docker container definition
├── dvc.yaml           <- DVC pipeline stages
└── .github/workflows  <- CI/CD workflow definitions
```

##  🚀 CI/CD Pipeline

This repository includes a GitHub Actions workflow [.github/workflows/ci-cd.yaml] (https://github.com/AyushiGupta9723/Swiggy-Food-Delivery-Time-Prediction/actions/runs/18060788695/workflow) that automates the following:

✅ Install dependencies and set up environment

🔑 Configure AWS credentials

📊 Configure MLflow with DagsHub

📦 Pull datasets and models with DVC

🔄 Run the training pipeline (dvc repro)

🧪 Test model registry and performance with pytest

🏷️ Promote best model to production

🐳 Build & push Docker image to Amazon ECR

🚀 Deploy containerized app to AWS EC2

---

## ⚡ End-to-End Workflow


1. **Data Collection** → `data/raw/`
2. **Data Processing & Cleaning** → `src/`
3. **EDA** → `notebooks/`
4. **Feature Engineering** → `src/`
5. **Model Training** → `dvc repro`
6. **Model Evaluation** → `reports/`
7. **Inference / Prediction** → `app.py`
8. **Deployment** → Docker + AWS EC2
9. **Testing** → `pytest tests/`
10. **Continuous Integration** → GitHub Actions CI/CD

---

## 📦 DVC Pipeline

| Stage            | Script                             | Dependencies                  | Outputs                       | Params         |
| ---------------- | ---------------------------------- | ----------------------------- | ----------------------------- | -------------- |
| **Preprocess**   | `src/preprocess.py`                | `data/raw/*.csv`              | `data/processed/train.csv`    | `preprocess.*` |
| **Feature Engg** | `src/features.py`                  | `data/processed/train.csv`    | `data/processed/features.csv` | `features.*`   |
| **Train**        | `src/train.py`                     | `data/processed/features.csv` | `models/model.joblib`         | `train.*`      |
| **Evaluate**     | `src/evaluate.py`                  | `models/model.joblib`         | `reports/metrics.json`        | `evaluate.*`   |
| **Registry**     | `Scripts/promote_model_to_prod.py` | `models/model.joblib`         | `mlruns/`                     | —              |


---

## 📊 MLflow & DagsHub Integration

All experiments are tracked with **MLflow**, backed by **DagsHub**.

* ✅ Track metrics (`RMSE`, `MAE`, `R²`)
* ✅ Log parameters and artifacts
* ✅ Compare experiment runs
* ✅ Register best model → **Production stage**

🔗 [View Experiments on DagsHub](https://dagshub.com/ayushigupta9723/Swiggy-Delivery-Time-Prediction)

---

## 🚀 CI/CD Pipeline

See [`.github/workflows/ci-cd.yaml`](https://github.com/AyushiGupta9723/Swiggy-Food-Delivery-Time-Prediction/actions/runs/18060788695/workflow)

---

## 🛠️ Getting Started

```bash
# Clone
git clone https://github.com/AyushiGupta9723/Swiggy-Food-Delivery-Time-Prediction.git
cd Swiggy-Food-Delivery-Time-Prediction

# Setup env
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

# Run pipeline
dvc repro

# Run tests
pytest tests/

```

---

## 🐳 Deployment

Build & run locally:

```bash
docker build -t swiggy-delivery:latest .
docker run -p 8000:8000 swiggy-delivery:latest
```

Deploys automatically on **AWS EC2** via **GitHub Actions**.

---


## 🙌 Acknowledgments

* Inspired by **Swiggy** delivery system
* **DVC, MLflow, DagsHub, AWS, Docker** for MLOps backbone
