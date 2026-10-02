# Abalone Age Prediction

A production-style machine learning application for predicting the age of abalone using multiple regression models deployed on Microsoft Azure Machine Learning.

The project demonstrates an end-to-end machine learning and MLOps workflow, starting from model experimentation and evaluation, followed by model registration, real-time endpoint deployment, traffic splitting, REST API inference, and a publicly accessible Streamlit application.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Objective](#project-objective)
3. [Dataset](#dataset)
4. [Machine Learning Experimentation](#machine-learning-experimentation)
5. [Model Selection](#model-selection)
6. [Model Performance](#model-performance)
7. [Azure Machine Learning](#azure-machine-learning)
8. [Model Registration](#model-registration)
9. [Real-Time Endpoint](#real-time-endpoint)
10. [Traffic Splitting](#traffic-splitting)
11. [Endpoint Testing](#endpoint-testing)
12. [Streamlit Application](#streamlit-application)
13. [Public Deployment](#public-deployment)
14. [Architecture](#architecture)
15. [End-to-End Workflow](#end-to-end-workflow)
16. [Project Structure](#project-structure)
17. [Technologies Used](#technologies-used)
18. [Configuration](#configuration)
19. [Local Setup](#local-setup)
20. [API Input Format](#api-input-format)
21. [Example Prediction](#example-prediction)
22. [Security](#security)
23. [Future Improvements](#future-improvements)
24. [Limitations](#limitations)
25. [Conclusion](#conclusion)
26. [License](#license)

---

# Overview

The Abalone Age Prediction project is a machine learning application that predicts the number of rings of an abalone using physical measurements.

The project was developed as an end-to-end machine learning deployment workflow using Microsoft Azure Machine Learning.

Instead of deploying only one model, multiple regression models were evaluated and compared using Azure Machine Learning. Two selected models were then registered and deployed to the same managed real-time endpoint.

Traffic was configured between the two deployments using an 80% / 20% split.

A Streamlit application was then created as the frontend. The application collects abalone measurements from the user, sends them to the Azure ML endpoint through a REST API, and displays the prediction.

The overall workflow is:

```text
Dataset
   |
   v
Data Preparation
   |
   v
Model Experimentation
   |
   v
Evaluate Multiple Models
   |
   v
Compare Model Metrics
   |
   v
Select Models
   |
   v
Register Models in Azure ML
   |
   v
Deploy Models to Real-Time Endpoint
   |
   v
Configure Traffic Split
   |
   v
REST API Inference
   |
   v
Streamlit Application
   |
   v
Public Deployment
```

---

# Project Objective

The main objective of this project is to demonstrate how a machine learning model can move from experimentation to a publicly accessible application.

The project covers:

- Training and evaluating multiple regression models.
- Comparing model performance using different evaluation metrics.
- Selecting models for deployment.
- Registering models using Azure Machine Learning.
- Deploying multiple models to a managed online endpoint.
- Splitting traffic between model deployments.
- Testing real-time inference through a REST API.
- Building a user-friendly Streamlit interface.
- Deploying the Streamlit application publicly.
- Managing Azure endpoint credentials securely.

---

# Dataset

The project uses the Abalone dataset.

The target variable is:

```text
Rings
```

The model uses eight input features.

| Feature | Description |
| ------- | ----------- |
| `Sex` | Sex of the abalone |
| `Length` | Longest shell measurement |
| `Diameter` | Diameter measurement |
| `Height` | Height measurement |
| `Whole_weight` | Whole abalone weight |
| `Shucked_weight` | Weight after removing the shell |
| `Viscera_weight` | Weight of the viscera |
| `Shell_weight` | Weight of the shell |

The target variable is:

```text
Rings
```

`Rings` is not supplied to the deployed model because it is the value being predicted.

The deployed model therefore expects exactly these eight inputs:

```text
Sex
Length
Diameter
Height
Whole_weight
Shucked_weight
Viscera_weight
Shell_weight
```

---

# Machine Learning Experimentation

The machine learning experimentation was performed using Azure Machine Learning.

Multiple regression models and model configurations were evaluated and compared.

The Azure ML experiment displayed 23 models.

The evaluated models included different algorithms and model combinations such as:

- Voting Ensemble
- LightGBM
- ElasticNet
- XGBoost
- Extreme Random Trees
- Decision Tree
- Random Forest
- Other regression models and preprocessing combinations

The experiment results included the following models near the top of the displayed results:

```text
Voting Ensemble
Normalized RMSE: 0.07756

MaxAbsScaler + LightGBM
Normalized RMSE: 0.07990
```

The Voting Ensemble used multiple algorithms as part of its ensemble configuration, including:

```text
LightGBM
ElasticNet
ElasticNet
ElasticNet
ElasticNet
XGBoostRegressor
```

The experiment provided multiple evaluation metrics and predicted-versus-true visualizations for the evaluated models.

---

# Model Selection

Two models were selected for deployment:

1. Voting Ensemble
2. MaxAbsScaler + LightGBM

The Voting Ensemble was selected as the primary deployment, while MaxAbsScaler + LightGBM was selected as the secondary deployment.

The project uses the following traffic configuration:

```text
Voting Ensemble          80%
MaxAbsScaler + LightGBM  20%
```

This configuration allows both models to be deployed under the same real-time endpoint.

---

# Model Performance

The following metrics were obtained from the Azure Machine Learning experiment.

## Voting Ensemble

| Metric | Value |
| ------ | ----- |
| Explained Variance | 0.5564085 |
| Mean Absolute Error | 1.523987 |
| Mean Absolute Percentage Error | 15.36744 |
| Median Absolute Error | 1.030426 |
| Normalized Mean Absolute Error | 0.05442809 |
| Normalized Median Absolute Error | 0.03680093 |
| Normalized Root Mean Squared Error | 0.07755897 |
| Normalized Root Mean Squared Log Error | 0.06561576 |
| R² Score | 0.5564083 |
| Root Mean Squared Error | 2.171651 |
| Root Mean Squared Log Error | 0.1776908 |
| Spearman Correlation | 0.7299540 |

The Azure ML experiment also provided a predicted-versus-true visualization for the Voting Ensemble.

---

## MaxAbsScaler + LightGBM

| Metric | Value |
| ------ | ----- |
| Explained Variance | 0.5292171 |
| Mean Absolute Error | 1.550834 |
| Mean Absolute Percentage Error | 15.67565 |
| Median Absolute Error | 1.017619 |
| Normalized Mean Absolute Error | 0.05538691 |
| Normalized Median Absolute Error | 0.03634353 |
| Normalized Root Mean Squared Error | 0.07990389 |
| Normalized Root Mean Squared Log Error | 0.06768520 |
| R² Score | 0.5291797 |
| Root Mean Squared Error | 2.237309 |
| Root Mean Squared Log Error | 0.1832949 |
| Spearman Correlation | 0.7086981 |

The Azure ML experiment also provided a predicted-versus-true visualization for the LightGBM model.

---

## Model Comparison

| Metric | Voting Ensemble | MaxAbsScaler + LightGBM |
| ------ | --------------- | ----------------------- |
| Explained Variance | 0.5564085 | 0.5292171 |
| Mean Absolute Error | 1.523987 | 1.550834 |
| Mean Absolute Percentage Error | 15.36744 | 15.67565 |
| Median Absolute Error | 1.030426 | 1.017619 |
| Normalized Mean Absolute Error | 0.05442809 | 0.05538691 |
| Normalized Median Absolute Error | 0.03680093 | 0.03634353 |
| Normalized Root Mean Squared Error | 0.07755897 | 0.07990389 |
| Normalized Root Mean Squared Log Error | 0.06561576 | 0.06768520 |
| R² Score | 0.5564083 | 0.5291797 |
| Root Mean Squared Error | 2.171651 | 2.237309 |
| Root Mean Squared Log Error | 0.1776908 | 0.1832949 |
| Spearman Correlation | 0.7299540 | 0.7086981 |

The project uses the Voting Ensemble as the primary deployment and MaxAbsScaler + LightGBM as the secondary deployment.

---

# Azure Machine Learning

Microsoft Azure Machine Learning is used as the main machine learning and deployment platform.

The Azure ML workspace was used for:

- Model experimentation
- Model evaluation
- Model registration
- Managed online endpoint creation
- Model deployment
- Traffic management
- Real-time inference
- Endpoint testing

## Azure ML Workspace

```text
Workspace:
Atif_Workspace

Resource Group:
Atif_Resource_Group

Region:
Canada Central

Subscription:
Azure subscription 1
```

---

# Model Registration

The selected models were registered in Azure Machine Learning before deployment.

The Voting Ensemble model was registered as:

```text
Jorvis-VotingEnsemble:1
```

The model was registered using the MLflow format.

The model artifact contains MLflow model information and the required model/environment files, including:

```text
MLmodel
model.pkl
conda.yaml
python_env.yaml
requirements.txt
```

The MLflow model signature defines the eight input columns:

```text
Sex
Length
Diameter
Height
Whole_weight
Shucked_weight
Viscera_weight
Shell_weight
```

The output is an integer prediction representing the predicted number of rings.

---

# Real-Time Endpoint

The selected models were deployed to the same Azure Machine Learning managed online endpoint.

Endpoint name:

```text
atif-workspace-lzitd-ensemble
```

Endpoint type:

```text
Managed Online Endpoint
```

Primary deployment:

```text
jorvis-votingensemble-1
```

REST scoring URL:

```text
https://atif-workspace-lzitd-ensemble.canadacentral.inference.ml.azure.com/score
```

The endpoint uses key-based authentication.

The endpoint is configured for real-time inference and provides an HTTP interface that can be called by the Streamlit application.

---

# Endpoint Configuration

The endpoint uses managed compute.

```text
Compute Type:
Managed

SKU:
Standard_D2as_v4

Instance Count:
1
```

Public network access was enabled so that the Streamlit application could communicate with the Azure ML inference endpoint.

---

# Traffic Splitting

Both selected models are deployed under the same Azure ML real-time endpoint.

The traffic configuration is:

```text
Voting Ensemble          80%
MaxAbsScaler + LightGBM  20%
```

The traffic flow is:

```text
                         Azure ML Endpoint
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
             Voting Ensemble          LightGBM
                  80%                     20%
                    |                       |
                    +-----------+-----------+
                                |
                                v
                           Prediction
```

The traffic split allows both deployments to receive real-time prediction requests through the same endpoint.

The traffic allocation can be modified later from the Azure Machine Learning endpoint configuration.

---

# Endpoint Testing

The deployed endpoint was tested using Azure ML CLI and REST API requests.

A sample abalone record was used for testing.

Input:

```text
Sex              = M
Length           = 0.455
Diameter         = 0.365
Height           = 0.095
Whole_weight     = 0.514
Shucked_weight   = 0.2245
Viscera_weight   = 0.101
Shell_weight     = 0.15
```

A successful real-time request returned a prediction approximately equal to:

```text
10.2224
```

This confirmed that the deployed model was able to receive the request and return a real-time prediction.

---

# Azure ML CLI Testing

The endpoint can be tested using the Azure CLI.

Create a request file:

```json
{
  "input_data": {
    "columns": [
      "Sex",
      "Length",
      "Diameter",
      "Height",
      "Whole_weight",
      "Shucked_weight",
      "Viscera_weight",
      "Shell_weight"
    ],
    "data": [
      [
        "M",
        0.455,
        0.365,
        0.095,
        0.514,
        0.2245,
        0.101,
        0.15
      ]
    ],
    "index": [0]
  }
}
```

Save the file as:

```text
request.json
```

Then run:

```bash
az ml online-endpoint invoke \
  --name atif-workspace-lzitd-ensemble \
  --deployment-name jorvis-votingensemble-1 \
  --request-file request.json \
  --resource-group Atif_Resource_Group \
  --workspace-name Atif_Workspace
```

A successful request returns the model prediction.

---

# REST API Testing

The same endpoint can also be called directly using an HTTP request.

Example Python request:

```python
import requests

endpoint_url = "https://atif-workspace-lzitd-ensemble.canadacentral.inference.ml.azure.com/score"
endpoint_key = "YOUR_ENDPOINT_KEY"

headers = {
    "Authorization": f"Bearer {endpoint_key}",
    "Content-Type": "application/json",
    "azureml-model-deployment": "jorvis-votingensemble-1"
}

payload = {
    "input_data": {
        "columns": [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole_weight",
            "Shucked_weight",
            "Viscera_weight",
            "Shell_weight"
        ],
        "data": [
            [
                "M",
                0.455,
                0.365,
                0.095,
                0.514,
                0.2245,
                0.101,
                0.15
            ]
        ],
        "index": [0]
    }
}

response = requests.post(
    endpoint_url,
    headers=headers,
    json=payload
)

print(response.status_code)
print(response.text)
```

Replace:

```text
YOUR_ENDPOINT_KEY
```

with the current endpoint key stored securely in your environment or Streamlit Secrets.

---

# Streamlit Application

A Streamlit application was created as the user-facing interface for the machine learning service.

The application provides input fields for all eight model features.

The interface includes:

- Abalone feature inputs
- Input validation
- Prediction button
- Azure ML API integration
- Prediction result display
- Clean and responsive user interface

The Streamlit application does not perform the model prediction locally.

Instead, it sends the input data to the Azure ML real-time endpoint.

---

# Streamlit Prediction Flow

```text
User
 |
 v
Enter Abalone Measurements
 |
 v
Click Predict
 |
 v
Streamlit Application
 |
 v
Create JSON Request
 |
 v
Azure ML REST Endpoint
 |
 v
Traffic Split
 |
 +-------------------------+
 |                         |
 v                         v
Voting Ensemble          LightGBM
    80%                    20%
 |                         |
 +------------+------------+
              |
              v
         Prediction
              |
              v
      Streamlit Result
              |
              v
             User
```

---

# Streamlit Application Example

A simplified version of the prediction logic is:

```python
import requests
import streamlit as st

ENDPOINT_URL = st.secrets["AZURE_ENDPOINT_URL"]
ENDPOINT_KEY = st.secrets["AZURE_ENDPOINT_KEY"]

headers = {
    "Authorization": f"Bearer {ENDPOINT_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "input_data": {
        "columns": [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole_weight",
            "Shucked_weight",
            "Viscera_weight",
            "Shell_weight"
        ],
        "data": [
            [
                sex,
                length,
                diameter,
                height,
                whole_weight,
                shucked_weight,
                viscera_weight,
                shell_weight
            ]
        ],
        "index": [0]
    }
}

response = requests.post(
    ENDPOINT_URL,
    headers=headers,
    json=payload
)

prediction = response.json()

st.write(prediction)
```

The complete application can contain additional UI styling, validation, error handling, and result formatting.

---

# Public Deployment

The Streamlit application can be deployed publicly using Streamlit Community Cloud.

The deployment architecture is:

```text
Local Streamlit Application
          |
          v
     GitHub Repository
          |
          v
Streamlit Community Cloud
          |
          v
   Public Web Application
          |
          v
 Azure ML REST Endpoint
          |
          v
     Model Deployment
          |
          v
       Prediction
```

The public application allows users to interact with the deployed machine learning model through a web browser.

---

# Streamlit Community Cloud Setup

## 1. Push the Project to GitHub

Create a GitHub repository containing:

```text
app.py
requirements.txt
README.md
.gitignore
```

Do not upload secret files.

## 2. Add Requirements

A basic `requirements.txt` may contain:

```text
streamlit
requests
```

Add any additional packages required by the Streamlit application.

## 3. Deploy on Streamlit Community Cloud

Select the GitHub repository and configure:

```text
Main file:
app.py
```

Streamlit Community Cloud will install the dependencies and start the application.

## 4. Configure Secrets

In the Streamlit deployment settings, add:

```toml
AZURE_ENDPOINT_URL = "https://atif-workspace-lzitd-ensemble.canadacentral.inference.ml.azure.com/score"
AZURE_ENDPOINT_KEY = "YOUR_ENDPOINT_KEY"
```

The application can then access them using:

```python
import streamlit as st

ENDPOINT_URL = st.secrets["AZURE_ENDPOINT_URL"]
ENDPOINT_KEY = st.secrets["AZURE_ENDPOINT_KEY"]
```

---

# Architecture

The complete architecture contains four major layers:

```text
                    USER
                     |
                     v
          +----------------------+
          |   Streamlit UI       |
          |                      |
          | Feature Inputs       |
          | Prediction Button    |
          | Result Display       |
          +----------+-----------+
                     |
                     | HTTPS REST API
                     v
          +----------------------+
          | Azure ML Endpoint    |
          |                      |
          | Managed Online       |
          | Endpoint             |
          +----------+-----------+
                     |
             Traffic Routing
                     |
          +----------+-----------+
          |                      |
          v                      v
 +----------------+    +----------------+
 | Voting         |    | MaxAbsScaler   |
 | Ensemble       |    | + LightGBM     |
 |                |    |                |
 | 80%            |    | 20%            |
 +-------+--------+    +-------+--------+
          |                    |
          +----------+---------+
                     |
                     v
                Prediction
                     |
                     v
             Streamlit Result
```

---

# End-to-End Workflow

The complete project lifecycle is:

```text
                         MACHINE LEARNING
                              |
                              v
                         Dataset
                              |
                              v
                    Data Preparation
                              |
                              v
                  Model Experimentation
                              |
                              v
                     23 Models Evaluated
                              |
                              v
                      Model Comparison
                              |
                +-------------+-------------+
                |                           |
                v                           v
         Voting Ensemble              LightGBM
                |                           |
                +-------------+-------------+
                              |
                              v
                       Model Registration
                              |
                              v
                    Azure ML Model Registry
                              |
                              v
                     Model Deployment
                              |
                              v
                  Managed Online Endpoint
                              |
                              v
                       Traffic Splitting
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
              80% Traffic            20% Traffic
                  |                       |
                  v                       v
          Voting Ensemble             LightGBM
                  |                       |
                  +-----------+-----------+
                              |
                              v
                         REST API
                              |
                              v
                       Streamlit App
                              |
                              v
                  Streamlit Community Cloud
                              |
                              v
                     Public Application
```

---

# Project Structure

```text
abalone-age-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

A possible expanded structure is:

```text
abalone-age-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml
│
└── assets/
    └── screenshots/
```

The exact project structure can be adjusted depending on the final Streamlit application implementation.

---

# Technologies Used

## Machine Learning

- Python
- Scikit-learn
- LightGBM
- XGBoost
- Ensemble Learning
- Regression
- MLflow

## Cloud and MLOps

- Microsoft Azure
- Azure Machine Learning
- Azure ML Studio
- Azure ML Model Registry
- Managed Online Endpoints
- Real-Time Inference
- Traffic Splitting

## Application

- Streamlit
- Python
- Requests
- REST API

## Deployment

- GitHub
- Streamlit Community Cloud
- Azure Machine Learning

---

# Azure ML Resources

The main Azure resources used in the project are:

| Resource | Value |
| -------- | ----- |
| Workspace | `Atif_Workspace` |
| Resource Group | `Atif_Resource_Group` |
| Region | `Canada Central` |
| Subscription | `Azure subscription 1` |
| Endpoint | `atif-workspace-lzitd-ensemble` |
| Primary Deployment | `jorvis-votingensemble-1` |
| Registered Model | `Jorvis-VotingEnsemble:1` |
| Endpoint Type | Managed Online Endpoint |
| Authentication | Key |
| Compute Type | Managed |
| Compute SKU | `Standard_D2as_v4` |
| Instance Count | `1` |

---

# Configuration

The Streamlit application requires two main configuration values.

| Variable | Description |
| -------- | ----------- |
| `AZURE_ENDPOINT_URL` | Azure ML real-time scoring endpoint |
| `AZURE_ENDPOINT_KEY` | Authentication key for the Azure ML endpoint |

Example:

```env
AZURE_ENDPOINT_URL=https://atif-workspace-lzitd-ensemble.canadacentral.inference.ml.azure.com/score
AZURE_ENDPOINT_KEY=your_endpoint_key
```

For Streamlit Community Cloud, use Streamlit Secrets instead of committing these values to the repository.

Example:

```toml
AZURE_ENDPOINT_URL = "https://atif-workspace-lzitd-ensemble.canadacentral.inference.ml.azure.com/score"
AZURE_ENDPOINT_KEY = "your_endpoint_key"
```

---

# Local Setup

## Prerequisites

Make sure the following are installed:

- Python 3.10+
- Git
- Azure ML endpoint
- Azure ML endpoint authentication key

---

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd abalone-age-prediction
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

A basic `requirements.txt` can contain:

```text
streamlit
requests
```

---

## 4. Configure Credentials

For local development, create:

```text
.streamlit/secrets.toml
```

Add:

```toml
AZURE_ENDPOINT_URL = "https://atif-workspace-lzitd-ensemble.canadacentral.inference.ml.azure.com/score"
AZURE_ENDPOINT_KEY = "your_endpoint_key"
```

Make sure this file is included in `.gitignore`.

---

## 5. Run the Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# API Input Format

Azure ML expects the request in the following format:

```json
{
  "input_data": {
    "columns": [
      "Sex",
      "Length",
      "Diameter",
      "Height",
      "Whole_weight",
      "Shucked_weight",
      "Viscera_weight",
      "Shell_weight"
    ],
    "data": [
      [
        "M",
        0.455,
        0.365,
        0.095,
        0.514,
        0.2245,
        0.101,
        0.15
      ]
    ],
    "index": [0]
  }
}
```

The request contains:

```text
columns
data
index
```

The `columns` array contains the model feature names.

The `data` array contains the corresponding feature values.

The `index` identifies the input row.

---

# API Request Headers

The Azure ML endpoint uses authentication.

A request can include:

```python
headers = {
    "Authorization": f"Bearer {ENDPOINT_KEY}",
    "Content-Type": "application/json"
}
```

When targeting a specific deployment, the deployment header can also be supplied:

```python
headers["azureml-model-deployment"] = "jorvis-votingensemble-1"
```

The endpoint configuration should determine how traffic is routed between deployments.

---

# Example Prediction

Example input:

```text
Sex              M
Length           0.455
Diameter         0.365
Height           0.095
Whole_weight     0.514
Shucked_weight   0.2245
Viscera_weight   0.101
Shell_weight     0.15
```

Example prediction returned during endpoint testing:

```text
10.2224
```

The value represents the predicted number of rings.

Because the endpoint uses multiple deployments and traffic splitting, the exact prediction can depend on which deployed model handles the request.

---

# Security

The Azure ML endpoint uses key-based authentication.

Endpoint credentials should always be treated as secrets.

The following security practices should be followed:

- Never commit Azure ML endpoint keys to GitHub.
- Never hardcode endpoint credentials in `app.py`.
- Store credentials using Streamlit Secrets for public deployment.
- Use environment variables or local secret files during development.
- Add secret files to `.gitignore`.
- Rotate endpoint keys if they are accidentally exposed.
- Do not include credentials in screenshots, documentation, or source code.

Example `.gitignore`:

```text
.env
.streamlit/secrets.toml
__pycache__/
*.pyc
venv/
.venv/
```

---

# Error Handling

The Streamlit application should handle common endpoint failures.

Example:

```python
try:
    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

except requests.exceptions.Timeout:
    st.error("The Azure ML endpoint took too long to respond.")

except requests.exceptions.RequestException as e:
    st.error(f"Prediction request failed: {e}")

except Exception as e:
    st.error(f"Unexpected error: {e}")
```

This prevents the application from failing silently when the endpoint is unavailable or the request is invalid.

---

# Troubleshooting

## Azure ML Endpoint Returns an Error

Check:

- Endpoint status
- Deployment status
- Traffic allocation
- Endpoint authentication
- Request JSON format
- Model input columns
- Deployment logs

The input columns must match the model signature:

```text
Sex
Length
Diameter
Height
Whole_weight
Shucked_weight
Viscera_weight
Shell_weight
```

---

## Streamlit Cannot Connect to Azure ML

Check:

```text
AZURE_ENDPOINT_URL
AZURE_ENDPOINT_KEY
```

Also verify:

- Azure endpoint is running.
- Public network access is enabled.
- Endpoint authentication key is valid.
- The request payload is correctly formatted.
- The Streamlit application has internet access.

---

## Model Prediction Fails

Check the Azure ML deployment logs.

Common causes include:

- Incorrect input schema.
- Missing input columns.
- Incorrect data types.
- Invalid JSON format.
- Deployment configuration errors.
- Model environment issues.

---

## Streamlit Secrets Error

If Streamlit reports that a secret is missing, verify:

```text
.streamlit/secrets.toml
```

for local development.

For Streamlit Community Cloud, verify the application's Secrets configuration.

The required keys are:

```text
AZURE_ENDPOINT_URL
AZURE_ENDPOINT_KEY
```

---

# Future Improvements

Possible improvements to the project include:

- Add automated model retraining.
- Add automated model evaluation pipelines.
- Add model monitoring.
- Add data drift detection.
- Add prediction logging.
- Add Application Insights monitoring.
- Add CI/CD using GitHub Actions.
- Add automated Azure ML deployment pipelines.
- Add authentication for the Streamlit application.
- Add prediction history.
- Add improved input validation.
- Add model performance monitoring.
- Monitor the performance of each deployed model independently.
- Evaluate the two model deployments using production inference data.
- Add dashboards for endpoint traffic and latency.
- Add automated rollback for problematic deployments.

---

# Limitations

The current project has several limitations:

- The application depends on the availability of the Azure ML endpoint.
- Endpoint authentication is required for prediction requests.
- Prediction quality depends on the trained models and input data.
- The project currently uses a fixed traffic split.
- The Streamlit frontend depends on the Azure ML REST endpoint.
- Public deployment requires secure handling of Azure credentials.
- The model predicts the number of rings rather than directly predicting age in years.
- Real-time performance depends on Azure ML endpoint availability and compute configuration.
- The traffic split does not by itself represent a controlled statistical comparison of model performance.
- The current project does not include automated model retraining or continuous deployment.

---

# Key Project Highlights

This project demonstrates practical machine learning and MLOps concepts including:

- Multi-model experimentation
- Regression model evaluation
- Ensemble learning
- LightGBM
- XGBoost
- Azure Machine Learning
- MLflow model registration
- Managed online endpoints
- Real-time inference
- Multiple deployments behind one endpoint
- Traffic splitting
- REST API integration
- Streamlit application development
- Public cloud deployment
- Secure secret management
- End-to-end ML deployment

---

# Learning Outcomes

By completing this project, the following concepts are demonstrated:

```text
Machine Learning
       |
       v
Model Evaluation
       |
       v
Model Selection
       |
       v
Model Registration
       |
       v
Model Deployment
       |
       v
Real-Time Inference
       |
       v
Traffic Management
       |
       v
API Integration
       |
       v
Frontend Development
       |
       v
Cloud Deployment
```

The project therefore covers both the machine learning side and the deployment side of a complete ML application.

---

# Conclusion

The Abalone Age Prediction project demonstrates a complete machine learning deployment lifecycle using Azure Machine Learning.

Multiple regression models were evaluated in Azure ML, with 23 models displayed during experimentation. Two selected models, Voting Ensemble and MaxAbsScaler + LightGBM, were prepared for deployment.

The models were registered and deployed to the same Azure ML Managed Online Endpoint. Traffic was configured as:

```text
Voting Ensemble          80%
MaxAbsScaler + LightGBM  20%
```

The endpoint provides real-time inference through a REST API.

A Streamlit application was developed as the user-facing layer. Users can enter the physical measurements of an abalone, submit the request, and receive the prediction from the Azure ML endpoint.

The Streamlit application can then be deployed using Streamlit Community Cloud, making the machine learning solution accessible through a public web interface.

The final project combines:

```text
Machine Learning
        +
Model Experimentation
        +
Azure Machine Learning
        +
MLflow
        +
Real-Time Model Deployment
        +
Traffic Splitting
        +
REST API
        +
Streamlit
        +
Cloud Deployment
```

This provides an end-to-end example of taking machine learning models from experimentation to real-time cloud inference and a public user-facing application.

---

# License

This project is intended for educational, portfolio, and demonstration purposes.

Refer to the repository license for the applicable licensing terms.
