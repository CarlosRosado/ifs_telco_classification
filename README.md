# Telco Customer Churn Prediction: End-to-End Solution

This repository contains a complete solution for predicting customer churn problem in IFS Challenge. The solution includes data preprocessing, model training, deployment as a REST API, and monitoring using Prometheus.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Why Random Forest and Techniques Used](#random-forest-and-techniques-used)
4. [MLflow, Docker, and Kubernetes](#mlflow-docker-and-kubernetes)
5. [Liveness and Readiness Probes](#liveness-and-readiness-probes)
6. [Prometheus for Monitoring](#prometheus-for-monitoring)
7. [How to Run the Project](#how-to-run-the-project)
8. [Makefile Instructions](#makefile-instructions)
9. [Example API Usage](#example-api-usage)
10. [Retention Incentive Examples](#retention-incentive-examples)
11. [Live Demo Deliverables](#live-demo-deliverables)
12. [Future Improvements](#future-improvements)

---

## Problem Statement

The goal of this project is to predict customer churn using the `telco_customer_churn_data.csv` dataset. The dataset contains customer demographics, subscription details, billing methods, service usage, and churn status. Additionally, the project includes a bonus task to generate personalized retention incentives using a pre-trained language model.

---

## Solution Overview

1. **Data Exploration and Preprocessing**:
   - Cleaned and prepared the dataset for modeling.
   - Handled missing values and engineered features for better model performance.

2. **Model Development**:
   - Trained a `RandomForestClassifier` to predict customer churn.
   - Used SMOTE to handle class imbalance and `RandomizedSearchCV` for hyperparameter tuning.

3. **Model Deployment**:
   - Deployed the trained model as a REST API using FastAPI.
   - Integrated Prometheus for monitoring and added liveness/readiness probes for Kubernetes.

4. **Retention Incentive Generation**:
   - Used GPT-2 to generate personalized retention incentives based on churn predictions.

---

## Random Forest and Techniques Used

### Why Random Forest?
- **Robustness**: Random Forest is a robust ensemble learning method that combines multiple decision trees to reduce overfitting and improve generalization.
- **Feature Importance**: It provides insights into feature importance, helping us understand which factors contribute most to customer churn.
- **Ease of Use**: It works well with both categorical and numerical data, making it suitable for the telco churn dataset.

### Techniques Used in `train_mlflow.py`:
1. **SMOTE (Synthetic Minority Oversampling Technique)**:
   - Addressed the class imbalance in the dataset by oversampling the minority class (churned customers).
   - Improved the model's ability to predict churned customers.

2. **RandomizedSearchCV**:
   - Performed hyperparameter tuning to find the best combination of parameters for the Random Forest model.
   - Reduced computational cost compared to a grid search.

3. **Precision-Recall Curve**:
   - Evaluated the model's performance using precision-recall curves, which are more informative for imbalanced datasets.

4. **MLflow Integration**:
   - Tracked experiments, hyperparameters, and model performance metrics.
   - Simplified model versioning and deployment.

---

## MLflow, Docker, and Kubernetes

### MLflow:
- **Experiment Tracking**: MLflow tracks experiments, including hyperparameters, metrics, and artifacts, making it easier to compare models.
- **Model Registry**: It provides a centralized model registry for versioning and deployment.
- **Ease of Deployment**: MLflow simplifies the deployment of models as REST APIs.

### Docker:
- **Portability**: Docker ensures that the application runs consistently across different environments.
- **Isolation**: It isolates the application and its dependencies, reducing conflicts.
- **Ease of Deployment**: Docker images can be easily deployed to Kubernetes or other container orchestration platforms.

### Kubernetes:
- **Scalability**: Kubernetes allows the application to scale horizontally to handle increased traffic.
- **Resilience**: It provides self-healing capabilities through liveness and readiness probes.
- **Automation**: Kubernetes automates deployment, scaling, and management of containerized applications.

---

## Why Liveness and Readiness Probes

### Liveness Probe
- **Purpose**: Ensures the application is running and responsive.
- **How It Works**: Kubernetes periodically checks the `/healthz` endpoint. If the probe fails, Kubernetes restarts the container.

### Readiness Probe
- **Purpose**: Ensures the application is ready to serve traffic.
- **How It Works**: Kubernetes checks the `/readyz` endpoint. If the probe fails, Kubernetes removes the pod from the service's load balancer until it becomes ready.

### Benefits
- **Improved Resilience**: Automatically restarts unhealthy containers.
- **Traffic Management**: Ensures only healthy pods receive traffic.

---

## Why Prometheus for Monitoring

- **Real-Time Metrics**: Prometheus collects real-time metrics, such as request counts and response times, from the REST API.
- **Alerting**: It can trigger alerts based on predefined thresholds, such as high latency or error rates.
- **Integration**: Prometheus integrates seamlessly with Kubernetes and FastAPI.

In this project, Prometheus tracks:
1. Total number of requests (`request_count`).
2. Time spent processing requests (`request_processing_seconds`).

---

## How to Run the Project

### Prerequisites
1. Install Python 3.8 or higher.
2. Install Docker and Kubernetes (e.g., Minikube for local Kubernetes).

### Steps to Run
1. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory with the following variables:
     ```env
     DATA_URL=<URL to dataset>
     LOCAL_DATA_PATH=./data/telco_customer_churn_data.csv
     MODEL_OUTPUT_PATH=./models/churn_model.pkl
     MLFLOW_TRACKING_URI=http://localhost:5001
     OPENAPI_FILE_PATH=./app/prediction-openapi.yaml
     ```

2. **Train the Model**:
   - Run the training script:
     ```bash
     python src/model/train_mlflow.py
     ```

3. **Build and Run Docker Container**:
   - Build the Docker image:
     ```bash
     docker build -t telco-churn-api .
     ```
   - Run the container:
     ```bash
     docker run -p 8000:8000 telco-churn-api
     ```

4. **Deploy to Kubernetes**:
   - Apply the Kubernetes manifests:
     ```bash
     kubectl apply -f k8s/
     ```

5. **Start Prometheus**:
   - Deploy Prometheus to Kubernetes:
     ```bash
     kubectl apply -f prometheus.yaml
     ```

6. **Test the API**:
   - Use tools like `curl` or Postman to test the `/predict` and `/specifications` endpoints.

---

## Makefile Instructions

Run the following commands from the project root:

### 1. Make clean:
* Purpose: Cleans up Docker containers and Kubernetes resources.
* Usage:

    ```bash
        make clean
    ```

 ### 2. Make all
* Purpose: Runs the entire pipeline, including:

    -  Installing dependencies.
    - Training the model.
    - Building the Docker image.
    - Running the Docker container.
* Usage:

    ```bash
        make all
    ```
 ### 2. Make test
* Purpose: Runs all unit tests using pytest.
* Usage:

    ```bash
        make test
    ```
---

## Example API Usage

/predict Endpoint

* Request:
    ```bash
        {
        "tenure": "12",
        "MonthlyCharges": "70.35",
        "TotalCharges": "843.75",
        "gender": "1",
        "Partner": "0",
        "Dependents": "0",
        "PhoneService": "1",
        "PaperlessBilling": "1",
        "InternetService_Fiber_optic": "1",
        "InternetService_No": "0",
        "Contract_Month_to_month": "1",
        "Contract_One_year": "0",
        "PaymentMethod_Electronic_check": "1"
        }
     ```
* Response:
    ```bash
        {
        "prediction": "Likely to churn",
        "retention_incentive": "Offer a 10% discount for the next 6 months."
        }
     ```

---


## Retention Incentive Examples

Here are a few examples of retention incentives generated using GPT-2, demonstrating the effectiveness of prompt engineering in tailoring the responses to individual customer contexts:

1. **Customer with High Monthly Charges**:
   - **Input**: "Customer with high monthly charges and likely to churn."
   - **Generated Incentive**: "Offer a 15% discount on their monthly bill for the next 6 months."

2. **Customer with Long Tenure**:
   - **Input**: "Loyal customer with a tenure of 5 years but likely to churn."
   - **Generated Incentive**: "Provide a free upgrade to a premium plan for 3 months as a token of appreciation."

3. **Customer with Payment Issues**:
   - **Input**: "Customer with frequent payment delays and likely to churn."
   - **Generated Incentive**: "Offer a flexible payment plan with no late fees for the next 3 months."

---

## Live Demo Deliverables

In the following example, generate a response based in the following curl input:

```bash
    curl -X POST "http://0.0.0.0:30081/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "tenure": 12,
        "MonthlyCharges": 70.35,
        "TotalCharges": 843.0,
        "gender": 1,
        "Partner": 0,
        "Dependents": 0,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "InternetService_Fiber_optic": 1,
        "InternetService_No": 0,
        "Contract_Month_to_month": 1,
        "Contract_One_year": 0,
        "PaymentMethod_Electronic_check": 1
    }'
```
Obtain the following response:
```bash
    {"prediction":"Likely to churn","retention_incentive":"Customer Satisfaction: 3.5 months"}% 
```

---

## Future Improvements

* Experiment with advanced models like XGBoost or LightGBM.
* Fine-tune GPT-2 for domain-specific retention incentive generation.
* Add Grafana dashboards for better visualization of Prometheus metrics.
