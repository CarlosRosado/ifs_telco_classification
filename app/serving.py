from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import os
import pandas as pd
from pydantic import BaseModel
from prometheus_client import start_http_server, Summary, Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi.openapi.utils import get_openapi
import yaml
from transformers import pipeline


# Set the MLflow Tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

# Initialize FastAPI app
app = FastAPI()

# Create a metric to track in Prometheus
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total number of requests')


# Load the model at startup
MODEL_NAME = os.getenv("MODEL_NAME", "ifs_Random_Forest")  # Default model name
MODEL_VERSION = os.getenv("MODEL_VERSION")  # Optional: Specify a version

OPENAPI_FILE_PATH = os.getenv("OPENAPI_FILE_PATH", "/app/app/prediction-openapi.yaml")

try:
    if MODEL_VERSION:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    else:
        model_uri = f"models:/{MODEL_NAME}/latest"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model '{MODEL_NAME}' loaded successfully from URI: {model_uri}")
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    model = None

# Load the Hugging Face model for text generation
try:
    # Load the smaller GPT-2 model
    generator_pipeline = pipeline("text-generation", model="gpt2")
    print("Text generation model loaded successfully.")
except Exception as e:
    print(f"Error loading text generation model: {e}")
    generator_pipeline = None

# Define the input schema using Pydantic
class PredictionInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: int
    Partner: int
    Dependents: int
    PhoneService: int
    PaperlessBilling: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_Month_to_month: int
    Contract_One_year: int
    PaymentMethod_Electronic_check: int

# Define the prediction endpoint
@app.post("/predict")
@REQUEST_TIME.time()
@REQUEST_COUNT.count_exceptions()
def predict(input_data: PredictionInput):
    """
    Predict churn and generate a personalized retention incentive.

    Args:
        input_data (PredictionInput): Customer details.

    Returns:
        dict: Churn prediction and retention incentive.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Cannot make predictions.")

    if generator_pipeline is None:
        raise HTTPException(status_code=500, detail="Text generation model is not loaded.")

    try:
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Align input features with the model's expected schema
        required_features = [
            "tenure", "MonthlyCharges", "TotalCharges", "gender", "Partner", "Dependents",
            "PhoneService", "PaperlessBilling", "InternetService_Fiber optic", "InternetService_No",
            "Contract_Month-to-month", "Contract_One year", "PaymentMethod_Electronic check",
            "SeniorCitizen", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "tenure_years", "avg_monthly_charge",
            "is_high_value_customer", "has_multiple_services", "MultipleLines_No",
            "MultipleLines_No phone service", "MultipleLines_Yes", "InternetService_DSL",
            "Contract_Two year", "PaymentMethod_Bank transfer (automatic)",
            "PaymentMethod_Credit card (automatic)", "PaymentMethod_Mailed check"
        ]

        # Rename input columns to match the model's schema
        input_df.rename(columns={
            "InternetService_Fiber_optic": "InternetService_Fiber optic",
            "Contract_Month_to_month": "Contract_Month-to-month",
            "Contract_One_year": "Contract_One year",
            "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check"
        }, inplace=True)

        # Add missing features with default values
        for feature in required_features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value for missing features

        # Drop extra features not required by the model
        input_df = input_df[required_features]

        # Make predictions
        prediction = model.predict(input_df)[0]

        churn_prediction = "Likely to churn" if prediction == 1 else "Not likely to churn"

        # Generate retention incentive if the customer is predicted to churn
        retention_incentive = None
        if prediction == 1:  # 1 indicates churn
            prompt = f"""
                    The customer is likely to churn. Here are the customer's details:
                    - Tenure: {input_data.tenure} months
                    - Monthly Charges: ${input_data.MonthlyCharges}
                    - Total Charges: ${input_data.TotalCharges}

                As a customer retention specialist, provide a specific retention incentive. Include actionable offers, discounts, or benefits to retain the customer. Be concise and specific.
                """

            try:
                response = generator_pipeline(prompt, max_new_tokens=50, num_return_sequences=1, temperature=0.7)
                retention_incentive = response[0]["generated_text"].strip()

                # Post-process the generated text to remove unnecessary characters
                retention_incentive = retention_incentive.replace(prompt.strip(), "").strip()
                retention_incentive = retention_incentive.replace("\n", " ").replace("-", "").strip()
            except Exception as e:
                retention_incentive = "Unable to generate a retention incentive at this time."
        else:
            retention_incentive = "Customer is not likely to churn. No retention incentive needed."

        # Return the prediction and retention incentive
        return {
            "prediction": churn_prediction,
            "retention_incentive": retention_incentive
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

@app.get("/")
def read_root():
    """
    Root endpoint for the API.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the IFS Churn Prediction API"}

@app.get("/metrics")
def metrics():
    """
    Endpoint for returning the current metrics of the service.

    Returns:
        Response: The current metrics in Prometheus format.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Load OpenAPI specification from YAML file
with open(OPENAPI_FILE_PATH, "r") as f:
    openapi_spec = yaml.safe_load(f)

@app.get("/specifications")
def get_specifications():
    """
    Endpoint for returning the OpenAPI specifications.

    Returns:
        dict: The OpenAPI specifications.
    """
    return openapi_spec

# Start up the server to expose the metrics.
start_http_server(9091)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving:app", host="0.0.0.0", port=8001)