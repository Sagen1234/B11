from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import pickle
import io
import base64
import json
import logging
import os
from pathlib import Path
import uuid
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Fraud Detection System",
    description="A comprehensive fraud detection system for credit card transactions",
    version="1.0.0"
)

# Global variables for model and preprocessing components
model_data = {
    'model': None,
    'scaler': None,
    'encoders': None,
    'feature_columns': None,
    'is_trained': False
}

# Allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Data Models
class TransactionBase(BaseModel):
    amt: float = Field(..., description="Transaction amount")
    category: str = Field(..., description="Transaction category")
    merchant: str = Field(..., description="Merchant name")
    gender: str = Field(..., description="Customer gender")
    job: str = Field(..., description="Customer job")


class Transaction(TransactionBase):
    trans_date_trans_time: Optional[str] = Field(None, description="Transaction datetime")
    cc_num: Optional[str] = Field(None, description="Credit card number")
    first: Optional[str] = Field(None, description="First name")
    last: Optional[str] = Field(None, description="Last name")
    street: Optional[str] = Field(None, description="Street address")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State")
    zip: Optional[str] = Field(None, description="ZIP code")
    lat: Optional[float] = Field(None, description="Latitude")
    long: Optional[float] = Field(None, description="Longitude")
    city_pop: Optional[int] = Field(None, description="City population")
    unix_time: Optional[int] = Field(None, description="Unix timestamp")
    merch_lat: Optional[float] = Field(None, description="Merchant latitude")
    merch_long: Optional[float] = Field(None, description="Merchant longitude")


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    transaction_id: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


class TrainingResponse(BaseModel):
    success: bool
    accuracy: float
    training_time: float
    model_id: str
    message: str


class AnalysisResponse(BaseModel):
    fraud_statistics: Dict[str, Any]
    trends: Dict[str, Any]
    insights: List[str]


class ComplianceReport(BaseModel):
    report_id: str
    generated_at: str
    period: str
    total_transactions: int
    fraud_cases: int
    fraud_rate: float
    actions_taken: List[str]
    regulatory_compliance: Dict[str, Any]


# Utility Functions
def load_csv_robust(filename: str) -> Optional[pd.DataFrame]:
    """Robust CSV loading function that handles parsing errors"""
    logger.info(f"Loading {filename}...")

    methods = [
        lambda: pd.read_csv(filename, on_bad_lines='skip'),
        lambda: pd.read_csv(filename, error_bad_lines=False, warn_bad_lines=True),
        lambda: pd.read_csv(filename, quoting=3, on_bad_lines='skip'),
        lambda: pd.read_csv(filename, engine='python', on_bad_lines='skip'),
    ]

    for i, method in enumerate(methods, 1):
        try:
            data = method()
            logger.info(f"Successfully loaded {filename} using method {i}")
            return data
        except Exception as e:
            logger.warning(f"Method {i} failed: {str(e)[:100]}...")
            continue

    logger.error(f"Failed to load {filename}")
    return None


def preprocess_data(data: pd.DataFrame, encoders=None, scaler=None, fit_transformers=True) -> tuple:
    """Preprocess the data with consistent transformations"""
    data = data.copy()

    # Convert date columns if they exist
    if 'trans_date_trans_time' in data.columns:
        data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"], errors='coerce')
    if 'dob' in data.columns:
        data["dob"] = pd.to_datetime(data["dob"], errors='coerce')

    # Drop unnecessary columns
    columns_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num',
                       'trans_date_trans_time']
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    if existing_columns_to_drop:
        data.drop(columns=existing_columns_to_drop, inplace=True)

    # Drop rows with missing values
    original_shape = data.shape
    data.dropna(inplace=True)
    logger.info(f"Dropped {original_shape[0] - data.shape[0]} rows with missing values")

    # Encode categorical variables
    categorical_columns = ['merchant', 'category', 'gender', 'job']

    if encoders is None:
        encoders = {}

    for col in categorical_columns:
        if col in data.columns:
            if fit_transformers:
                encoders[col] = LabelEncoder()
                data[col] = encoders[col].fit_transform(data[col].astype(str))
            else:
                # Handle unseen categories
                try:
                    data[col] = data[col].astype(str)
                    mask = ~data[col].isin(encoders[col].classes_)
                    if mask.any():
                        most_frequent = encoders[col].classes_[0]
                        data.loc[mask, col] = most_frequent

                    data[col] = encoders[col].transform(data[col])
                except Exception as e:
                    logger.error(f"Error encoding {col}: {e}")

    return data, encoders


def get_risk_level(probability: float) -> str:
    """Determine risk level based on fraud probability"""
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.6:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    elif probability >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"


def create_plot_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return img_str


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fraud Detection System API",
        "version": "1.0.0",
        "endpoints": {
            "train": "/train - Train the fraud detection model",
            "predict": "/predict - Predict single transaction",
            "predict_batch": "/predict/batch - Predict multiple transactions",
            "analyze": "/analyze - Analyze fraud trends and insights",
            "report": "/report/compliance - Generate compliance report",
            "model_status": "/model/status - Check model status"
        }
    }


@app.get("/model/status")
async def get_model_status():
    """Get current model status"""
    return {
        "is_trained": model_data['is_trained'],
        "model_type": "Logistic Regression" if model_data['is_trained'] else None,
        "features": model_data['feature_columns'] if model_data['feature_columns'] else None
    }


@app.post("/train", response_model=TrainingResponse)
async def train_model(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Train the fraud detection model with uploaded CSV data"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Load and preprocess data
        train_data = load_csv_robust(tmp_file_path)
        os.unlink(tmp_file_path)  # Clean up temp file

        if train_data is None:
            raise HTTPException(status_code=400, detail="Failed to load training data")

        if 'is_fraud' not in train_data.columns:
            raise HTTPException(status_code=400, detail="'is_fraud' column not found in training data")

        # Preprocess data
        train_data, encoders = preprocess_data(train_data, fit_transformers=True)

        # Prepare features and target
        X = train_data.drop(columns=["is_fraud"])
        y = train_data["is_fraud"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train model
        start_time = time.time()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time

        # Store model and preprocessing components
        model_id = str(uuid.uuid4())
        model_data.update({
            'model': model,
            'scaler': scaler,
            'encoders': encoders,
            'feature_columns': list(X.columns),
            'is_trained': True
        })

        logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")

        return TrainingResponse(
            success=True,
            accuracy=accuracy,
            training_time=training_time,
            model_id=model_id,
            message=f"Model trained successfully with {accuracy:.4f} accuracy"
        )

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: TransactionBase):
    """Predict fraud for a single transaction"""
    if not model_data['is_trained']:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")

    try:
        # Convert transaction to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])

        # Preprocess the data
        processed_df, _ = preprocess_data(df, encoders=model_data['encoders'], fit_transformers=False)

        # Ensure all required features are present
        for col in model_data['feature_columns']:
            if col not in processed_df.columns:
                processed_df[col] = 0  # Default value for missing features

        # Reorder columns to match training data
        processed_df = processed_df[model_data['feature_columns']]

        # Scale features
        X_scaled = model_data['scaler'].transform(processed_df)

        # Make prediction
        prediction = model_data['model'].predict(X_scaled)[0]
        probability = model_data['model'].predict_proba(X_scaled)[0][1]

        transaction_id = str(uuid.uuid4())

        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(probability),
            risk_level=get_risk_level(probability),
            transaction_id=transaction_id
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Predict fraud for multiple transactions from CSV file"""
    if not model_data['is_trained']:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Load data
        data = load_csv_robust(tmp_file_path)
        os.unlink(tmp_file_path)  # Clean up temp file

        if data is None:
            raise HTTPException(status_code=400, detail="Failed to load batch data")

        # Preprocess data
        processed_data, _ = preprocess_data(data, encoders=model_data['encoders'], fit_transformers=False)

        # Ensure all required features are present
        for col in model_data['feature_columns']:
            if col not in processed_data.columns:
                processed_data[col] = 0

        # Reorder columns
        processed_data = processed_data[model_data['feature_columns']]

        # Scale features
        X_scaled = model_data['scaler'].transform(processed_data)

        # Make predictions
        predictions = model_data['model'].predict(X_scaled)
        probabilities = model_data['model'].predict_proba(X_scaled)[:, 1]

        # Create response
        prediction_list = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            prediction_list.append(PredictionResponse(
                is_fraud=bool(pred),
                fraud_probability=float(prob),
                risk_level=get_risk_level(prob),
                transaction_id=str(uuid.uuid4())
            ))

        # Calculate summary statistics
        total_transactions = len(predictions)
        fraud_count = sum(predictions)
        fraud_rate = fraud_count / total_transactions if total_transactions > 0 else 0

        summary = {
            "total_transactions": total_transactions,
            "fraud_detected": int(fraud_count),
            "fraud_rate": fraud_rate,
            "average_fraud_probability": float(np.mean(probabilities)),
            "high_risk_transactions": sum(1 for p in probabilities if p >= 0.6)
        }

        return BatchPredictionResponse(
            predictions=prediction_list,
            summary=summary
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_fraud_trends(file: UploadFile = File(...)):
    """Analyze fraud trends and patterns in transaction data"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Load data
        data = load_csv_robust(tmp_file_path)
        os.unlink(tmp_file_path)

        if data is None:
            raise HTTPException(status_code=400, detail="Failed to load analysis data")

        if 'is_fraud' not in data.columns:
            raise HTTPException(status_code=400, detail="'is_fraud' column required for analysis")

        # Basic fraud statistics
        total_transactions = len(data)
        fraud_cases = data['is_fraud'].sum()
        fraud_rate = fraud_cases / total_transactions

        fraud_statistics = {
            "total_transactions": total_transactions,
            "fraud_cases": int(fraud_cases),
            "legitimate_transactions": int(total_transactions - fraud_cases),
            "fraud_rate": fraud_rate,
            "fraud_percentage": fraud_rate * 100
        }

        # Analyze trends by different dimensions
        trends = {}

        # Fraud by category
        if 'category' in data.columns:
            category_fraud = data.groupby('category')['is_fraud'].agg(['count', 'sum', 'mean']).round(4)
            trends['by_category'] = category_fraud.to_dict('index')

        # Fraud by amount ranges
        if 'amt' in data.columns:
            data['amount_range'] = pd.cut(data['amt'], bins=[0, 50, 100, 500, 1000, float('inf')],
                                          labels=['0-50', '51-100', '101-500', '501-1000', '1000+'])
            amount_fraud = data.groupby('amount_range')['is_fraud'].agg(['count', 'sum', 'mean']).round(4)
            trends['by_amount'] = amount_fraud.to_dict('index')

        # Fraud by gender
        if 'gender' in data.columns:
            gender_fraud = data.groupby('gender')['is_fraud'].agg(['count', 'sum', 'mean']).round(4)
            trends['by_gender'] = gender_fraud.to_dict('index')

        # Generate insights
        insights = []

        if fraud_rate > 0.05:
            insights.append(f"High fraud rate detected: {fraud_rate * 100:.2f}% - Consider immediate action")
        elif fraud_rate > 0.02:
            insights.append(f"Moderate fraud rate: {fraud_rate * 100:.2f}% - Monitor closely")
        else:
            insights.append(f"Low fraud rate: {fraud_rate * 100:.2f}% - Within acceptable range")

        if 'amt' in data.columns:
            avg_fraud_amount = data[data['is_fraud'] == 1]['amt'].mean()
            avg_legit_amount = data[data['is_fraud'] == 0]['amt'].mean()
            if avg_fraud_amount > avg_legit_amount * 1.5:
                insights.append("Fraudulent transactions tend to have higher amounts")
            elif avg_fraud_amount < avg_legit_amount * 0.5:
                insights.append("Fraudulent transactions tend to have lower amounts")

        if 'category' in data.columns:
            high_risk_categories = data.groupby('category')['is_fraud'].mean().sort_values(ascending=False).head(3)
            insights.append(f"Highest risk categories: {', '.join(high_risk_categories.index.tolist())}")

        return AnalysisResponse(
            fraud_statistics=fraud_statistics,
            trends=trends,
            insights=insights
        )

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/report/compliance", response_model=ComplianceReport)
async def generate_compliance_report(
        file: UploadFile = File(...),
        period: str = "monthly",
        include_actions: bool = True
):
    """Generate regulatory compliance and audit report"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Load data
        data = load_csv_robust(tmp_file_path)
        os.unlink(tmp_file_path)

        if data is None:
            raise HTTPException(status_code=400, detail="Failed to load report data")

        if 'is_fraud' not in data.columns:
            raise HTTPException(status_code=400, detail="'is_fraud' column required for compliance report")

        # Calculate metrics
        total_transactions = len(data)
        fraud_cases = data['is_fraud'].sum()
        fraud_rate = fraud_cases / total_transactions if total_transactions > 0 else 0

        # Actions taken (simulated based on fraud cases)
        actions_taken = []
        if include_actions:
            if fraud_cases > 0:
                actions_taken.extend([
                    f"Flagged {fraud_cases} suspicious transactions for investigation",
                    "Enhanced monitoring protocols activated",
                    "Customer notifications sent for high-risk transactions"
                ])

                if fraud_rate > 0.05:
                    actions_taken.append("Emergency fraud response team activated")
                    actions_taken.append("Additional verification procedures implemented")

        # Regulatory compliance metrics
        regulatory_compliance = {
            "pci_dss_compliance": True,  # Simulated
            "aml_reporting": {
                "suspicious_transactions_reported": int(fraud_cases),
                "reporting_threshold_met": fraud_rate > 0.01
            },
            "data_protection": {
                "data_encrypted": True,
                "access_logs_maintained": True,
                "retention_policy_followed": True
            },
            "audit_trail": {
                "all_transactions_logged": True,
                "investigation_records_complete": fraud_cases > 0,
                "documentation_compliant": True
            }
        }

        report_id = str(uuid.uuid4())
        generated_at = datetime.datetime.now().isoformat()

        return ComplianceReport(
            report_id=report_id,
            generated_at=generated_at,
            period=period,
            total_transactions=total_transactions,
            fraud_cases=int(fraud_cases),
            fraud_rate=fraud_rate,
            actions_taken=actions_taken,
            regulatory_compliance=regulatory_compliance
        )

    except Exception as e:
        logger.error(f"Compliance report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/visualizations/fraud-distribution")
async def get_fraud_distribution_chart(file: UploadFile = File(...)):
    """Generate fraud distribution visualization"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Load data
        data = load_csv_robust(tmp_file_path)
        os.unlink(tmp_file_path)

        if data is None:
            raise HTTPException(status_code=400, detail="Failed to load visualization data")

        if 'is_fraud' not in data.columns:
            raise HTTPException(status_code=400, detail="'is_fraud' column required for visualization")

        # Create fraud distribution chart
        fraud_counts = data["is_fraud"].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(fraud_counts, labels=["Legitimate", "Fraud"], autopct="%0.1f%%", startangle=90)
        ax.set_title("Transaction Fraud Distribution")

        # Convert to base64
        chart_base64 = create_plot_base64(fig)

        return {
            "chart_data": chart_base64,
            "fraud_statistics": {
                "total_transactions": len(data),
                "fraud_cases": int(fraud_counts[1]) if 1 in fraud_counts else 0,
                "legitimate_cases": int(fraud_counts[0]) if 0 in fraud_counts else 0,
                "fraud_percentage": (fraud_counts[1] / fraud_counts.sum() * 100) if 1 in fraud_counts else 0
            }
        }

    except Exception as e:
        logger.error(f"Visualization generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_trained": model_data['is_trained']
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)