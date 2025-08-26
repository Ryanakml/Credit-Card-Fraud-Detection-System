import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn

# 1. Database Connection and Data Loading
def load_data_from_db():
    """Loads processed data from the PostgreSQL database."""
    print("STARTING DATA LOADING PROCESS")
    
    print("Setting up database connection...")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "fraud_detection")
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    print(f"Connecting to database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    try:
        engine = create_engine(DATABASE_URL)
        print("Database connection established successfully!")
        print("Executing query: SELECT * FROM processed_transactions")
        df = pd.read_sql('SELECT * FROM processed_transactions', engine)
        print(f"Data loaded successfully from database.")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")

        return None

# 2. Model Training and Evaluation Logic
def train_and_evaluate():
    """
    Main function to train models and track experiments with MLflow.
    """
    print("STARTING FRAUD DETECTION MODEL TRAINING PIPELINE")
    
    # Load data
    df = load_data_from_db()
    if df is None:
        print("Failed to load data. Exiting...")
        return

    print("DATA PREPARATION PHASE")
    
    # Define features (X) and target (y)
    print("Preparing features and target variable...")
    X = df.drop(columns=['CardID', 'Class'], axis=1) # CardID is for grouping, not a feature
    y = df['Class']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    print(f"Fraud rate: {y.mean():.4f}")

    # Data Splitting (CRITICAL: Split before SMOTE)
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training fraud rate: {y_train.mean():.4f}")
    print(f"Test fraud rate: {y_test.mean():.4f}")
    
    print("\nDATA PREPROCESSING PHASE")
    
    # Preprocessing
    # Note: We scale 'Amount' and other engineered features. PCA features are already scaled.
    # For this project, we'll build a pipeline that scales all features for consistency.
    print("Creating preprocessing pipeline...")
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # handle missing values
        ('scaler', RobustScaler())                     # scale features
    ])

    # Fit preprocessor on training data and transform both train and test
    print("Fitting preprocessor on training data...")
    X_train_scaled = preprocessor.fit_transform(X_train)
    print("Transforming test data...")
    X_test_scaled = preprocessor.transform(X_test)
    
    # Convert to DataFrame to preserve feature names for LightGBM
    feature_names = X.columns.tolist()
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    print("Data preprocessing completed!")
    
    # Save the fitted preprocessor
    print("Saving preprocessor to disk...")
    os.makedirs("processor", exist_ok=True)
    with open('processor/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Preprocessor saved as 'preprocessor.pkl'")

    print("\nDATA BALANCING PHASE")
    
    # Handle Imbalance with SMOTE (CRITICAL: Apply only to training data)
    print("Applying SMOTE to the training data...")
    print(f"Original training set shape: {X_train_scaled.shape}")
    print(f"Original class distribution: {np.bincount(y_train)}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Convert back to DataFrame with feature names
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=feature_names)
    
    print(f"SMOTE applied successfully!")
    print(f"New training set shape: {X_train_resampled.shape}")
    print(f"New class distribution: {np.bincount(y_train_resampled)}")

    print("\nMODEL TRAINING PHASE")

    # Model Definitions
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000, solver='liblinear'),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, force_row_wise=True)
    }

    print(f"Models to train: {list(models.keys())}")
    print(f"Total models: {len(models)}")

    # MLflow Experiment Tracking
    print("\nSetting up MLflow experiment...")
    mlflow.set_experiment("Credit Card Fraud Detection")
    print("MLflow experiment initialized!")

    print("TRAINING AND EVALUATION PROCESS")

    for i, (name, model) in enumerate(models.items(), 1):
        print(f"\n[{i}/{len(models)}] TRAINING: {name}")
        
        with mlflow.start_run(run_name=name) as run:
            print(f"Starting MLflow run for {name}...")
            print(f"Run ID: {run.info.run_id}")
            
            # Train model
            print(f"Training {name} model...")
            model.fit(X_train_resampled, y_train_resampled)
            print(f"{name} training completed!")
            
            # Make predictions
            print("Making predictions on test set...")
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            print("Predictions completed!")
            
            # Calculate metrics
            print("Calculating evaluation metrics...")
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auprc = average_precision_score(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print("Metrics calculated!")
            
            # Log parameters (can be done automatically or manually)
            print("Logging model parameters to MLflow...")
            mlflow.log_params(model.get_params())
            
            # Log metrics
            print("Logging metrics to MLflow...")
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auprc", auprc)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Log model with signature and input example
            print("Logging model to MLflow...")
            # Create input example from first few rows of test data
            input_example = X_test_scaled[:5]
            mlflow.sklearn.log_model(
                model, 
                artifact_path="model",  # Use artifact_path instead of name
                input_example=input_example
            )
            
            # Log artifacts (e.g., confusion matrix)
            print("Creating and logging confusion matrix...")
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            os.makedirs("confusion_matrix", exist_ok=True)
            cm_path = f"confusion_matrix/confusion_matrix_{name}.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()
            print(f"Confusion matrix saved and logged: {cm_path}")

            print(f"\nRESULTS FOR {name}:")
            print(f"   • Precision: {precision:.4f}")
            print(f"   • Recall: {recall:.4f}")
            print(f"   • F1-Score: {f1:.4f}")
            print(f"   • AUPRC: {auprc:.4f}")
            print(f"   • ROC AUC: {roc_auc:.4f}")
            print(f"{name} training and evaluation completed!")

    print("ALL MODELS TRAINING COMPLETED SUCCESSFULLY!")
    print("Generated files:")
    print("   • preprocessor.pkl (saved preprocessing pipeline)")
    print("   • confusion_matrix_*.png (confusion matrices for each model)")
    print("Check MLflow UI for detailed experiment tracking results")

if __name__ == "__main__":
    train_and_evaluate()