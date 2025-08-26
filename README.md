# End-to-End Credit Card Fraud Detection System

## Project Structure

```
fraud-detection-system/
├──.github/
│   └── workflows/
│       └── deploy.yml          # CI/CD pipeline definition
├── data/
│   └── creditcard.csv          # Raw dataset (or instructions to download)
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory Data Analysis notebook
├── scripts/
│   ├── etl_pipeline.py         # Script for ETL process
│   ├── train_model.py          # Script for model training and MLflow tracking
│   └── drift_detector.py       # Script for drift detection
├── src/
│   ├── main.py                 # FastAPI application code
│   ├── db.py                   # Database models and session management
│   └──...                     # Other source code modules
├── dags/
│   └── retraining_dag.py       # Airflow DAG for automated retraining
├── dashboard/
│   └── dashboard.py            # Streamlit dashboard application
├──.dockerignore               # Files to ignore when building Docker image
├──.gitignore                  # Files to ignore for Git
├── Dockerfile                  # Instructions to build the application container
├── requirements.txt            # Python dependencies for the project
└── README.md                   # The most important file: project documentation
```

## Project Overview and Business Problems 

This repository contains end-to-end machine learning for detection fraudulent credit card transaction. Credit Card fraud is a big problem for financial institution that make them lost billion of dollars and losing trustworhty. This project aims to build a robust, scalable and maintainable fraud detection system that can detect fraudulent transaction with high accuracy, and minimizing financial loss.

## System Architecture

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*beSO1ehuC-VmF9BZGGiOyw.png)

This system build using MLOps Stack that designed for automation, scalability, and maintability. The architecture contains several key components :

- **Data Pipeline (ETL)** : ETL is Extract, Transform, and Load. we will using python script to transform raw data to usefull information. from cleaning, transformation, feature engeneering, and loads it into PostgreSQL database.
- **Model Training and Experimental Tracking** : Training pipeline will using MLFlow to do Model Selection process, which is we will systematically training multiple models (Logistic Regression, Random Forest, LightGBM), tuning the parameter, and tracking all experiments result. 
- **Model Serving** : After model trained, model will be packed and served via REST API built with FAST API
- **Deployment** : All project environment will be packed in Docker container, and will be deployed to EC2 server by AWS, and for security, the communication will pass Nginx reverse proxy, and will be secured with SSL certificate.
- **CI/CD** : Used for automatically deployment everytime there is a change in our system. 
- **Monitoring and Retraining** : Monitoring will write all prediction, and retraining will using Airflow DAG to retrain model automatically.
- **Dashboard** : Streamlit dashboard will be used to visualize the model performance, and monitoring the system.

## Tech Stack

- **Languages**: Python
    
- **Libraries**: Scikit-learn, Pandas, LightGBM, Imbalanced-learn (for SMOTE)
    
- **Experiment Tracking**: MLflow
    
- **API**: FastAPI, Uvicorn, Gunicorn
    
- **Containerization**: Docker
    
- **Cloud & Deployment**: AWS EC2, Nginx
    
- **CI/CD**: GitHub Actions
    
- **Workflow Orchestration**: Apache Airflow
    
- **Dashboard**: Streamlit
    
- **Database**: PostgreSQL, SQLite

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
```

2. Create and activate virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Instal dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file in the root directory.
- Add necessary environment variables (e.g., database credentials, API keys).

Install postgresql

```bash
brew install postgresql
```
Run it

```bash
/opt/homebrew/opt/postgresql@14/bin/postgres -D /opt/homebrew/var/postgresql@14
```

or

```bash
brew services start postgresql@14
```

Check if its running successfully
```bash
psql -U postgres -h localhost
```

Make the database for project
```bash
CREATE DATABASE fraud_detection;
```

Add this to ur env 
```bash
export DB_USER=postgres
export DB_PASSWORD=postgres
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=fraud_detection
```


```bash
DATABASE_URL=postgresql://user:password@localhost:5432/frauddb
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_ACCESS_KEY_ID=xxxx
AWS_SECRET_ACCESS_KEY=xxxx
AWS_REGION=us-east-1
AWS_S3_BUCKET=fraud-detection-model
AWS_S3_BUCKET_MODEL=fraud-detection-model
```

5. Docker Services
```bash
docker-compose up -d
```

## Usage

1. Run ETL Pipeline (Optional before training) :
```bash
python etl_pipeline.py
```

2. Run training pipeline (MLFlow) :
```bash
python training/train.py
```

3. Run FastAPI app (for serving)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

4. Run dashboard
```bash
streamlit run dashboard/dashboard.py
```

## Results

The final model is a tuned LightGBM classifier. It was selected after comparing its performance against several baseline and advanced models.

|Model|Recall|AUPRC|
|---|---|---|
|Logistic Regression|0.918|0.758|
|Random Forest|0.827|0.843|
|LightGBM (Tuned)|**0.847**|**0.895**|

The model successfully balances high recall (catching a large percentage of fraud) with a strong AUPRC, indicating robust performance on the imbalanced test set.
