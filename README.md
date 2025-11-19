# BentoML ML Service – Model Serving API

Production-style **machine learning model serving project** built with BentoML.  
This repo simulates a real-world ML backend where a trained model is exposed as an HTTP API for online predictions.

## 🔍 Overview

- Serves a trained ML model behind a **`/predict`** REST endpoint.  
- Supports **single** and **batch** predictions.  
- Includes basic **input validation**, structured logging, and versioned model artifacts.  
- Follows a clean production-style layout for interview-ready projects.

## 🧱 Architecture

### BentoML Service

- `service.py` — defines the BentoML service, API routes, and inference logic.  
- `model/` — placeholder folder containing exported ML models.

### Dependencies

- `requirements.txt` — Python dependencies required for serving.

### Metadata

- `README.md` — project documentation.  
- `.gitignore` — ignores virtualenv, build files, and local artifacts.

## ▶️ How It Works

1. A machine learning model is trained offline using any framework  
   (for example: scikit-learn, XGBoost, or LightGBM).  
2. The trained model is **saved and registered** with BentoML.  
3. The BentoML service exposes a **`/predict`** endpoint that:
   - Accepts JSON payloads  
   - Runs preprocessing + model inference  
   - Returns prediction outputs in JSON format  

## 🚀 Example Usage

Once the service is built and started, a client could call:

    curl -X POST "http://localhost:3000/predict" \
      -H "Content-Type: application/json" \
      -d '{
        "features": [
          [0.3, 1.2, 5.1, 0],
          [0.9, 0.4, 3.3, 1]
        ]
      }'

And receive a response like:

    {
      "predictions": [0, 1],
      "scores": [0.18, 0.87],
      "model_version": "v1.0.0"
    }

## 🔮 Future Improvements (great for interviews)

- Add a real training notebook and export a production-ready model.  
- Write unit tests for the service’s input/output contract.  
- Containerize with Docker and deploy using BentoML’s deployment tools.  
- Add monitoring (latency, throughput) and model version rollback.

## 🧰 Tech Stack

Python · BentoML · REST API · Machine Learning · Model Serving · JSON · Git/GitHub
