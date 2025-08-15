Concrete Strength Prediction Using Machine Learning

This project predicts 28-day concrete compressive strength using machine learning models. It includes a Flask API, evaluation metrics, and a live public demo link.

Project Overview

Goal: Predict the compressive strength of concrete based on its ingredients and age.

Models used:

Linear Regression

Random Forest

Neural Network

Features used:

Cement, Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age

Live Demo

You can access the running app here:
https://my-app-660887990276.us-central1.run.app/<img width="1194" height="80" alt="image" src="https://github.com/user-attachments/assets/80f94e7e-8d74-48f4-995e-ca1bf83e8662" />



Endpoints available:

Endpoint	Method	Description
/predict	POST	Predict concrete strength. Input JSON with features.
/evaluate/json	GET	Get model evaluation metrics in JSON format.
/evaluate/html	GET	Get model evaluation metrics as an HTML table.

Example POST /predict JSON payload:

{
  "cement": 540,
  "slag": 0,
  "fly_ash": 0,
  "water": 162,
  "superplasticizer": 2.5,
  "coarse": 1040,
  "fine": 676,
  "age": 28
}

Project Structure
concrete-strength-demo/
│
├─ app.py                  # Flask API
├─ evaluation.py           # Model evaluation script
├─ linear_regression.pkl   # Pretrained Linear Regression model
├─ random_forest.pkl       # Pretrained Random Forest model
├─ neural_network.pkl      # Pretrained Neural Network model
├─ scaler.pkl              # Data scaler
├─ requirements_api.txt    # Dependencies for API
├─ Dockerfile.api          # Dockerfile for API
└─ Project.pptx            # Presentation slides

Setup

Clone the repository:

git clone https://github.com/ammoradi1981/concrete-strength-demo.git
cd concrete-strength-demo


Install dependencies:

pip install -r requirements_api.txt


Run the API locally:

python app.py


The app will run on the publick link:  https://my-app-660887990276.us-central1.run.app/<img width="1194" height="80" alt="image" src="https://github.com/user-attachments/assets/f5cb07f5-43e4-43b8-a209-0730a6e26a20" />



Author

Amirmohammad Moradi
GitHub
