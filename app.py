from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load models
lr = joblib.load("linear_regression.pkl")
rf = joblib.load("random_forest.pkl")
mlp = joblib.load("neural_network.pkl")
scaler = joblib.load("scaler.pkl")

# HTML template for browser demo
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Concrete Strength Demo</title>
</head>
<body>
    <h2>Concrete 28-Day Strength Prediction</h2>
    <form action="/predict" method="get">
        Cement (kg/m³): <input type="number" name="cement" value="540"><br>
        Slag (kg/m³): <input type="number" name="slag" value="0"><br>
        Fly Ash (kg/m³): <input type="number" name="fly_ash" value="0"><br>
        Water (kg/m³): <input type="number" name="water" value="162"><br>
        Superplasticizer (kg/m³): <input type="number" name="superplasticizer" value="0"><br>
        Coarse Aggregate (kg/m³): <input type="number" name="coarse" value="1200"><br>
        Fine Aggregate (kg/m³): <input type="number" name="fine" value="1000"><br>
        Age (days): <input type="number" name="age" value="28"><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <h3>Prediction Results:</h3>
        <ul>
            <li>Linear Regression (MPa): {{ prediction['LinearRegression'] }}</li>
            <li>Random Forest (MPa): {{ prediction['RandomForest'] }}</li>
            <li>Neural Network (MPa): {{ prediction['NeuralNetwork'] }}</li>
        </ul>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(html_template)

@app.route("/predict", methods=["POST", "GET"])
def predict():
    # GET from browser form
    if request.method == "GET":
        data = {
            "cement": float(request.args.get("cement", 540)),
            "slag": float(request.args.get("slag ", 0)),
            "fly_ash": float(request.args.get("fly_ash", 0)),
            "water": float(request.args.get("water", 162)),
            "superplasticizer": float(request.args.get("superplasticizer", 0)),
            "coarse": float(request.args.get("coarse", 1200)),
            "fine": float(request.args.get("fine", 1000)),
            "age": float(request.args.get("age", 28))
        }
    else:  # POST via API
        data = request.get_json()

    sample = np.array([[data["cement"], data["slag"], data["fly_ash"], data["water"],
                        data["superplasticizer"], data["coarse"], data["fine"], data["age"]]])
    sample_scaled = scaler.transform(sample)

    prediction = {
        "LinearRegression": round(lr.predict(sample_scaled)[0], 2),
        "RandomForest": round(rf.predict(sample)[0], 2),
        "NeuralNetwork": round(mlp.predict(sample_scaled)[0], 2)
    }

    # Render results in browser for GET, return JSON for POST
    if request.method == "GET":
        return render_template_string(html_template, prediction=prediction)
    else:
        return jsonify(prediction)

    # Return results with units
    return jsonify({
        "LinearRegression, MPa": f"{lr_pred} MPa",
        "RandomForest, MPa": f"{rf_pred} MPa",
        "NeuralNetwork, MPa": f"{nn_pred} MPa"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
