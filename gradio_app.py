import gradio as gr
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load saved models and scaler
lr = joblib.load("linear_regression.pkl")
rf = joblib.load("random_forest.pkl")
mlp = joblib.load("neural_network.pkl")
scaler = joblib.load("scaler.pkl")

def predict_and_plot(cement, slag, fly_ash, water, superplasticizer, coarse, fine, age):
    sample = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse, fine, age]])
    
    # Scale sample for models that require it
    sample_scaled = scaler.transform(sample)
    
    # Make predictions
    lr_pred = lr.predict(sample_scaled)[0]
    rf_pred = rf.predict(sample)[0]
    nn_pred = mlp.predict(sample_scaled)[0]
    
    # Create a bar chart
    models = ["Linear Regression", "Random Forest", "Neural Network"]
    predictions = [lr_pred, rf_pred, nn_pred]
    
    plt.figure(figsize=(6,4))
    bars = plt.bar(models, predictions, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel("Predicted Strength (MPa)")
    plt.title("Concrete Strength Predictions (28-day)")
    plt.ylim(0, max(predictions)+10)
    
    # Annotate bars with values
    for bar, pred in zip(bars, predictions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5, f"{pred:.2f}", 
                 ha='center', fontsize=10)
    
    plt.tight_layout()
    
    return plt

# Create input sliders
inputs = [
    gr.Slider(100, 600, value=350, label="Cement (kg/m³)"),
    gr.Slider(0, 300, value=50, label="Blast Furnace Slag (kg/m³)"),
    gr.Slider(0, 200, value=0, label="Fly Ash (kg/m³)"),
    gr.Slider(100, 250, value=160, label="Water (kg/m³)"),
    gr.Slider(0, 30, value=5, label="Superplasticizer (kg/m³)"),
    gr.Slider(800, 1200, value=1000, label="Coarse Aggregate (kg/m³)"),
    gr.Slider(600, 1000, value=700, label="Fine Aggregate (kg/m³)"),
    gr.Slider(1, 365, value=28, label="Age (days)")
]

outputs = gr.Plot(label="Predicted Strength Comparison")

# Launch Gradio interface
gr.Interface(
    fn=predict_and_plot,
    inputs=inputs,
    outputs=outputs,
    title="🧱 Concrete Strength Prediction",
    description="Enter your concrete mix design to predict 28-day compressive strength. The results are shown as a bar chart comparing the three ML models."
).launch()
