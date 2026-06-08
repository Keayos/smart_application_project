# smart_application_project
An AI-powered predictive maintenance assistant that uses an Artificial Neural Network (ANN) to analyze real-time sensor metrics (temperature, vibration, pressure) and predict machine failure risk through an interactive desktop dashboard.
# Geometric_Diagram Template

## Description
This project is an AI-powered predictive maintenance tool designed to monitor industrial machinery and prevent unexpected downtimes. It uses an Artificial Neural Network (ANN) trained on real-time sensor streams (including temperature, vibration, pressure, humidity, and energy consumption). The system evaluates operational anomalies and calculates live risk probabilities, displaying recommendations through a clean, dark-themed desktop dashboard.

---

## Features
- **ANN Risk Engine:** Powered by a neural network that processes multi-variable telemetry data to predict if machinery requires immediate maintenance.
- **Pure Python Math Logic:** Features a custom inference system inside the interface that runs data scaling, ReLU activation, and matrix operations manually using raw mathematics.
- **Live Diagnostics Dashboard:** A sleek Tkinter desktop window featuring slider-based telemetry controls, instant status readouts, and automated maintenance recommendations.
- **Rolling Log History:** Displays a history tracker tracking the last 5 telemetry evaluations with timestamps and colored condition flags.

---

## Technologies Used
* **Python 3.x**
* **Tkinter** (For building the dark-themed desktop controls)
* **PyTorch** (For building, training, and exporting the initial neural network)
* **Pandas & Scikit-Learn** (For analyzing industrial data logs and computing data preprocessing states)

---

## How to Use
1. **Train and Export the Model:** Run the `model.py` script. This will train the Artificial Neural Network on the sensor dataset and automatically export the trained weights, architecture configurations, threshold values, and the scaler's mean and standard deviation parameters as JSON configurations.
   ```bash
   python model.py
2. **Launch the Dashboard:** Run the interactive user interface script to load the exported parameters and boot the live system diagnostic controls:
```bash
   python "Maintenance_ui.py"
