import pickle
import numpy as np

# Load the model
with open("Diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input
x_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # Replace with your inputs
prediction = model.predict(x_data)

print("Prediction:", prediction)  # Should be 0 or 1
