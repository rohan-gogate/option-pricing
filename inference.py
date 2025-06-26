import torch
import numpy as np
import pandas as pd
from model import OptionPricingModel
from preprocess import preprocess_data
import joblib

df = pd.read_csv("option_pricing_data.csv")
dataset = preprocess_data(df)

x_scaler = dataset["x_scaler"]
y_scaler = dataset["y_scaler"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OptionPricingModel().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)
model.eval()

input_params = np.array([[100, 100, 0.2, 90, 0.01, 0.02]])  # Example input

input_scaled = x_scaler.transform(input_params)

with torch.no_grad():
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    output = model(input_tensor)

output = output.cpu().numpy()

predicted_prices = y_scaler.inverse_transform(output)

mc_price, binomial_price = predicted_prices[0]

print(f"Monte Carlo Price Prediction: {mc_price:.4f}")
print(f"Binomial Tree Price Prediction: {binomial_price:.4f}")