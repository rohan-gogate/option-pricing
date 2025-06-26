import torch
import matplotlib.pyplot as plt
import numpy as np
from model import OptionPricingModel
import pandas as pd
from preprocess import preprocess_data


df = pd.read_csv("option_pricing_data.csv")
dataset = preprocess_data(df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OptionPricingModel().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)

y_true = dataset['y_scaler'].inverse_transform(dataset['y_test'])

model.eval()
X_test_tensor = torch.tensor(dataset['X_test'], dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred = model(X_test_tensor).cpu().numpy()

y_pred = dataset['y_scaler'].inverse_transform(y_pred)

plt.figure(figsize=(8, 8))

plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, label="Monte Carlo", color='blue')

plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5, label="Binomial", color='green')

min_price = min(y_true.min(), y_pred.min())
max_price = max(y_true.max(), y_pred.max())
plt.plot([min_price, max_price], [min_price, max_price], 'r--', label="Perfect Prediction")

plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs True Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate errors
error_mc = y_pred[:, 0] - y_true[:, 0]         # Monte Carlo error
error_binomial = y_pred[:, 1] - y_true[:, 1]   # Binomial error

# Plot Monte Carlo error
plt.figure(figsize=(8,6))
plt.scatter(y_true[:, 0], error_mc, alpha=0.5, color='blue', label='Monte Carlo')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('True Monte Carlo Price')
plt.ylabel('Prediction Error')
plt.title('Prediction Error vs True Monte Carlo Price')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot Binomial error
plt.figure(figsize=(8,6))
plt.scatter(y_true[:, 1], error_binomial, alpha=0.5, color='green', label='Binomial')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('True Binomial Price')
plt.ylabel('Prediction Error')
plt.title('Prediction Error vs True Binomial Price')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()