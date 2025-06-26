import torch
import numpy as np
import pandas as pd
import time
import QuantLib as ql

from model import OptionPricingModel
from preprocess import preprocess_data


df = pd.read_csv("option_pricing_data.csv")
dataset = preprocess_data(df)

x_scaler = dataset['x_scaler']
y_scaler = dataset['y_scaler']

X_test = dataset['X_test']
y_test = dataset['y_test']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OptionPricingModel()
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)


y_true = y_scaler.inverse_transform(y_test)

with torch.no_grad():
    y_pred = model(X_test_tensor).cpu().numpy()

y_pred = y_scaler.inverse_transform(y_pred)

from sklearn.metrics import mean_squared_error

rmse_mc = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
rmse_binomial = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))

print("\n📈 Accuracy:")
print(f"RMSE vs Monte Carlo: {rmse_mc:.6f}")
print(f"RMSE vs Binomial: {rmse_binomial:.6f}")


start = time.time()

for _ in range(1000):
    _ = model(X_test_tensor)

if torch.cuda.is_available():
    torch.cuda.synchronize()

nn_time = (time.time() - start) / 1000
nn_time_per_price = nn_time / X_test.shape[0]  

print(f"\n⚡ Neural Net Avg Time per Price Pair: {nn_time_per_price*1000:.6f} ms")


#speed test monte carlo
start = time.time()

for row in X_test:
    spot, strike, vol, maturity_days, rate, div = x_scaler.inverse_transform([row])[0]

    # Set up QuantLib pricing
    todays_date = ql.Date.todaysDate()
    maturity_date = todays_date + int(maturity_days)

    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    ql.Settings.instance().evaluationDate = todays_date

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, rate, ql.Actual365Fixed()))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, div, ql.Actual365Fixed()))
    flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(todays_date, calendar, vol, ql.Actual365Fixed()))
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    mc_engine = ql.MCEuropeanEngine(bsm_process, "pseudorandom", timeSteps=100, requiredSamples=10000)
    option.setPricingEngine(mc_engine)
    _ = option.NPV()

mc_time = (time.time() - start) / X_test.shape[0]
print(f"⏳ Monte Carlo Avg Time per Price Pair: {mc_time*1000:.6f} ms")

#speed test binomial tree
start = time.time()

for row in X_test:
    spot, strike, vol, maturity_days, rate, div = x_scaler.inverse_transform([row])[0]

    todays_date = ql.Date.todaysDate()
    maturity_date = todays_date + int(maturity_days)

    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    ql.Settings.instance().evaluationDate = todays_date

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, rate, ql.Actual365Fixed()))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, div, ql.Actual365Fixed()))
    flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(todays_date, calendar, vol, ql.Actual365Fixed()))
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", 100)
    option.setPricingEngine(binomial_engine)
    _ = option.NPV()

binomial_time = (time.time() - start) / X_test.shape[0]
print(f"⏳ Binomial Avg Time per Price Pair: {binomial_time*1000:.6f} ms")


print("\n🚀 Speedup Factors:")
print(f"NN vs Monte Carlo: {mc_time / nn_time_per_price:.2f}x faster")
print(f"NN vs Binomial: {binomial_time / nn_time_per_price:.2f}x faster")
