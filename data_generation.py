import numpy as np
import pandas as pd
import QuantLib as ql

def generate_data(num_samples=10):
    results = []
    for i in range(num_samples):
        spot = np.random.uniform(0.01,200)
        strike = np.random.uniform(0.01,200)
        vol = np.random.uniform(0.05,0.5)
        maturity = np.random.randint(1,1095) #days
        risk_free_rate = np.random.uniform(-0.02,0.08)
        q = np.random.uniform(0,0.08) #div yield

        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        todays_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = todays_date
        maturity_date = todays_date + maturity

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, risk_free_rate, ql.Actual365Fixed()))
        dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, q, ql.Actual365Fixed()))
        flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(todays_date, calendar, vol, ql.Actual365Fixed()))
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts)

        european_exercise = ql.EuropeanExercise(maturity_date)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        option = ql.VanillaOption(payoff, european_exercise)

        #binomial
        option.setPricingEngine(ql.BinomialVanillaEngine(bsm_process, "crr", 100))
        binomial_price = option.NPV()

        #MC
        option.setPricingEngine(ql.MCEuropeanEngine(bsm_process, "pseudorandom", timeSteps=100, requiredSamples=10000))
        mc_price = option.NPV()

        results.append([spot, strike, vol, maturity, risk_free_rate, q, binomial_price, mc_price])

    columns = ['spot', 'strike', 'vol', 'maturity_days', 'rate', 'div_yield', 'binomial_price', 'mc_price']
    return pd.DataFrame(results, columns=columns)

df = generate_data(1000)
df.to_csv("option_pricing_data.csv", index=False)
print(df.head())