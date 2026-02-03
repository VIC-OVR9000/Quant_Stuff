import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime

class BlackScholesGreeks:
    """
    Methods implement the standard Black--Scholes Greeks.
    Derivations:
      - Delta (call): Delta = N(d1)
      - Gamma: Gamma = n(d1) / (S * sigma * sqrt(T))
      - Vega: Vega = S * n(d1) * sqrt(T)             (note: code returns vega/100 -> per %-point)
      - Theta (call): Theta = -S*n(d1)*sigma/(2*sqrt(T)) - r*K*e^{-rT}*N(d2)
                      (note: code divides by 365 -> returns per day)
      - Rho (call):  Rho = K*T*e^{-rT}*N(d2)         (note: code returns rho/100 -> per %-point)
    Small epsilons are included to avoid division by zero in d1/d2 and gamma.
    """
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        # T must be positive; add small lower bound to avoid division by zero in formulas
        self.T = max(T, 0.00001)
        self.r = r
        self.sigma = sigma
        
        # small eps in denominator helps numeric stability when T or sigma are extremely small
        self.d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T)+0.0000001)
        self.d2 = self.d1 - sigma * np.sqrt(self.T)

    def delta(self, option_type='call'):
        # Delta for call: N(d1). For put: N(d1)-1
        if option_type == 'call':
            return norm.cdf(self.d1)
        return norm.cdf(self.d1) - 1

    def gamma(self):
        # Gamma = n(d1) / (S * sigma * sqrt(T))
        # small eps added in denominator to avoid division by zero
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T)+0.0000001)

    def vega(self):
        # Vega = S * n(d1) * sqrt(T)
        # NOTE: code returns vega / 100 to represent "change per 1% move in vol"
        return (self.S * norm.pdf(self.d1) * np.sqrt(self.T)) / 100

    def theta(self, option_type='call'):
        # Theta per day: standard Theta formula divided by 365
        # Standard (per-year) call theta:
        #   Theta_call = -S*n(d1)*sigma/(2*sqrt(T)) - r*K*e^{-rT}*N(d2)
        # Code computes term1 = -S*n(d1)*sigma/(2*sqrt(T)) then subtracts risk-free term.
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            # code returns per-day theta by dividing annual theta by 365
            return (term1 - term2) / 365
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            return (term1 + term2) / 365

    def rho(self, option_type='call'):
        # Rho (per unit r): Rho_call = K*T*e^{-rT}*N(d2)
        # NOTE: code returns rho / 100 to represent "change per 1% move in interest rates"
        if option_type == 'call':
            return (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)) / 100
        else:
            return (-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)) / 100



# 1. Fetch market data
ticker = yf.Ticker("ROBN")
current_price = ticker.fast_info['lastPrice']
expirations = ticker.options

#print(f"{'Exp Date':<12} | {'Delta':<8} | {'Gamma':<8} | {'Vega':<8} | {'Theta':<8} | {'Rho':<8} | {'S':<8} | {'K':<8} | {'T':<8} | {'r':<8} | {'sigma':<8}")
#print("-" * 140)
print(expirations)

for j in range(len(expirations)):
    print(f"{'Exp Date':<12} | {'Delta':<8} | {'Gamma':<8} | {'Vega':<8} | {'Theta':<8} | {'Rho':<8} | {'S':<8} | {'K':<8} | {'T':<8} | {'r':<8} | {'sigma':<8}")
    print("-" * 140)
    
    for i in range(len(expirations[j])-1):
        chain = ticker.option_chain(expirations[j])
        
        # 2. Extract parameters (using first call for each expiration)
        opt = chain.calls.iloc[i]
        #print(opt)
        S = current_price
        K = opt.strike
        sigma = opt.impliedVolatility
        r = 0.04  # Risk-free rate (approx 4% in 2026)
        
        # 3. Calculate Time to Expiration (T)
        days_to_expiry = (datetime.strptime(expirations[j], '%Y-%m-%d') - datetime.now()).days
        T = max(days_to_expiry, 1) / 365
        
        # 4. Compute Greeks
        greeks = BlackScholesGreeks(S, K, T, r, sigma)
        
        d = greeks.delta('call')   # = N(d1)
        g = greeks.gamma()         # = n(d1) / (S*sigma*sqrt(T))
        v = greeks.vega()          # = S*n(d1)*sqrt(T) / 100  -> per 1% vol
        t = greeks.theta('call')   # = (Theta_per_year)/365 -> per day
        rho = greeks.rho('call')   # = (K*T*e^{-rT}*N(d2))/100 -> per 1% rate
                        #### <8.4 is for space and deciaml format
        print(f"{expirations[j]:<12} | {d:>8.4f} | {g:>8.4f} | {v:>8.4f} | {t:>8.4f} | {rho:>8.4f} | {S:<8.4} | {K:<8.4} | {T:<8.4} | {r:<8.4} | {sigma:<8.4}")
