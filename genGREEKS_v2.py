import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import pandas as pd

class BlackScholesGreeks:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = max(T, 0.00001)  # Floor to avoid division by zero
        self.r = r
        self.sigma = max(sigma, 0.0001) # Floor to prevent zero-volatility crash
        
        # Core Black-Scholes d1/d2 variables
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def theoretical_price(self, option_type='call'):
        """Calculates the theoretical Black-Scholes fair value."""
        if option_type.lower() == 'call':
            return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)

    def prob_itm(self, option_type='call'):
        """The structural Trend / Probability of expiring In-The-Money."""
        if option_type.lower() == 'call':
            return norm.cdf(self.d2)
        else:
            return norm.cdf(-self.d2)

    def delta(self, option_type='call'):
        if option_type.lower() == 'call':
            return norm.cdf(self.d1)
        return norm.cdf(self.d1) - 1

    def gamma(self):
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return (self.S * norm.pdf(self.d1) * np.sqrt(self.T)) / 100

    def theta(self, option_type='call'):
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type.lower() == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            return (term1 - term2) / 365
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            return (term1 + term2) / 365

    def rho(self, option_type='call'):
        if option_type.lower() == 'call':
            return (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)) / 100
        else:
            return (-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)) / 100

   # Refined Higher Order Greeks
    def vanna(self):
        # Result is sensitivity of delta to 1% change in vol
        return (self.vega() / self.S) * (1 - self.d1 / (self.sigma * np.sqrt(self.T)))

    def charm(self, option_type='call'):
        term1 = norm.pdf(self.d1) * ((self.r / (self.sigma * np.sqrt(self.T))) - (self.d2 / (2 * self.T)))
        if option_type.lower() == 'call':
            return -term1
        else:
            # Puts have an extra term related to the drift of the strike
            return -term1 - (self.r * np.exp(-self.r * self.T) * norm.cdf(-self.d2))

    def vomma(self):
        # Sensitivity of Vega to Vol
        return self.vega() * self.d1 * self.d2 / self.sigma


def generate_greeks_csv(ticker_sym):
    print(f"📡 Compiling institutional data block for {ticker_sym}...")
    ticker = yf.Ticker(ticker_sym.upper())
    
    current_price = ticker.info.get('currentPrice')
    if not current_price:
        current_price = ticker.history(period="1d")['Close'].iloc[-1]

    expirations = ticker.options
    if not expirations:
        return False, f"No options data available for {ticker_sym}."

    # Updated Headers including Price, Trend, and Higher-Order Greeks
    headers = [
        'Contract_Symbol', 'Exp_Date', 'Type', 'Strike', 'Underlying_Price', 
        'BS_Theo_Price', 'Last_Price', 'Pricing_Edge', 'IV', 'Days_To_Expiry', 'Prob_ITM_Trend',
        'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vanna', 'Charm', 'Vomma'
    ]
    
    calls_data = []
    puts_data = []
    r = 0.04  # Risk-free rate assumption

    for exp in expirations:
        chain = ticker.option_chain(exp)
        days_to_expiry = max((datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days, 1)
        T = days_to_expiry / 365.0

        # Process CALLS
        for _, row in chain.calls.iterrows():
            greeks = BlackScholesGreeks(current_price, row['strike'], T, r, row['impliedVolatility'])
            theo_price = greeks.theoretical_price('call')
            edge = round(row['lastPrice'] - theo_price, 2) # Positive = Overpriced by market
            
            calls_data.append([
                row['contractSymbol'], exp, 'CALL', row['strike'], current_price, 
                theo_price, row['lastPrice'], edge, row['impliedVolatility'], days_to_expiry, greeks.prob_itm('call'),
                greeks.delta('call'), greeks.gamma(), greeks.vega(), greeks.theta('call'), greeks.rho('call'),
                greeks.vanna(), greeks.charm('call'), greeks.vomma()
            ])

        # Process PUTS
        for _, row in chain.puts.iterrows():
            greeks = BlackScholesGreeks(current_price, row['strike'], T, r, row['impliedVolatility'])
            theo_price = greeks.theoretical_price('put')
            edge = round(row['lastPrice'] - theo_price, 2)
            
            puts_data.append([
                row['contractSymbol'], exp, 'PUT', row['strike'], current_price, 
                theo_price, row['lastPrice'], edge, row['impliedVolatility'], days_to_expiry, greeks.prob_itm('put'),
                greeks.delta('put'), greeks.gamma(), greeks.vega(), greeks.theta('put'), greeks.rho('put'),
                greeks.vanna(), greeks.charm('put'), greeks.vomma()
            ])

    # Convert to DataFrames
    df_calls = pd.DataFrame(calls_data, columns=headers)
    df_puts = pd.DataFrame(puts_data, columns=headers)

    calls_file = f"DATA_{ticker_sym.upper()}_CALLS.csv"
    puts_file = f"DATA_{ticker_sym.upper()}_PUTS.csv"
    
    df_calls.to_csv(calls_file, index=False)
    df_puts.to_csv(puts_file, index=False)

    return True, f"Engine complete. Generated full derivative matrices for {calls_file} and {puts_file}."
