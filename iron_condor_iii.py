import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. Black-Scholes Gamma Function (from search results) ---
def black_scholes_gamma(S, K, T, r, sigma):
    """ Calculates the Gamma of an option using Black-Scholes. """
    T = max(T, 1e-6) # Avoid division by zero
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# --- 2. P/L at Expiration Function ---
def iron_condor_payoff(s_t, k1, k2, k3, k4, c_net):
    """ Calculates the P/L of an Iron Condor at expiration. """
    long_put_payoff = np.maximum(0, k1 - s_t)
    short_put_payoff = np.maximum(0, k2 - s_t)
    short_call_payoff = np.maximum(0, s_t - k3)
    long_call_payoff = np.maximum(0, s_t - k4)
    payoff = long_put_payoff - short_put_payoff + short_call_payoff - long_call_payoff + c_net
    return payoff

# --- 3. Current Gamma Profile Function ---
def iron_condor_gamma(S, K1, K2, K3, K4, T, r, sigma):
    """ Calculates the total gamma of an Iron Condor position. """
    # Long put (+Gamma), Short put (-Gamma), Short call (-Gamma), Long call (+Gamma)
    gamma_total = (
        black_scholes_gamma(S, K1, T, r, sigma) -
        black_scholes_gamma(S, K2, T, r, sigma) -
        black_scholes_gamma(S, K3, T, r, sigma) +
        black_scholes_gamma(S, K4, T, r, sigma)
    )
    return gamma_total

# --- 4. Define Multiple Iron Condors ---
# Format: [Long Put K1, Short Put K2, Short Call K3, Long Call K4, Net Credit]

con1 = [90, 95, 105, 110, 2.0]
shift2 = 5
con2 = list(  np.array(con1) + np.array([shift2]*int(len(con1)))  )


iron_condors = [
    [90, 95, 105, 110, 2.0], # Condor 1 (Blue)
    [85, 90, 110, 115, 2.5],  # Condor 2 (Red)
    con2
]

# --- 5. Inputs for Gamma Calculation (Requires real-time data) ---
SPOT_PRICE = 100        # Current underlying price
RISK_FREE_RATE = 0.05   # e.g., 5%
VOLATILITY = 0.20       # e.g., 20%
DAYS_TO_EXPIRY = 21     # Time to expiration in days
TIME_TO_EXPIRY = DAYS_TO_EXPIRY / 365.0

# --- 6. Generate Prices for Plotting ---
s_t = np.arange(75, 125, 0.1)
combined_payoff = np.zeros_like(s_t)
combined_gamma = np.zeros_like(s_t)

# --- 7. Plotting Function ---
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx() # Secondary axis for Gamma

colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58'] # Google colors for style

for i, ic in enumerate(iron_condors):
    k1, k2, k3, k4, credit = ic
    # Calculate P/L
    payoff = iron_condor_payoff(s_t, k1, k2, k3, k4, credit)
    ax1.plot(s_t, payoff, label=f'Condor {i+1} P/L', color=colors[i], linestyle='--')
    combined_payoff += payoff
    
    # Calculate Gamma for current spot price range (requires an array of spot prices)
    gamma_profile = iron_condor_gamma(s_t, k1, k2, k3, k4, TIME_TO_EXPIRY, RISK_FREE_RATE, VOLATILITY)
    combined_gamma += gamma_profile

# Plot combined P/L and Gamma
ax1.plot(s_t, combined_payoff, label='Combined P/L', color='black', linewidth=2)
ax2.plot(s_t, combined_gamma, label='Combined Gamma', color='grey', linestyle=':', linewidth=2)

# --- 8. Formatting the Plot ---
ax1.set_xlabel('Underlying Price ($)')
ax1.set_ylabel('Profit/Loss ($)')
ax2.set_ylabel('Total Gamma')
ax1.set_title(f'Iron Condor P/L and Gamma Profile (T={DAYS_TO_EXPIRY} days)')
ax1.axhline(0, color='gray', linestyle='-') # Breakeven line

# Add current spot price marker and gamma magnitude
# Index the gamma profile at the current spot price (approx)
current_gamma_index = np.abs(s_t - SPOT_PRICE).argmin()
current_gamma_value = combined_gamma[current_gamma_index]
ax2.axvline(SPOT_PRICE, color='green', linestyle='--', label=f'Spot Price: ${SPOT_PRICE}')
ax2.annotate(f'Gamma: {current_gamma_value:.4f}', xy=(SPOT_PRICE, current_gamma_value), 
             xytext=(SPOT_PRICE + 2, current_gamma_value + 0.01),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='left')

fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.grid(True)
plt.show()

