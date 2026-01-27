import numpy as np
import matplotlib.pyplot as plt

def calculate_iron_condor_payoff(sT, p_long, p_short, c_short, c_long, net_premium):
    """Calculates payoff for one Iron Condor."""
    # Long Put + Short Put + Short Call + Long Call
    payoff = (np.maximum(0, p_long - sT) +
              np.maximum(0, sT - p_short) + 
              np.maximum(0, c_short - sT) + # Correction: Payoff definition [1,11]
              np.maximum(0, sT - c_long) - (p_short + c_short - p_long - c_long) + net_premium)
    # Corrected payoff formula based on typical IC construction [1, 5]
    payoff = (np.maximum(0, p_long - sT) - np.maximum(0, p_short - sT) - 
              np.maximum(0, sT - c_short) + np.maximum(0, sT - c_long) + net_premium)
    return payoff

# 1. Define Stock Price Range
spot_price = 100
sT = np.arange(0.8 * spot_price, 1.2 * spot_price, 1)

# 2. Define Multiple Iron Condors (Put Long, Put Short, Call Short, Call Long, Credit)
condors = [
    {'p_l': 80, 'p_s': 90, 'c_s': 110, 'c_l': 120, 'prem': 5},  # Condor 1
    {'p_l': 85, 'p_s': 95, 'c_s': 105, 'c_l': 115, 'prem': 4},  # Condor 2
]

# 3. Calculate Combined Payoff
total_payoff = np.zeros(len(sT))
for i, condor in enumerate(condors):
    payoff = calculate_iron_condor_payoff(sT, condor['p_l'], condor['p_s'], 
                                          condor['c_s'], condor['c_l'], condor['prem'])
    total_payoff += payoff
    plt.plot(sT, payoff, '--', label=f'Condor {i+1}')

# 4. Plot Combined Results
plt.figure(figsize=(10, 6))
plt.plot(sT, total_payoff, label='Total Portfolio P&L', color='black', linewidth=3)
plt.axhline(0, color='red', linestyle='--')
plt.axvline(spot_price, color='blue', linestyle='--', label='Current Spot')
plt.title('Combined Profit/Loss for Multiple Iron Condors')
plt.xlabel('Stock Price at Expiry')
plt.ylabel('Profit/Loss')
plt.legend()
plt.grid(True)
plt.show()
