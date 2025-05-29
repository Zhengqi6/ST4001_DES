import numpy as np
import pandas as pd

def get_futures_price(spot_price: float, risk_free_rate: float, time_to_maturity_years: float) -> float:
    """
    Calculates the theoretical price of a futures contract.
    F0 = S0 * exp(r * T)

    Args:
        spot_price (float): Current spot price of the underlying asset.
        risk_free_rate (float): Annual risk-free interest rate (e.g., 0.05 for 5%).
        time_to_maturity_years (float): Time to maturity (in years).

    Returns:
        float: Theoretical futures price.
    """
    if time_to_maturity_years < 0:
        raise ValueError("Time to maturity cannot be negative.")
    if spot_price < 0:
        raise ValueError("Spot price cannot be negative.")
        
    futures_price = spot_price * np.exp(risk_free_rate * time_to_maturity_years)
    return futures_price

def calculate_fair_swap_rate(carbon_prices_at_maturity: pd.Series) -> float:
    """
    Calculates the fair fixed rate for a carbon swap based on expected future spot prices.
    This is typically the mean of the simulated spot prices at the swap's maturity.

    Args:
        carbon_prices_at_maturity (pd.Series): A pandas Series of simulated carbon prices
                                                 at the swap's maturity date, across all scenarios.
    Returns:
        float: The fair fixed swap rate.
    """
    if carbon_prices_at_maturity is None or carbon_prices_at_maturity.empty:
        raise ValueError("Carbon prices at maturity series cannot be None or empty.")
    if not isinstance(carbon_prices_at_maturity, pd.Series):
        raise TypeError("carbon_prices_at_maturity must be a pandas Series.")
        
    fair_rate = carbon_prices_at_maturity.mean()
    return fair_rate

if __name__ == '__main__':
    # Example usage
    spot = 300.0
    rate = 0.05
    time_1m = 1/12
    time_3m = 3/12

    f_1m = get_futures_price(spot, rate, time_1m)
    f_3m = get_futures_price(spot, rate, time_3m)

    print(f"Spot Price: {spot}")
    print(f"Risk-Free Rate: {rate*100:.2f}%")
    print(f"Futures Price (1 Month Maturity): {f_1m:.2f}")
    print(f"Futures Price (3 Months Maturity): {f_3m:.2f}")

    # Example usage for calculate_fair_swap_rate
    # Import pandas for the example Series
    import pandas as pd # Make sure pandas is imported if not already for the main block
    example_prices_at_maturity_1m = pd.Series([290, 305, 310, 295, 315]) # Example prices for 1-month maturity
    example_prices_at_maturity_3m = pd.Series([280, 310, 320, 300, 330]) # Example prices for 3-month maturity
    
    fair_swap_rate_1m = calculate_fair_swap_rate(example_prices_at_maturity_1m)
    fair_swap_rate_3m = calculate_fair_swap_rate(example_prices_at_maturity_3m)
    print(f"Fair Swap Rate (1 Month, based on example scenarios): {fair_swap_rate_1m:.2f}")
    print(f"Fair Swap Rate (3 Months, based on example scenarios): {fair_swap_rate_3m:.2f}") 