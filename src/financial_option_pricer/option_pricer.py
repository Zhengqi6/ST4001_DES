import numpy as np
from scipy.stats import norm

def black_scholes_european_call(S, K, T, r, sigma):
    """
    Calculates the price of a European call option using the Black-Scholes formula.

    Args:
        S (float): Current spot price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        r (float): Annual risk-free interest rate (e.g., 0.05 for 5%).
        sigma (float): Annual volatility of the underlying asset (e.g., 0.2 for 20%).

    Returns:
        float: Price of the European call option.
    """
    if T == 0: # If option is at expiry
        return np.maximum(0, S - K)
    if sigma == 0: # If volatility is zero, no uncertainty
        return np.maximum(0, S * np.exp(-r * T) - K * np.exp(-r * T)) # Discounted payoff
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def black_scholes_european_put(S, K, T, r, sigma):
    """
    Calculates the price of a European put option using the Black-Scholes formula.

    Args:
        S (float): Current spot price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        r (float): Annual risk-free interest rate.
        sigma (float): Annual volatility of the underlying asset.

    Returns:
        float: Price of the European put option.
    """
    if T == 0: # If option is at expiry
        return np.maximum(0, K - S)
    if sigma == 0: # If volatility is zero, no uncertainty
        return np.maximum(0, K * np.exp(-r * T) - S * np.exp(-r * T))

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return put_price

def price_european_option(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility, option_type='call', model_type='BlackScholes'):
    """
    Prices a European option. Currently supports Black-Scholes.
    Args:
        spot_price (float): Current spot price.
        strike_price (float): Option strike price.
        time_to_maturity_years (float): Time to maturity in years.
        risk_free_rate (float): Annual risk-free rate.
        volatility (float): Annual volatility of the underlying.
        option_type (str): 'call' or 'put'.
        model_type (str): Pricing model to use (default 'BlackScholes').
    Returns:
        float: Calculated option price.
    """
    if model_type == 'BlackScholes':
        if option_type.lower() == 'call':
            return black_scholes_european_call(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility)
        elif option_type.lower() == 'put':
            return black_scholes_european_put(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility)
        else:
            raise ValueError("option_type must be 'call' or 'put' for BlackScholes.")
    else:
        raise NotImplementedError(f"Pricing model '{model_type}' is not yet implemented.")

if __name__ == '__main__':
    # Example from data_preparation
    from src.utils.data_preparation import get_financial_option_specs, RISK_FREE_RATE, CARBON_PRICE_BASELINE, CARBON_PRICE_VOLATILITY_ANNUAL

    option_specs_list = get_financial_option_specs()
    r = RISK_FREE_RATE
    S0 = CARBON_PRICE_BASELINE
    sigma_annual = CARBON_PRICE_VOLATILITY_ANNUAL # This is a key input, could come from GARCH model's long-run vol or implied vol

    print(f"Pricing options with Spot={S0}, Risk-free rate={r*100:.2f}%, Annual Volatility={sigma_annual*100:.2f}%\n")

    for spec in option_specs_list:
        if spec['option_type'] == 'call':
            K = spec['strike_price']
            T = spec['time_to_maturity_years']
            
            # For demonstration, use the constant annual volatility.
            # In a real setup, volatility for option pricing might be derived from:
            # 1. Historical volatility over a period matching T.
            # 2. Implied volatility from market option prices (if available).
            # 3. Forecasted volatility from a GARCH model for period T.
            current_sigma = sigma_annual # Simplification for this example
            
            call_price = price_european_option(S0, K, T, r, current_sigma, option_type='call')
            print(f"European Call Option ({spec['label']}): Strike={K}, Maturity={T*12:.0f}M, Price={call_price:.2f} CNY")

    # Example of pricing based on scenario paths from carbon_pricer
    # This would typically involve averaging option payoffs over many paths if using Monte Carlo for pricing,
    # or using path-specific volatilities if the model provides them.
    # For Black-Scholes, we usually need a single volatility parameter per option contract.
    
    # If we wanted to price an option for each scenario end-price (less common for BS, more for MC payoff averaging):
    # from src.carbon_pricer.carbon_price_models import generate_synthetic_historical_prices, generate_price_scenarios_gbm
    # hist_p = generate_synthetic_historical_prices()
    # scenarios = generate_price_scenarios_gbm(initial_price=hist_p.iloc[-1], drift=0.02, volatility=0.2, horizon_days=30, num_scenarios=3)
    # if scenarios is not None:
    #     last_day_prices = scenarios.iloc[-1]
    #     print("\nExample pricing for an option expiring in 30 days, based on scenario end prices (as S at expiry):")
    #     for i, s_at_expiry in enumerate(last_day_prices):
    #         # Here, T would be effectively 0, so payoff is max(0, s_at_expiry - K)
    #         # This is not Black-Scholes pricing before expiry, but payoff calculation.
    #         payoff = np.maximum(0, s_at_expiry - 200) # Example K=200
    #         print(f"Scenario {i+1} payoff at expiry (S={s_at_expiry:.2f}, K=200): {payoff:.2f}") 