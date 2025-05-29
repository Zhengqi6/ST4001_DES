import pandas as pd
import numpy as np
from arch import arch_model
import datetime

def generate_synthetic_historical_prices(days=1000, initial_price=150, mu=0.02, sigma=0.2):
    """Generates a synthetic daily GBM price series for fitting GARCH."""
    dt = 1/252 # Assume 252 trading days
    prices = [initial_price]
    for _ in range(1, days):
        drift = mu * dt
        shock = sigma * np.sqrt(dt) * np.random.normal()
        price = prices[-1] * np.exp(drift + shock)
        prices.append(price)
    
    # Create a DatetimeIndex
    # Ensure we have a start date that allows for 'days' number of business days
    start_date = datetime.datetime.now() - pd.tseries.offsets.BDay(days)
    date_index = pd.bdate_range(start=start_date, periods=days)
    
    return pd.Series(prices, index=date_index)

def fit_garch_model(historical_prices):
    """Fits a GARCH(1,1) model to historical price returns."""
    returns = 100 * historical_prices.pct_change().dropna()
    if returns.empty or returns.std() == 0: # Handle cases with no variance or too few data points
        print("Warning: Returns series is empty or has no variance. Cannot fit GARCH model.")
        return None
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        #model = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, rescale=False) # Alternative with mean
        #model = arch_model(returns, p=1, q=1) # Simpler call, arch defaults mean to zero
        garch_fit = model.fit(disp='off', show_warning=False)
        return garch_fit
    except Exception as e:
        print(f"Error fitting GARCH model: {e}")
        return None

def generate_price_scenarios_gbm(initial_price, drift, volatility, horizon_days, num_scenarios, start_datetime=None):
    """
    Generates carbon price paths using Geometric Brownian Motion.
    Ensures timestamps include H:M:S.ms for consistency.
    """
    dt = 1/252 # Daily steps, assuming 252 trading days a year
    # prices = np.zeros((horizon_days, num_scenarios))
    # prices[0] = initial_price
    
    # Store paths in a list of lists first
    paths = []

    for i in range(num_scenarios):
        path = [initial_price]
        for _ in range(1, horizon_days):
            price_drift = drift * dt
            shock = volatility * np.sqrt(dt) * np.random.normal()
            price = path[-1] * np.exp(price_drift + shock)
            path.append(price)
        paths.append(path)

    # Transpose paths so that rows are time steps and columns are scenarios
    prices_array = np.array(paths).T

    if start_datetime is None:
        start_datetime = datetime.datetime.now()
    
    # Generate date range with specific time component
    time_component = start_datetime.time()
    dates = pd.date_range(start=start_datetime.date(), periods=horizon_days, freq='B') # Business days
    
    # Combine date with the fixed time component, then adjust to ensure unique timestamps for CSV
    timestamps = []
    current_time = start_datetime
    for d in dates:
        # We use the date from date_range and time from start_datetime, then add microseconds for uniqueness
        # Or, more simply, increment from the precise start_datetime
        timestamps.append(current_time)
        current_time += datetime.timedelta(days=1) # For daily scenarios, simply add a day

    df_prices = pd.DataFrame(prices_array, index=timestamps[:horizon_days], 
                             columns=[f'scenario_{i+1}' for i in range(num_scenarios)])
    return df_prices


def generate_price_scenarios_garch(garch_fit, initial_price, horizon_days, num_scenarios, start_datetime=None):
    """
    Generates price paths using a fitted GARCH model's forecast simulations.
    Ensures timestamps include H:M:S.ms for consistency.
    """
    if garch_fit is None:
        return None
    try:
        # Simulate paths for returns
        # The 'simulations' method returns an object from which we can extract paths
        # We need to provide the initial variance (sigma2) and initial returns (resid) for the simulations
        # Get the last conditional variance and standardized residual
        last_obs_date = garch_fit.resid.index[-1]
        # sigma2_0 = garch_fit.conditional_volatility[last_obs_date]**2 # More direct for GARCH package
        # resid_0 = garch_fit.resid[last_obs_date] # More direct for GARCH package

        # For the 'arch' package, simulations are based on forecast objects.
        # forecast = garch_fit.forecast(horizon=horizon_days, method='simulation', simulations=num_scenarios)
        # simulated_mean_returns = forecast.mean.values.T # shape (horizon, num_scenarios)
        # simulated_variances = forecast.variance.values.T # shape (horizon, num_scenarios)
        # # Generate random shocks
        # random_shocks = np.random.normal(size=(horizon_days, num_scenarios))
        # # Calculate returns: r_t = mu_t + sigma_t * e_t
        # simulated_returns = simulated_mean_returns + np.sqrt(simulated_variances) * random_shocks
        
        # Simpler simulation approach using forecast.simulations
        # The 'arch' library's forecast method with 'simulation' is preferred
        sim_results = garch_fit.forecast(horizon=horizon_days, method='simulation', simulations=num_scenarios)
        simulated_returns_paths = sim_results.simulations.values # This should give (num_paths, horizon, num_scenarios_per_path)

        # The output shape of sim_results.simulations.values might be (1, horizon_days, num_scenarios) or (num_scenarios, horizon_days)
        # Let's check and adjust. Typically, it's (num_draws, horizon, n_simulated_series_for_each_draw)
        # For a single model fit, we usually have one "draw" of the model parameters.
        # The .values attribute returns a 3D array. We want the first "draw".
        
        # If 'arch' version >= 5.0, .simulations.values is (nobs x horizon x nsims)
        # For older versions, it might be different or require .residual_variances etc.
        # Assuming recent 'arch' version:
        if simulated_returns_paths.ndim == 3:
            simulated_returns = simulated_returns_paths[0].T / 100.0 # Get first "draw", transpose, and scale from percentage
        elif simulated_returns_paths.ndim == 2: # (horizon, nsims)
             simulated_returns = simulated_returns_paths / 100.0 # Already (horizon, nsims), scale from percentage
        else:
            raise ValueError(f"Unexpected shape for GARCH simulated returns: {simulated_returns_paths.shape}")


        # Convert returns to prices
        prices_array = np.zeros((horizon_days, num_scenarios))
        prices_array[0, :] = initial_price
        for t in range(1, horizon_days):
            prices_array[t, :] = prices_array[t-1, :] * (1 + simulated_returns[t-1, :]) # t-1 for returns as it's for period t->t+1

        # Ensure positive prices (common adjustment for GARCH where returns can be large)
        prices_array = np.maximum(prices_array, 0.01) # Floor at a small positive number

        if start_datetime is None:
            start_datetime = datetime.datetime.now()

        timestamps = []
        current_time = start_datetime
        for _ in range(horizon_days):
            timestamps.append(current_time)
            current_time += datetime.timedelta(days=1)

        df_prices = pd.DataFrame(prices_array, index=timestamps,
                                 columns=[f'scenario_{i+1}' for i in range(num_scenarios)])
        return df_prices
    except Exception as e:
        print(f"Error generating GARCH scenarios: {e}")
        return None

def generate_price_scenarios_jump_diffusion(initial_price, drift, volatility, jump_intensity, jump_mean, jump_std, horizon_days, num_scenarios, start_datetime=None):
    """
    Generates carbon price paths using a Merton Jump-Diffusion Model (placeholder).
    Actual implementation would require more sophisticated handling of jump processes.
    """
    print(f"Warning: generate_price_scenarios_jump_diffusion is a placeholder and currently falls back to GBM.")
    # Placeholder: For now, falls back to GBM for structure. 
    # A full implementation would discretize the jump-diffusion process:
    # dS/S = (mu - lambda*kappa)dt + sigma*dW_t + dJ_t
    # where dJ_t is a compound Poisson process. kappa is E[Y-1] where Y is jump size.
    # For simplicity, the GBM function is called here. Replace with actual jump-diffusion logic.
    return generate_price_scenarios_gbm(initial_price, drift, volatility, horizon_days, num_scenarios, start_datetime)

def generate_price_scenarios_regime_switching(initial_price, params_regime1, params_regime2, transition_matrix, horizon_days, num_scenarios, start_datetime=None):
    """
    Generates carbon price paths using a Regime-Switching Model (placeholder).
    Actual implementation would involve simulating state transitions and regime-specific dynamics.
    params_regime1/2 should be dicts like {'drift': ..., 'volatility': ...}
    transition_matrix is a 2x2 np.array for probabilities P_ij.
    """
    print(f"Warning: generate_price_scenarios_regime_switching is a placeholder and currently falls back to GBM using regime1 params.")
    # Placeholder: For now, falls back to GBM using parameters from the first regime for structure.
    # A full implementation would:
    # 1. Simulate the state (regime) sequence using the transition_matrix.
    # 2. For each time step, apply the drift and volatility of the current regime.
    # This is a simplified fallback.
    return generate_price_scenarios_gbm(initial_price, params_regime1.get('drift', 0.02), params_regime1.get('volatility', 0.2), horizon_days, num_scenarios, start_datetime)

if __name__ == '__main__':
    # Demo for GBM
    print("Generating GBM scenarios...")
    initial_carbon_price = 150
    annual_drift = 0.02 
    annual_volatility = 0.15
    days_horizon = 90 
    n_scenarios = 10
    
    # Precise start time for consistency with example output
    demo_start_time = datetime.datetime(2025, 5, 22, 1, 4, 1, 123456)


    gbm_scenarios = generate_price_scenarios_gbm(
        initial_price=initial_carbon_price,
        drift=annual_drift,
        volatility=annual_volatility,
        horizon_days=days_horizon,
        num_scenarios=n_scenarios,
        start_datetime=demo_start_time
    )
    print("GBM Scenarios Head:")
    print(gbm_scenarios.head())
    print(f"GBM DataFrame shape: {gbm_scenarios.shape}")
    if not gbm_scenarios.empty:
        print(f"Timestamps from {gbm_scenarios.index[0]} to {gbm_scenarios.index[-1]}")

    # Demo for GARCH
    print("\nGenerating GARCH scenarios...")
    # 1. Generate synthetic historical data
    hist_prices = generate_synthetic_historical_prices(days=500, initial_price=140, mu=0.015, sigma=0.25)
    print(f"Synthetic historical prices generated: {len(hist_prices)} points.")

    # 2. Fit GARCH model
    if not hist_prices.empty:
        garch_model_fit = fit_garch_model(hist_prices)
        if garch_model_fit:
            print("\nGARCH Model Summary:")
            print(garch_model_fit.summary())
            
            # 3. Generate scenarios from fitted GARCH
            garch_scenarios = generate_price_scenarios_garch(
                garch_fit=garch_model_fit,
                initial_price=initial_carbon_price, # Start scenarios from the current price
                horizon_days=days_horizon,
                num_scenarios=n_scenarios,
                start_datetime=demo_start_time
            )
            if garch_scenarios is not None:
                print("\nGARCH Scenarios Head:")
                print(garch_scenarios.head())
                print(f"GARCH DataFrame shape: {garch_scenarios.shape}")
                if not garch_scenarios.empty:
                     print(f"Timestamps from {garch_scenarios.index[0]} to {garch_scenarios.index[-1]}")

            else:
                print("Failed to generate GARCH scenarios.")
        else:
            print("Failed to fit GARCH model. Skipping GARCH scenario generation.")
    else:
        print("Historical prices are empty. Skipping GARCH.")

    # Demo for Jump-Diffusion (Placeholder)
    print("\nGenerating Jump-Diffusion scenarios (Placeholder)...")
    jump_diffusion_scenarios = generate_price_scenarios_jump_diffusion(
        initial_price=initial_carbon_price,
        drift=annual_drift,
        volatility=annual_volatility,
        jump_intensity=0.1, # Example: average 0.1 jumps per year
        jump_mean=0.0,    # Example: mean jump size (log terms)
        jump_std=0.15,    # Example: std dev of jump size (log terms)
        horizon_days=days_horizon,
        num_scenarios=n_scenarios,
        start_datetime=demo_start_time
    )
    if jump_diffusion_scenarios is not None:
        print("Jump-Diffusion Scenarios Head (Placeholder Fallback to GBM):")
        print(jump_diffusion_scenarios.head())

    # Demo for Regime-Switching (Placeholder)
    print("\nGenerating Regime-Switching scenarios (Placeholder)...")
    regime1_params = {'drift': 0.01, 'volatility': 0.10}
    regime2_params = {'drift': 0.05, 'volatility': 0.30}
    # P_ij = probability of switching from i to j. Rows sum to 1.
    # P = [[P_11, P_12], [P_21, P_22]]
    trans_matrix = np.array([[0.95, 0.05], [0.03, 0.97]]) 
    regime_switching_scenarios = generate_price_scenarios_regime_switching(
        initial_price=initial_carbon_price,
        params_regime1=regime1_params,
        params_regime2=regime2_params,
        transition_matrix=trans_matrix,
        horizon_days=days_horizon,
        num_scenarios=n_scenarios,
        start_datetime=demo_start_time
    )
    if regime_switching_scenarios is not None:
        print("Regime-Switching Scenarios Head (Placeholder Fallback to GBM with Regime1 Params):")
        print(regime_switching_scenarios.head())

    # Example of how it might be used in the main script:
    # model_type_to_use = "GARCH" # or "GBM"
    # final_scenarios = None
    # if model_type_to_use == "GARCH":
    #     if 'garch_model_fit' in locals() and garch_model_fit is not None:
    #         final_scenarios = generate_price_scenarios_garch(...)
    #         if final_scenarios is None:
    #             print("GARCH scenario generation failed, falling back to GBM.")
    #             final_scenarios = generate_price_scenarios_gbm(...)
    #     else:
    #         print("GARCH model not fitted, falling back to GBM for scenarios.")
    #         final_scenarios = generate_price_scenarios_gbm(...)
    # elif model_type_to_use == "GBM":
    #     final_scenarios = generate_price_scenarios_gbm(...)
    
    # if final_scenarios is not None:
    #     print("\nFinal Selected Scenarios (example):")
    #     print(final_scenarios.head())
    #     # final_scenarios.to_csv("carbon_price_scenarios_example.csv")
