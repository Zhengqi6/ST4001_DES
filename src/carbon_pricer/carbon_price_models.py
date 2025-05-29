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
    Generates carbon price paths using a Merton Jump-Diffusion Model.
    The model for the log-price d(lnS) is:
    d(lnS_t) = (mu - lambda*kappa - 0.5*sigma^2)dt + sigma*dW_t + dN_t * Y_jump
    where:
    - mu is the total drift (config: 'drift')
    - sigma is the volatility of the Brownian motion part (config: 'volatility')
    - lambda is the jump intensity (average number of jumps per year) (config: 'jump_intensity')
    - dW_t is the Wiener process
    - N_t is a Poisson process with intensity lambda. dN_t = 1 if jump, 0 otherwise.
    - Y_jump is the log of the jump size, Y_jump ~ N(jump_mean_log, jump_std_log^2).
      We take 'jump_mean' as jump_mean_log and 'jump_std' as jump_std_log.
    - kappa = E[exp(Y_jump)-1] = exp(jump_mean_log + 0.5*jump_std_log^2) - 1. This is the expected relative jump size if Y is the multiplier S_after/S_before.
    The term (mu - lambda*kappa) is the drift of the continuous part of the price process dS/S.
    The term (mu - lambda*kappa - 0.5*sigma^2) is the drift of the d(lnS) process's continuous part.
    """
    dt = 1/252  # Daily steps, assuming 252 trading days a year

    # Calculate kappa: E[exp(Y_jump) - 1]
    # Y_jump is log(JumpSizeMultiplier), so JumpSizeMultiplier = exp(Y_jump)
    # E[JumpSizeMultiplier] = exp(jump_mean + 0.5 * jump_std**2)
    kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1

    # Adjusted drift for the continuous part of d(lnS)
    # This is mu' = mu - lambda*kappa for the price process dS/S = mu'*dt + sigma*dW + (Y-1)*dN
    # For d(lnS) = (mu' - 0.5*sigma^2)dt + sigma*dW + Y_jump*dN
    # where mu is the overall expected return E[dS/S]/dt
    # The drift for the GBM component of log-returns:
    gbm_drift_log = drift - jump_intensity * kappa - 0.5 * volatility**2

    paths = []
    for i in range(num_scenarios):
        log_prices = [np.log(initial_price)]
        for _ in range(1, horizon_days):
            # Continuous part (GBM)
            gbm_increment = gbm_drift_log * dt + volatility * np.sqrt(dt) * np.random.normal()

            # Jump part
            jump_val_log = 0.0
            if np.random.poisson(jump_intensity * dt) > 0: # Check if a jump occurs in this dt
                # Draw jump size (log of the multiplicative factor)
                jump_val_log = np.random.normal(jump_mean, jump_std)
            
            current_log_price = log_prices[-1] + gbm_increment + jump_val_log
            log_prices.append(current_log_price)
        paths.append(np.exp(log_prices))

    prices_array = np.array(paths).T # Transpose so rows are time, columns are scenarios

    if start_datetime is None:
        start_datetime = datetime.datetime.now()

    timestamps = []
    current_time = start_datetime
    for _ in range(horizon_days):
        timestamps.append(current_time)
        current_time += datetime.timedelta(days=1)

    df_prices = pd.DataFrame(prices_array, index=timestamps[:horizon_days],
                             columns=[f'scenario_{i+1}' for i in range(num_scenarios)])
    return df_prices

def generate_price_scenarios_regime_switching(initial_price, params_regime1, params_regime2, transition_matrix, horizon_days, num_scenarios, start_datetime=None):
    """
    Generates carbon price paths using a Regime-Switching Model.
    Switches between two GBM regimes based on a transition matrix.
    The log-price d(lnS) in regime i follows:
    d(lnS_t) = (mu_i - 0.5*sigma_i^2)dt + sigma_i*dW_t
    params_regime1/2: dicts with {'drift': float, 'volatility': float} for regime 0 and 1 respectively.
                      'drift' is mu_i, 'volatility' is sigma_i.
    transition_matrix: 2x2 numpy array, P_ij = probability of moving from regime i to regime j in the next step.
                       [[P_00, P_01], [P_10, P_11]]
                       Rows must sum to 1. P_01 is prob of switching from 0 to 1. P_10 is prob of switching from 1 to 0.
    """
    dt = 1/252  # Daily steps

    regime_params = [params_regime1, params_regime2] # Store params in a list for easy access

    # Validate transition matrix rows sum to 1 (approximately)
    if not (np.allclose(np.sum(transition_matrix, axis=1), 1.0)):
        raise ValueError("Rows of the transition matrix must sum to 1.")

    paths = []
    for i in range(num_scenarios):
        log_prices = [np.log(initial_price)]
        # Assume starting in regime 0 for all scenarios.
        # A more advanced approach could use stationary distribution of the Markov chain if it exists and is unique.
        current_regime = 0 
        
        regime_drifts = [p.get('drift', 0.0) for p in regime_params]
        regime_vols = [p.get('volatility', 0.0) for p in regime_params]

        for _ in range(1, horizon_days):
            drift_val = regime_drifts[current_regime]
            vol_val = regime_vols[current_regime]

            # Log-price increment for the current regime
            log_increment = (drift_val - 0.5 * vol_val**2) * dt + vol_val * np.sqrt(dt) * np.random.normal()
            current_log_price = log_prices[-1] + log_increment
            log_prices.append(current_log_price)

            # Determine next regime
            rand_val = np.random.rand()
            if current_regime == 0:
                if rand_val > transition_matrix[0, 0]: # Prob of staying in 0 is P_00
                    current_regime = 1 # Switch to regime 1 (with prob P_01 = 1 - P_00)
            else: # current_regime == 1
                if rand_val > transition_matrix[1, 1]: # Prob of staying in 1 is P_11
                    current_regime = 0 # Switch to regime 0 (with prob P_10 = 1 - P_11)
        
        paths.append(np.exp(log_prices))

    prices_array = np.array(paths).T

    if start_datetime is None:
        start_datetime = datetime.datetime.now()

    timestamps = []
    current_time = start_datetime
    for _ in range(horizon_days):
        timestamps.append(current_time)
        current_time += datetime.timedelta(days=1)

    df_prices = pd.DataFrame(prices_array, index=timestamps[:horizon_days],
                             columns=[f'scenario_{i+1}' for i in range(num_scenarios)])
    return df_prices

# --- Advanced Model Implementations (Conceptual Placeholders) ---
# These would require significant additional work and data for calibration.

# For Kou Jump Diffusion (double exponential jumps):
# def generate_price_scenarios_kou_jump_diffusion(...):
#     pass # Placeholder for Kou model

# For Heston Stochastic Volatility:
# def generate_price_scenarios_heston(...):
#     pass # Placeholder for Heston model

# For Bates (Stochastic Volatility + Jumps):
# def generate_price_scenarios_bates(...):
#     pass # Placeholder for Bates model


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
        print("Jump-Diffusion Scenarios Head:")
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
        print("Regime-Switching Scenarios Head:")
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

    # --- Example Usage ---
    test_start_time = datetime.datetime(2024, 1, 1, 10, 30, 0) # Example fixed start time

    # 1. Synthetic Historical Prices for GARCH
    print("\n--- Generating Synthetic Historical Prices for GARCH ---")
    hist_prices = generate_synthetic_historical_prices(days=500, initial_price=140, mu=0.015, sigma=0.25)
    print(f"Shape of GARCH historical prices: {hist_prices.shape}")
    print(hist_prices.head())

    # 2. Fit GARCH model
    if not hist_prices.empty:
        garch_model_fit = fit_garch_model(hist_prices)
        if garch_model_fit:
            print("\nGARCH Model Summary:")
            print(garch_model_fit.summary())
            
            # 3. Generate scenarios from fitted GARCH
            garch_scenarios = generate_price_scenarios_garch(
                garch_fit=garch_model_fit,
                initial_price=140, # Start scenarios from the current price
                horizon_days=60,
                num_scenarios=3,
                start_datetime=test_start_time
            )
            if garch_scenarios is not None:
                print("\nGARCH Scenarios Head:")
                print(garch_scenarios.head())
                print(f"Shape of GARCH scenarios: {garch_scenarios.shape}")
                if not garch_scenarios.empty:
                     print(f"Timestamps from {garch_scenarios.index[0]} to {garch_scenarios.index[-1]}")

            else:
                print("Failed to generate GARCH scenarios.")
        else:
            print("Failed to fit GARCH model. Skipping GARCH scenario generation.")
    else:
        print("Historical prices are empty. Skipping GARCH.")

    # 4. Jump-Diffusion Model Example (currently falls back to GBM)
    print("\n--- Generating Jump-Diffusion Model Scenarios (Placeholder Fallback to GBM) ---")
    jd_params = {'drift': 0.05, 'volatility': 0.2}
    jd_scenarios = generate_price_scenarios_jump_diffusion(
        initial_price=100,
        drift=jd_params['drift'],
        volatility=jd_params['volatility'],
        jump_intensity=0.1, # Example: average 0.1 jumps per year
        jump_mean=0.0,    # Example: mean jump size (log terms)
        jump_std=0.1,     # Example: std dev of jump size (log terms)
        horizon_days=60,
        num_scenarios=3,
        start_datetime=test_start_time
    )
    print(f"Shape of Jump-Diffusion scenarios: {jd_scenarios.shape}")
    print(jd_scenarios.head())

    # 5. Regime-Switching Model Example (currently falls back to GBM)
    print("\n--- Generating Regime-Switching Model Scenarios (Placeholder Fallback to GBM) ---")
    rs_params1 = {'drift': 0.01, 'volatility': 0.15}
    rs_params2 = {'drift': 0.05, 'volatility': 0.35}
    rs_trans_matrix = np.array([[0.98, 0.02], [0.03, 0.97]])
    rs_scenarios = generate_price_scenarios_regime_switching(
        initial_price=200, 
        params_regime1=rs_params1, 
        params_regime2=rs_params2,
        transition_matrix=rs_trans_matrix,
        horizon_days=60, 
        num_scenarios=3,
        start_datetime=test_start_time
    )
    print(f"Shape of Regime-Switching (GBM fallback) scenarios: {rs_scenarios.shape}")
    print(rs_scenarios.head())

    # Testing the placeholder functions directly for their fallback messages
    print("\n--- Direct Test of Placeholder Fallback Messages ---")
    print("Testing Jump-Diffusion placeholder direct call:")
    _ = generate_price_scenarios_jump_diffusion(100, 0.05, 0.2, 0.1, 0, 0.1, 10, 2, test_start_time)

    print("\nTesting Regime-Switching placeholder direct call:")
    _ = generate_price_scenarios_regime_switching(100, rs_params1, rs_params2, rs_trans_matrix, 10, 2, test_start_time)
