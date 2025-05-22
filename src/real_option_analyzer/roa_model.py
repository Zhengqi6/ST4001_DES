import numpy as np
import pandas as pd

def build_binomial_lattice(S0, u, d, N):
    """Builds a binomial lattice for the underlying asset price."""
    lattice = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            lattice[j, i] = S0 * (u**(i - j)) * (d**j)
    return lattice

def value_american_call_on_binomial_lattice(S0, K, T, r, sigma, N, underlying_value_func, investment_cost_func, project_params):
    """
    Values an American-style call option (like a deferral/investment option) using a binomial lattice.

    Args:
        S0 (float): Initial value of the underlying asset (e.g., current carbon price or initial NPV of project cash flows driven by it).
        K (float): Strike price (investment cost). This can be a function of time or underlying if needed.
        T (float): Time to maturity of the option (max deferral period in years).
        r (float): Annual risk-free interest rate.
        sigma (float): Annual volatility of the underlying asset value (S0).
        N (int): Number of time steps in the binomial lattice.
        underlying_value_func (function): A function that takes the current state of the stochastic variable (e.g., carbon price)
                                         at a node and other project_params, and returns the expected NPV of the project
                                         if undertaken *at that point in time* (excluding the initial investment cost K).
                                         Signature: func(stochastic_var_at_node, project_params, time_step_dt, remaining_steps)
        investment_cost_func (function): A function that returns the investment cost. Can be time-dependent.
                                        Signature: func(time_t, project_params)
        project_params (dict): Dictionary of project-specific parameters needed by underlying_value_func and investment_cost_func.

    Returns:
        float: Value of the American call (real) option.
        np.ndarray: The option value lattice.
        np.ndarray: The underlying asset value lattice.
    """
    dt = T / N  # Length of each time step

    # Binomial parameters
    u = np.exp(sigma * np.sqrt(dt))  # Up-factor
    d = 1 / u  # Down-factor
    q = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability of an up move

    if not (0 < q < 1):
        print(f"Warning: Risk-neutral probability q = {q:.4f} is not between 0 and 1. Check parameters (r, sigma, dt).")
        # Fallback or error, for simplicity, may proceed but results could be invalid
        # q = np.clip(q, 0.001, 0.999) # Simple clipping, not ideal
        if q <=0 and np.exp(r*dt) < d : print("r*dt is too low or d is too high")
        if q >=1 and np.exp(r*dt) > u : print("r*dt is too high or u is too low")
        # A common issue is if r*dt makes exp(r*dt) fall outside [d, u]
        # This implies arbitrage or inconsistent parameters.

    # 1. Initialize asset prices lattice (here, the stochastic variable, e.g., carbon price)
    stochastic_var_lattice = np.zeros((N + 1, N + 1))
    for i in range(N + 1): # Time steps from 0 to N
        for j in range(i + 1): # Number of down moves
            stochastic_var_lattice[j, i] = S0 * (u**(i - j)) * (d**j)

    # 2. Initialize option values lattice
    option_values = np.zeros((N + 1, N + 1))

    # 3. Calculate option values at maturity (time N)
    # At maturity, the option is exercised if Project_NPV_at_maturity > Investment_Cost_at_maturity
    # Value is max(0, Project_NPV_at_maturity - Investment_Cost_at_maturity)
    for j in range(N + 1):
        carbon_price_at_maturity = stochastic_var_lattice[j, N]
        # Project NPV if invested at this point (sum of future cash flows from this point, discounted to this point)
        project_npv_at_node = underlying_value_func(carbon_price_at_maturity, project_params, dt, 0) # 0 remaining steps for valuation
        current_investment_cost = investment_cost_func(T, project_params) # Investment cost at time T
        option_values[j, N] = np.maximum(0, project_npv_at_node - current_investment_cost)

    # 4. Backward induction for option values
    for i in range(N - 1, -1, -1):  # Iterate backwards from N-1 to 0 (today)
        current_time_t = i * dt
        current_investment_cost = investment_cost_func(current_time_t, project_params)
        for j in range(i + 1):
            # Value if exercised now
            carbon_price_at_node = stochastic_var_lattice[j, i]
            project_npv_if_invest_now = underlying_value_func(carbon_price_at_node, project_params, dt, N-i) # N-i remaining steps for valuation
            value_if_exercised = np.maximum(0, project_npv_if_invest_now - current_investment_cost)
            
            # Value if held (defer)
            expected_future_option_value = (q * option_values[j, i + 1] + (1 - q) * option_values[j + 1, i + 1])
            value_if_held = np.exp(-r * dt) * expected_future_option_value
            
            option_values[j, i] = np.maximum(value_if_exercised, value_if_held)
            
    return option_values[0, 0], option_values, stochastic_var_lattice

# --- Helper functions for the CCS ROA case study ---
def ccs_project_npv_at_node(carbon_price_at_node, project_params, time_step_dt, remaining_steps_in_option_life):
    """
    Calculates the expected NPV of the CCS project if investment is made when carbon price is 'carbon_price_at_node'.
    This NPV is for cash flows *after* the investment point, over the project's operational lifetime.
    The carbon price itself is assumed to follow its own stochastic process for these future cash flows.
    For simplicity here, we can either:
    1. Assume this carbon_price_at_node persists (less realistic for long-lived projects).
    2. Project a new GBM from this carbon_price_at_node for the project life (more complex within lattice step).
    3. Use carbon_price_at_node as the *average expected* price over the project life for simplification.
    
    Let's use method 3 for initial simplicity in this example function.
    A more rigorous approach would involve a nested simulation or a more complex lattice for the project value itself.
    """
    annual_net_cash_flow = (
        carbon_price_at_node * project_params['chp_co2_emission_ton_per_m3_gas'] * 
        (project_params['chp_annual_generation_kwh_assumed'] * project_params['chp_gas_consumption_m3_per_kwh_e']) * 
        project_params['capture_efficiency']
        - project_params['opex_increase_per_kwh_chp_cny'] * project_params['chp_annual_generation_kwh_assumed']
    )
    
    # Discount these annual cash flows over the project lifetime
    npv = 0
    for year in range(1, int(project_params['project_lifetime_years']) + 1):
        npv += annual_net_cash_flow / ((1 + project_params['risk_free_rate'])**year)
        
    return npv

def ccs_investment_cost_func(time_t, project_params):
    """Returns the investment cost. Can be made time-dependent if needed (e.g., cost erosion)."""
    return project_params['investment_cost_cny']


if __name__ == '__main__':
    from src.utils.data_preparation import get_roa_ccs_project_parameters

    # --- ROA Parameters for CCS Deferral Option ---
    roa_params = get_roa_ccs_project_parameters()

    S0_carbon = roa_params['carbon_price_initial_cny_per_ton'] # Initial carbon price
    # K_invest = roa_params['investment_cost_cny'] # This is handled by investment_cost_func
    T_defer = roa_params['max_deferral_years']      # Max deferral period (option life)
    r_risk_free = roa_params['risk_free_rate']
    sigma_carbon_vol = roa_params['carbon_price_gbm_volatility'] # Volatility of the carbon price
    N_steps = 100  # Number of steps in binomial tree for the option life

    print("--- Valuing CCS Deferral Option using Binomial Lattice ---")
    print(f"Initial Carbon Price (S0 for lattice): {S0_carbon:.2f} CNY/ton")
    print(f"Max Deferral Period (Option Life T): {T_defer} years")
    print(f"Risk-Free Rate: {r_risk_free*100:.2f}% (annual)")
    print(f"Carbon Price Volatility (for lattice): {sigma_carbon_vol*100:.2f}% (annual)")
    print(f"Number of Lattice Steps (N): {N_steps}")
    print(f"CCS Investment Cost: {roa_params['investment_cost_cny']:.2f} CNY")
    print(f"CCS Project Lifetime: {roa_params['project_lifetime_years']} years")

    # The underlying for the option is the NPV of the project, which is a function of carbon price.
    # So, the lattice is built for the carbon price, and at each node, we calculate the project's NPV.
    
    option_value, option_lattice, carbon_price_lattice = value_american_call_on_binomial_lattice(
        S0=S0_carbon, 
        K=None, # K is handled by the investment_cost_func and compared with underlying_value_func output
        T=T_defer,
        r=r_risk_free,
        sigma=sigma_carbon_vol,
        N=N_steps,
        underlying_value_func=ccs_project_npv_at_node, # Function to get project NPV (excl. invest cost) at a node
        investment_cost_func=ccs_investment_cost_func, # Function to get investment cost
        project_params=roa_params
    )

    print(f"\nCalculated CCS Deferral Option Value: {option_value:.2f} CNY")

    # To find the optimal investment threshold (critical carbon price):
    # We need to check at t=0 (or first few steps) at what carbon price in the lattice 
    # the decision to invest immediately is optimal (i.e., exercise value > hold value).
    # This is more involved as it requires inspecting the decision at each node.
    # For simplicity, we can say if option_value > 0, deferral has value over immediate NPV based on S0_carbon.
    
    # Calculate traditional NPV at S0_carbon for comparison:
    npv_at_S0_if_invest_now = ccs_project_npv_at_node(S0_carbon, roa_params, T_defer/N_steps, N_steps) # Full life from t=0
    traditional_npv = npv_at_S0_if_invest_now - roa_params['investment_cost_cny']
    v_roa = traditional_npv + option_value # This is not quite right; V_ROA = NPV_expanded = max(Traditional NPV, Option Value + PV(Exercise Price) if option is on NPV itself)
                                        # Or, V_ROA is the option_value if it's an option TO INVEST. If option_value is positive, it IS the value of flexibility.
                                        # The value of the *project with the option* is E[PV(cash flows)] - E[PV(investment)] where investment is optimally timed.
                                        # The `option_value` from the lattice *is* the value of this optimal timing strategy relative to a fixed strategy.

    print(f"Traditional NPV (invest now at S0={S0_carbon}): {traditional_npv:.2f} CNY")
    # The option_value itself represents the additional value from flexibility (deferral).
    # So, the total project value with deferral option (Expanded NPV) is: Traditional NPV (if positive) + Option Value, or just Option Value if Trad. NPV is negative.
    # More accurately, the value of the *opportunity* is the option_value. If invested, this value is realized or foregone.
    # The lattice directly calculates the value of the opportunity including optimal exercise.
    
    print(f"Value of Project WITH Deferral Option (at t=0): {option_value + traditional_npv if traditional_npv > 0 else option_value:.2f} CNY (approx Expanded NPV)") 
    # The option_values[0,0] IS the expanded NPV, assuming K was subtracted in payoff.
    # Let's re-check the formulation: option_values[j,N] = max(0, NPV_at_node - K).
    # And option_values[j,i] = max(NPV_at_node - K, discounted_continuation_value).
    # So, option_values[0,0] is indeed the expanded NPV (value of project including flexibility).
    print(f"Expanded NPV (Project value with deferral flexibility): {option_value:.2f} CNY (This is V_ROA)")

    # Note on Investment Threshold:
    # The threshold carbon price is the price at which immediate investment value equals continuation value.
    # This can be found by iterating through the first column of the `option_lattice` (for t=0 decisions)
    # and finding where `value_if_exercised` (i.e., `ccs_project_npv_at_node(...) - investment_cost_func(...)`)
    # becomes approximately equal to `option_values[j,0]` and is positive.
    # Or, where `ccs_project_npv_at_node(...) - investment_cost_func(...)` is just above the discounted expected future option value if not exercised.
    # This is a bit more complex to extract directly without inspecting the decision logic at each node during the backward pass.
    # For now, the key output is the option_value itself.
    
    # Plotting the first few steps of the carbon price lattice (underlying)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    steps_to_plot = min(5, N_steps)
    for i in range(steps_to_plot + 1):
        plt.plot([i]* (i+1) , carbon_price_lattice[:i+1, i], 'ro-', alpha=0.3)
    plt.title(f'Carbon Price Binomial Lattice (First {steps_to_plot} steps)')
    plt.xlabel('Time Step')
    plt.ylabel('Carbon Price (CNY/ton)')
    plt.show() 