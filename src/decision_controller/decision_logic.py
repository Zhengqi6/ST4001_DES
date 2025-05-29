# Placeholder for decision_controller module
# This module will integrate outputs from other modules to make decisions.

import pandas as pd
import numpy as np

def calculate_cvar(costs_series, alpha_level):
    """Calculates Conditional Value at Risk (CVaR) for a series of costs."""
    if costs_series is None or costs_series.empty:
        return np.nan
    # Ensure costs_series is a NumPy array or Pandas Series for percentile calculation
    if not isinstance(costs_series, (pd.Series, np.ndarray)):
        costs_series = pd.Series(costs_series)
    if costs_series.empty: # Double check after conversion
        return np.nan
        
    var = np.percentile(costs_series.dropna(), alpha_level * 100)
    cvar = costs_series[costs_series >= var].mean()
    return cvar

def make_operational_hedging_decision(des_summary, priced_options, carbon_price_scenarios_df, operational_params):
    """
    Makes a decision on operational hedging using financial carbon options.

    Args:
        des_summary (dict): Summary of DES operation from baseline run.
                            Expected to contain 'total_chp_co2_ton'.
        priced_options (list): List of dictionaries, each representing a priced option.
                               Each option dict should have 'label', 'price' (per ton), 
                               'strike_price', 'option_type' (e.g., 'call').
        carbon_price_scenarios_df (pd.DataFrame): DataFrame of carbon price scenarios.
                                                  Index=time, columns=scenarios.
                                                  Assumes prices are daily.
        operational_params (dict): Parameters for decision logic.
                                   Expected: 
                                   'cvar_alpha_level' (float, e.g., 0.95 for 95% CVaR),
                                   'co2_tons_to_hedge_from_des' (str key, e.g. 'total_chp_co2_ton', to get CO2 tons from des_summary),
                                   'hedge_decision_threshold_cvar_reduction_pct' (float, e.g., 0.05 for 5% reduction).

    Returns:
        tuple: (str: decision_action, dict: decision_details)
    """
    print("Decision Controller: Making operational hedging decision...")
    if des_summary is None or not priced_options or carbon_price_scenarios_df is None or carbon_price_scenarios_df.empty:
        print("Decision Controller: Insufficient data for hedging decision (DES summary, options, or scenarios missing).")
        return ("NoHedge", {"reason": "Insufficient input data."})

    alpha_level = operational_params.get('cvar_alpha_level', 0.95)
    co2_tons_key = operational_params.get('co2_tons_to_hedge_from_des', 'total_chp_co2_ton')
    co2_tons_to_hedge = des_summary.get(co2_tons_key, 0)
    min_cvar_reduction_pct = operational_params.get('hedge_decision_threshold_cvar_reduction_pct', 0.05) 

    if co2_tons_to_hedge <= 0:
        print(f"Decision Controller: No CO2 emissions ({co2_tons_key}={co2_tons_to_hedge}) to hedge.")
        return ("NoHedge", {"reason": f"No CO2 emissions ({co2_tons_key}={co2_tons_to_hedge}) to hedge."})

    if carbon_price_scenarios_df.iloc[-1].isnull().all():
        print("Decision Controller: Carbon price scenarios contain all NaNs at the final step.")
        return ("NoHedge", {"reason": "Invalid carbon price scenarios (all NaNs at maturity)."})
        
    future_carbon_prices = carbon_price_scenarios_df.iloc[-1].dropna()
    if future_carbon_prices.empty:
        print("Decision Controller: No valid future carbon prices after dropping NaNs.")
        return ("NoHedge", {"reason": "No valid future carbon prices from scenarios."})

    unhedged_total_carbon_costs = co2_tons_to_hedge * future_carbon_prices
    cvar_unhedged = calculate_cvar(unhedged_total_carbon_costs, alpha_level)
    print(f"Decision Controller: Unhedged CO2 Cost CVaR ({alpha_level*100:.0f}%): {cvar_unhedged:.2f} CNY for {co2_tons_to_hedge:.2f} tons")

    if np.isnan(cvar_unhedged):
        print("Decision Controller: Could not calculate unhedged CVaR.")
        return ("NoHedge", {"reason": "Could not calculate unhedged CVaR."})

    best_hedge_cvar = cvar_unhedged
    best_option_details = None
    recommended_strategy = "NoHedge"

    for option in priced_options:
        if option.get('option_type', '').lower() != 'call':
            continue

        option_cost_per_ton = option.get('price', float('inf'))
        strike_price = option.get('strike_price')
        option_label = option.get('name', 'Unknown Option')

        if strike_price is None:
            print(f"Decision Controller: Skipping option {option_label} due to missing strike price.")
            continue

        hedged_total_carbon_costs_option = co2_tons_to_hedge * (np.minimum(future_carbon_prices, strike_price) + option_cost_per_ton)
        cvar_hedged_option = calculate_cvar(hedged_total_carbon_costs_option, alpha_level)
        print(f"Decision Controller: Option '{option_label}' (Strike: {strike_price:.2f}, Price: {option_cost_per_ton:.2f}) - Hedged CVaR: {cvar_hedged_option:.2f} CNY")

        if not np.isnan(cvar_hedged_option) and cvar_hedged_option < best_hedge_cvar:
            best_hedge_cvar = cvar_hedged_option
            recommended_strategy = "HedgeWithFinancialOption"
            best_option_details = {
                'option_label': option_label,
                'strike_price': strike_price,
                'option_price_per_ton': option_cost_per_ton,
                'quantity_hedged_tons': co2_tons_to_hedge,
                'achieved_cvar': best_hedge_cvar,
                'unhedged_cvar': cvar_unhedged,
                'cvar_reduction_cny': cvar_unhedged - best_hedge_cvar,
                'cvar_reduction_pct': (cvar_unhedged - best_hedge_cvar) / cvar_unhedged if cvar_unhedged > 0 else 0
            }

    if recommended_strategy == "HedgeWithFinancialOption" and best_option_details:
        if best_option_details['cvar_reduction_pct'] >= min_cvar_reduction_pct:
            print(f"Decision Controller: Recommended Hedge: {best_option_details['option_label']} reduces CVaR by {best_option_details['cvar_reduction_pct']:.2%}.")
            return recommended_strategy, best_option_details
        else:
            print(f"Decision Controller: Best option {best_option_details['option_label']} CVaR reduction ({best_option_details['cvar_reduction_pct']:.2%}) is below threshold ({min_cvar_reduction_pct:.2%}). No hedge recommended.")
            return "NoHedge", {"reason": f"Best option {best_option_details['option_label']} CVaR reduction below threshold.", "details": best_option_details}
    
    print("Decision Controller: No suitable financial option found to improve CVaR significantly.")
    return "NoHedge", {"reason": "No suitable option found or CVaR reduction insufficient."}

def make_strategic_investment_decision(roa_results_ccs, strategic_params, market_outlook):
    """
    Makes a strategic investment decision for CCS based on ROA results.

    Args:
        roa_results_ccs (dict): Results from Real Option Analysis for CCS.
                                Expected: 'option_value' (Expanded NPV), 'traditional_npv'.
        strategic_params (dict): Parameters for strategic decision logic.
                                 Expected: 'npv_hurdle_rate_pct' (float, e.g., 0.10 for 10% of investment as hurdle),
                                           'min_option_premium_to_defer_pct' (float, e.g. 0.05 for 5% of investment as min premium to defer).
                                           'investment_cost_key_in_roa_params' (str, used to fetch investment cost for pct calculations)
        market_outlook (dict): Current market conditions.
                               Expected: 'current_carbon_price' (float), 'roa_ccs_project_parameters' (dict, containing CCS investment cost).

    Returns:
        dict: {'project': 'CCS', 'action': 'DEFER'|'INVEST_NOW'|'REJECT', 'reason': str}
    """
    print("Decision Controller: Making strategic CCS investment decision...")
    if roa_results_ccs is None or market_outlook.get('roa_ccs_project_parameters') is None:
        print("Decision Controller: Insufficient ROA data or project parameters for strategic decision.")
        return {'project': 'CCS', 'action': 'DecisionUnavailable', 'reason': 'Missing ROA results or CCS project parameters.'}

    e_npv = roa_results_ccs.get('option_value')
    trad_npv = roa_results_ccs.get('traditional_npv')
    
    if e_npv is None and trad_npv is None:
         return {'project': 'CCS', 'action': 'DecisionUnavailable', 'reason': 'ROA and NPV results are both missing.'}
    if e_npv is None: e_npv = trad_npv if trad_npv is not None else -float('inf') # Ensure e_npv has a value
    if trad_npv is None: trad_npv = -float('inf') # Ensure trad_npv has a value

    investment_cost_key = strategic_params.get('investment_cost_key_in_roa_params', 'investment_cost_cny')
    ccs_project_params = market_outlook['roa_ccs_project_parameters']
    investment_cost = ccs_project_params.get(investment_cost_key, 0)

    if investment_cost <= 0:
        return {'project': 'CCS', 'action': 'DecisionUnavailable', 'reason': f'Invalid investment cost ({investment_cost}) for CCS project.'}

    npv_hurdle_pct = strategic_params.get('npv_hurdle_rate_pct', 0.0) # Default 0% hurdle on investment cost
    min_option_premium_pct = strategic_params.get('min_option_premium_to_defer_pct', 0.05) # Default 5% of investment for premium

    npv_hurdle_value = npv_hurdle_pct * investment_cost
    min_option_premium_value = min_option_premium_pct * investment_cost
    
    current_carbon_price = market_outlook.get('current_carbon_price', 'N/A')
    action = "Undetermined"
    reason = ""
    option_premium = e_npv - trad_npv

    if option_premium > min_option_premium_value:
        if trad_npv < npv_hurdle_value:
            action = "DEFER"
            reason = (f"Significant value in deferral (ROA: {e_npv:,.0f}, NPV: {trad_npv:,.0f}, Premium: {option_premium:,.0f}). "
                      f"NPV below hurdle ({npv_hurdle_value:,.0f}). Carbon Price: {current_carbon_price}.")
        else: 
            action = "DEFER_BUT_MONITOR"
            reason = (f"NPV positive ({trad_npv:,.0f}) & above hurdle, but deferral offers more (ROA: {e_npv:,.0f}, Premium: {option_premium:,.0f}). "
                      f"Consider deferring. Carbon Price: {current_carbon_price}.")
    else:
        if trad_npv >= npv_hurdle_value:
            action = "INVEST_NOW"
            reason = (f"NPV positive ({trad_npv:,.0f}) & above hurdle. Deferral offers limited further upside (Premium: {option_premium:,.0f}). "
                      f"Carbon Price: {current_carbon_price}.")
        else:
            action = "REJECT"
            reason = (f"NPV negative ({trad_npv:,.0f}) or below hurdle. ROA does not provide sufficient uplift (Premium: {option_premium:,.0f}). "
                      f"Carbon Price: {current_carbon_price}.")

    print(f"Decision Controller: CCS Investment Decision: {action}. Reason: {reason}")
    return {'project': 'CCS', 'action': action, 'reason': reason}

if __name__ == '__main__':
    print("--- Testing Operational Hedging Decision Logic ---")
    dummy_des_summary = {'total_chp_co2_ton': 100} 
    dummy_priced_options = [
        {'label': 'Call_200_1M', 'strike_price': 200, 'time_to_maturity_years': 1/12, 'option_type': 'call', 'price': 5.0},
        {'label': 'Call_220_1M', 'strike_price': 220, 'time_to_maturity_years': 1/12, 'option_type': 'call', 'price': 2.5},
        {'label': 'Call_180_1M_Exp', 'strike_price': 180, 'time_to_maturity_years': 1/12, 'option_type': 'call', 'price': 15.0},
    ]
    np.random.seed(0)
    num_scenarios_test = 1000
    # Ensure future_carbon_prices is a Series for testing calculate_cvar
    future_carbon_prices_test = pd.Series(np.random.normal(loc=210, scale=30, size=num_scenarios_test))
    future_carbon_prices_test[future_carbon_prices_test < 0] = 0 
    dummy_scenarios_df = pd.DataFrame(future_carbon_prices_test, columns=['scenario_1']) 
    # For make_operational_hedging_decision, it expects scenarios as columns, and takes the last row.
    # Let's reshape for the test to be similar to what the main script provides (index=time, columns=scenarios)
    dummy_scenarios_df_reshaped = pd.DataFrame(np.random.normal(loc=210, scale=30, size=(10, num_scenarios_test)), 
                                             columns=[f's{i}' for i in range(num_scenarios_test)])
    dummy_scenarios_df_reshaped[dummy_scenarios_df_reshaped < 0] = 0

    dummy_op_params = {
        'cvar_alpha_level': 0.95, 
        'co2_tons_to_hedge_from_des': 'total_chp_co2_ton', 
        'hedge_decision_threshold_cvar_reduction_pct': 0.02 
    }
    op_decision, op_details = make_operational_hedging_decision(dummy_des_summary, dummy_priced_options, dummy_scenarios_df_reshaped, dummy_op_params)
    print(f"Operational Decision: {op_decision}, Details: {op_details}")

    print("\n--- Testing Strategic Investment Decision Logic ---")
    dummy_roa_ccs_params = {'investment_cost_cny': 1000000}
    dummy_roa_results_1 = {'option_value': 1200000, 'traditional_npv': 100000} # Deferral adds sig value
    dummy_roa_results_2 = {'option_value': 1050000, 'traditional_npv': 1000000} # Deferral adds little value, NPV positive
    dummy_roa_results_3 = {'option_value': -50000, 'traditional_npv': -100000}# NPV negative, ROA also negative
    dummy_roa_results_4 = {'option_value': 100000, 'traditional_npv': -200000}# NPV negative, ROA makes it positive
    dummy_roa_results_5 = {'option_value': None, 'traditional_npv': 50000} # ROA failed

    dummy_strat_params = {
        'npv_hurdle_rate_pct': 0.10, # 10% of investment cost
        'min_option_premium_to_defer_pct': 0.05, # 5% of investment cost
        'investment_cost_key_in_roa_params': 'investment_cost_cny'
    }
    dummy_market_outlook = {'current_carbon_price': 200, 'roa_ccs_project_parameters': dummy_roa_ccs_params}

    strat_decision_1 = make_strategic_investment_decision(dummy_roa_results_1, dummy_strat_params, dummy_market_outlook)
    print(f"Strategic Decision (1): {strat_decision_1}")
    strat_decision_2 = make_strategic_investment_decision(dummy_roa_results_2, dummy_strat_params, dummy_market_outlook)
    print(f"Strategic Decision (2): {strat_decision_2}")
    strat_decision_3 = make_strategic_investment_decision(dummy_roa_results_3, dummy_strat_params, dummy_market_outlook)
    print(f"Strategic Decision (3): {strat_decision_3}")
    strat_decision_4 = make_strategic_investment_decision(dummy_roa_results_4, dummy_strat_params, dummy_market_outlook)
    print(f"Strategic Decision (4): {strat_decision_4}")
    strat_decision_5 = make_strategic_investment_decision(dummy_roa_results_5, dummy_strat_params, dummy_market_outlook)
    print(f"Strategic Decision (5): {strat_decision_5}") 