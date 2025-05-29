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

def make_operational_hedging_decision(des_summary, priced_instruments, carbon_price_scenarios_df, operational_params):
    """
    Makes a decision on operational hedging using financial carbon instruments (options, futures).

    Args:
        des_summary (dict): Summary of DES operation.
        priced_instruments (list): List of dictionaries, each representing a priced instrument.
                                 Option dict: {'instrument_type': 'option', 'name', 'price' (premium), 'strike_price', 'option_type'}.
                                 Futures dict: {'instrument_type': 'futures', 'name', 'price' (futures_price), 'premium' (0.0)}.
        carbon_price_scenarios_df (pd.DataFrame): DataFrame of carbon price scenarios.
        operational_params (dict): Parameters for decision logic.

    Returns:
        tuple: (str: decision_action, dict: decision_details)
    """
    print("Decision Controller: Making operational hedging decision...")
    if des_summary is None or not priced_instruments or carbon_price_scenarios_df is None or carbon_price_scenarios_df.empty:
        print("Decision Controller: Insufficient data for hedging decision (DES summary, instruments, or scenarios missing).")
        return ("NoHedge", {"reason": "Insufficient input data."})

    alpha_level = operational_params.get('cvar_alpha_level', 0.95)
    co2_tons_key = operational_params.get('co2_tons_to_hedge_from_des', 'total_chp_co2_ton')
    co2_tons_to_hedge = des_summary.get(co2_tons_key, 0)
    min_cvar_reduction_pct_threshold = operational_params.get('hedge_decision_threshold_cvar_reduction_pct', 0.05) 

    if co2_tons_to_hedge <= 0:
        print(f"Decision Controller: No CO2 emissions ({co2_tons_key}={co2_tons_to_hedge}) to hedge.")
        return ("NoHedge", {"reason": f"No CO2 emissions ({co2_tons_key}={co2_tons_to_hedge}) to hedge."})

    # Determine a relevant horizon for unhedged cost calculation.
    # This could be the maximum maturity of options being considered or a fixed operational horizon.
    # For now, let's use the full horizon of the provided carbon_price_scenarios_df for unhedged CVaR.
    # This assumes we are interested in the risk over the full scenario period if unhedged.
    horizon_days_for_unhedged_eval = carbon_price_scenarios_df.shape[0]
    
    # Use carbon prices at the *end of this evaluation horizon* for unhedged cost scenarios.
    # This is consistent with evaluating exposure at the end of the period.
    # If using average price for unhedged, this would need to change.
    unhedged_carbon_prices_at_eval_horizon = carbon_price_scenarios_df.iloc[horizon_days_for_unhedged_eval - 1].dropna()

    if unhedged_carbon_prices_at_eval_horizon.empty:
        print(f"Decision Controller: No valid future carbon prices at evaluation horizon (day {horizon_days_for_unhedged_eval}) for unhedged CVaR calculation.")
        return ("NoHedge", {"reason": "No valid future carbon prices for unhedged CVaR decision."})

    unhedged_total_carbon_costs_scenario = co2_tons_to_hedge * unhedged_carbon_prices_at_eval_horizon
    cvar_unhedged = calculate_cvar(unhedged_total_carbon_costs_scenario, alpha_level)
    print(f"Decision Controller: Unhedged CO2 Cost CVaR ({alpha_level*100:.0f}%) over {horizon_days_for_unhedged_eval} days: {cvar_unhedged:.2f} CNY for {co2_tons_to_hedge:.2f} tons")

    if np.isnan(cvar_unhedged):
        print("Decision Controller: Could not calculate unhedged CVaR.")
        return ("NoHedge", {"reason": "Could not calculate unhedged CVaR."})

    best_hedge_metric = cvar_unhedged # Can be CVaR for options or deterministic cost for futures
    best_instrument_details = None
    recommended_strategy_action = "NoHedge"

    for instrument in priced_instruments:
        instrument_type = instrument.get('instrument_type')
        instrument_name = instrument.get('name', 'Unknown Instrument')
        instrument_price = instrument.get('price') # This is premium for options, futures price for futures

        if instrument_price is None:
            print(f"Decision Controller: Skipping instrument {instrument_name} due to missing price/premium.")
            continue

        current_instrument_metric = float('inf')
        current_instrument_hedged_costs_scenario = None

        # Determine the relevant carbon prices for this specific instrument based on its maturity
        instrument_maturity_years = instrument.get('time_to_maturity_years')
        payoff_carbon_prices_for_instrument = None

        if instrument_maturity_years is not None:
            maturity_days = int(round(instrument_maturity_years * 365)) # Approx days
            # Ensure maturity_days is within the scenario horizon and at least 1 day for slicing
            maturity_days = min(max(1, maturity_days), carbon_price_scenarios_df.shape[0])
            
            # For payoff calculation (options, swaps potentially), use average price up to maturity for each scenario
            # For cost calculation of a locked-in price (futures), the specific day's price might be less relevant
            # than the locked-in price itself.
            # The current logic for futures/swaps uses the locked price directly, which is fine.
            # For options and collars, we need the spot price distribution at THEIR maturity.

            # We will use the spot price on the day of maturity for options/collars.
            # This is more standard than average price for European option payoff at maturity.
            payoff_carbon_prices_for_instrument = carbon_price_scenarios_df.iloc[maturity_days - 1].dropna()
        else:
            # Default to overall evaluation horizon if instrument has no specific maturity (e.g., some abstract instrument)
            # Or if maturity is not relevant (like for a future that locks in a price regardless of spot at intermediate times)
            payoff_carbon_prices_for_instrument = unhedged_carbon_prices_at_eval_horizon # Fallback, but should be handled by type

        if payoff_carbon_prices_for_instrument.empty and instrument_type in ['option', 'collar_strategy']:
            print(f"Decision Controller: No valid carbon prices at instrument's maturity (approx day {maturity_days if instrument_maturity_years else 'N/A'}) for {instrument_name}. Skipping.")
            continue
            
        if instrument_type == 'option':
            option_type = instrument.get('option_type', '').lower()
            if option_type != 'call': # Currently only evaluating call options for hedging emissions cost
                continue
            
            strike_price = instrument.get('strike_price')
            option_premium_per_ton = instrument.get('premium', instrument_price)

            if strike_price is None:
                print(f"Decision Controller: Skipping option {instrument_name} due to missing strike price.")
                continue
            
            # Cost with call option: Min(Spot_at_maturity, Strike) + Premium
            # Spot_at_maturity comes from payoff_carbon_prices_for_instrument
            hedged_spot_price_per_ton = np.minimum(payoff_carbon_prices_for_instrument, strike_price)
            total_cost_per_ton_scenario = hedged_spot_price_per_ton + option_premium_per_ton
            current_instrument_hedged_costs_scenario = co2_tons_to_hedge * total_cost_per_ton_scenario
            current_instrument_metric = calculate_cvar(current_instrument_hedged_costs_scenario, alpha_level)
            print(f"Decision Controller: Option \'{instrument_name}\' (Maturity: {instrument_maturity_years*12 if instrument_maturity_years else 'N/A'}M, Strike: {strike_price:.2f}, Premium: {option_premium_per_ton:.2f}) - Hedged CVaR: {current_instrument_metric:.2f} CNY")
        
        elif instrument_type == 'futures':
            futures_price_locked = instrument_price # This is F0
            # Cost with futures is deterministic: Tons * Futures_Price
            deterministic_cost_with_futures = co2_tons_to_hedge * futures_price_locked
            current_instrument_metric = deterministic_cost_with_futures
            # For consistent reporting, use the unhedged_carbon_prices_at_eval_horizon's index for the Series
            current_instrument_hedged_costs_scenario = pd.Series([deterministic_cost_with_futures] * len(unhedged_carbon_prices_at_eval_horizon), index=unhedged_carbon_prices_at_eval_horizon.index)
            print(f"Decision Controller: Futures \'{instrument_name}\' (Price: {futures_price_locked:.2f}) - Hedged Deterministic Cost: {current_instrument_metric:.2f} CNY")

        elif instrument_type == 'swap':
            fixed_swap_rate = instrument.get('fixed_rate', instrument_price)
            if fixed_swap_rate is None or np.isnan(fixed_swap_rate):
                print(f"Decision Controller: Skipping swap {instrument_name} due to missing or invalid fixed rate.")
                continue
            
            deterministic_cost_with_swap = co2_tons_to_hedge * fixed_swap_rate
            current_instrument_metric = deterministic_cost_with_swap
            current_instrument_hedged_costs_scenario = pd.Series([deterministic_cost_with_swap] * len(unhedged_carbon_prices_at_eval_horizon), index=unhedged_carbon_prices_at_eval_horizon.index)
            print(f"Decision Controller: Swap \'{instrument_name}\' (Fixed Rate: {fixed_swap_rate:.2f}) - Hedged Deterministic Cost: {current_instrument_metric:.2f} CNY")

        elif instrument_type == 'collar_strategy':
            put_strike = instrument.get('put_strike_price')
            call_strike = instrument.get('call_strike_price')
            collar_net_premium = instrument.get('premium')

            if put_strike is None or call_strike is None or collar_net_premium is None:
                print(f"Decision Controller: Skipping collar {instrument_name} due to missing strike prices or net premium.")
                continue
            
            # Cost with collar per ton = Net Premium + Effective Price bounded by strikes
            # Effective price is min(K_call, max(K_put, Spot_at_maturity_for_instrument))
            # Spot_at_maturity_for_instrument comes from payoff_carbon_prices_for_instrument
            effective_price_with_collar = np.minimum(call_strike, np.maximum(put_strike, payoff_carbon_prices_for_instrument))
            cost_per_ton_with_collar = collar_net_premium + effective_price_with_collar
            current_instrument_hedged_costs_scenario = co2_tons_to_hedge * cost_per_ton_with_collar
            current_instrument_metric = calculate_cvar(current_instrument_hedged_costs_scenario, alpha_level)
            print(f"Decision Controller: Collar Strategy \'{instrument_name}\' (Maturity: {instrument_maturity_years*12 if instrument_maturity_years else 'N/A'}M, P:{put_strike:.2f}, C:{call_strike:.2f}, NetPrem:{collar_net_premium:.2f}) - Hedged CVaR: {current_instrument_metric:.2f} CNY")

        else:
            print(f"Decision Controller: Unknown instrument type '{instrument_type}' for {instrument_name}. Skipping.")
            continue

        if not np.isnan(current_instrument_metric) and current_instrument_metric < best_hedge_metric:
            best_hedge_metric = current_instrument_metric
            recommended_strategy_action = f"HedgeWith{instrument_type.capitalize()}" # e.g., HedgeWithOption, HedgeWithFutures
            
            achieved_metric_for_details = best_hedge_metric
            cvar_reduction_val = cvar_unhedged - achieved_metric_for_details
            cvar_reduction_pct_val = (cvar_reduction_val / cvar_unhedged) if cvar_unhedged > 0 else 0

            best_instrument_details = {
                'instrument_name': instrument_name,
                'instrument_type': instrument_type,
                'quantity_hedged_tons': co2_tons_to_hedge,
                'achieved_cost_metric': achieved_metric_for_details, # CVaR for options, Cost for futures
                'unhedged_cvar': cvar_unhedged,
                'metric_improvement_cny': cvar_reduction_val,
                'metric_improvement_pct': cvar_reduction_pct_val
            }
            if instrument_type == 'option':
                best_instrument_details['strike_price'] = strike_price
                best_instrument_details['premium_per_ton'] = option_premium_per_ton
            elif instrument_type == 'futures':
                best_instrument_details['futures_price_locked'] = futures_price_locked
            elif instrument_type == 'swap':
                best_instrument_details['fixed_swap_rate'] = fixed_swap_rate
            elif instrument_type == 'collar_strategy':
                best_instrument_details['put_strike_price'] = instrument.get('put_strike_price')
                best_instrument_details['call_strike_price'] = instrument.get('call_strike_price')
                best_instrument_details['net_premium'] = instrument.get('premium')
    
    if recommended_strategy_action != "NoHedge" and best_instrument_details:
        if best_instrument_details['metric_improvement_pct'] >= min_cvar_reduction_pct_threshold:
            print(f"Decision Controller: Recommended Hedge: {best_instrument_details['instrument_name']} ({best_instrument_details['instrument_type']}) reduces cost metric by {best_instrument_details['metric_improvement_pct']:.2%}.")
            return recommended_strategy_action, best_instrument_details
        else:
            print(f"Decision Controller: Best instrument {best_instrument_details['instrument_name']} metric improvement ({best_instrument_details['metric_improvement_pct']:.2%}) is below threshold ({min_cvar_reduction_pct_threshold:.2%}). No hedge recommended.")
            return "NoHedge", {"reason": f"Best instrument {best_instrument_details['instrument_name']} metric improvement below threshold.", "details": best_instrument_details}
    
    print("Decision Controller: No suitable financial instrument found to improve cost metric significantly.")
    return "NoHedge", {"reason": "No suitable instrument found or metric improvement insufficient."}

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
    dummy_priced_instruments_list = [
        {'instrument_type': 'option', 'name': 'Call_200_1M', 'strike_price': 200, 'time_to_maturity_years': 1/12, 'option_type': 'call', 'price': 5.0, 'premium': 5.0},
        {'instrument_type': 'option', 'name': 'Call_220_1M', 'strike_price': 220, 'time_to_maturity_years': 1/12, 'option_type': 'call', 'price': 2.5, 'premium': 2.5},
        {'instrument_type': 'option', 'name': 'Call_180_1M_Exp', 'strike_price': 180, 'time_to_maturity_years': 1/12, 'option_type': 'call', 'price': 15.0, 'premium': 15.0},
        {'instrument_type': 'futures', 'name': 'Futures_1M', 'time_to_maturity_years': 1/12, 'price': 205.0, 'premium': 0.0}, # Futures price is 205
        {'instrument_type': 'futures', 'name': 'Futures_3M', 'time_to_maturity_years': 3/12, 'price': 210.0, 'premium': 0.0},  # Futures price is 210
        {'instrument_type': 'swap', 'name': 'Swap_1M', 'time_to_maturity_years': 1/12, 'price': 208.0, 'fixed_rate': 208.0, 'premium': 0.0}, # Example Swap
        {'instrument_type': 'swap', 'name': 'Swap_3M_High', 'time_to_maturity_years': 3/12, 'price': 215.0, 'fixed_rate': 215.0, 'premium': 0.0}, # Example Swap
        # Example Collar Strategy (assuming Put_200_1M and Call_220_1M are priced with premiums that make sense for this test)
        {'instrument_type': 'collar_strategy', 'name': 'Collar_P200_C220_1M', 
         'put_strike_price': 200, 'call_strike_price': 220, 'premium': -2.0, # Negative premium means income
         'put_to_buy_name': 'Put_200_1M_Test', 'call_to_sell_name': 'Call_220_1M_Test'} 
    ]
    np.random.seed(0)
    num_scenarios_test = 1000
    # Ensure future_carbon_prices is a Series for testing calculate_cvar
    future_carbon_prices_test = pd.Series(np.random.normal(loc=210, scale=30, size=num_scenarios_test))
    future_carbon_prices_test[future_carbon_prices_test < 0] = 0 
    # dummy_scenarios_df = pd.DataFrame(future_carbon_prices_test, columns=['scenario_1']) # Old test setup
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
    # op_decision, op_details = make_operational_hedging_decision(dummy_des_summary, dummy_priced_options, dummy_scenarios_df_reshaped, dummy_op_params)
    op_decision, op_details = make_operational_hedging_decision(dummy_des_summary, dummy_priced_instruments_list, dummy_scenarios_df_reshaped, dummy_op_params)
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