import os
import pandas as pd
import numpy as np
import json
import datetime
import logging

# Import from project modules
from src.utils.data_preparation import (
    load_electricity_demand, load_heat_demand, load_pv_generation_factor,
    get_chp_parameters, get_bess_parameters, get_market_parameters,
    get_financial_option_specs, get_roa_ccs_project_parameters, get_simulation_parameters
)
from src.des_optimizer.des_model import build_des_model, solve_des_model, extract_results
from src.carbon_pricer.carbon_price_models import (
    generate_synthetic_historical_prices, fit_garch_model,
    generate_price_scenarios_gbm, generate_price_scenarios_garch
)
from src.financial_option_pricer.option_pricer import price_european_option
from src.real_option_analyzer.roa_model import value_american_call_on_binomial_lattice, ccs_project_npv_at_node, ccs_investment_cost_func
from src.decision_controller.decision_logic import make_operational_hedging_decision, make_strategic_investment_decision
from src.results_analyzer.analysis import (
    plot_des_dispatch, plot_carbon_price_scenarios, 
    display_option_prices, display_roa_results, plot_roa_lattice
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_single_experiment(config, base_output_path, global_sim_params, chp_params, bess_params, market_params_func, financial_option_specs, roa_details_func):
    """Runs a single experiment configuration."""
    baseline_cp = config["baseline_carbon_price"]
    cp_volatility = config["carbon_price_volatility"]
    scenario_name = config["name"]

    current_experiment_path = os.path.join(base_output_path, scenario_name)
    os.makedirs(current_experiment_path, exist_ok=True)
    logging.info(f"Running experiment: {scenario_name} -> Output path: {current_experiment_path}")
    
    # Fetch roa_details using the function, potentially dependent on scenario config if needed
    # For now, assume roa_details_func() doesn't need specific config here, or pass baseline_cp if it does
    roa_specific_details = roa_details_func() 

    experiment_summary = {
        "scenario_parameters_applied": {
            "baseline_carbon_price_for_des": baseline_cp,
            "carbon_price_gbm_drift": global_sim_params['carbon_price_gbm_drift'],
            "carbon_price_gbm_volatility": cp_volatility, # This is scenario specific
            "num_carbon_price_scenarios": global_sim_params['num_carbon_price_scenarios'],
            "carbon_scenario_horizon_days": global_sim_params['carbon_scenario_horizon_days'],
            "roa_lattice_steps": global_sim_params['roa_lattice_steps']
        },
        "des_operational_summary": None,
        "carbon_price_scenario_summary": None,
        "financial_options_priced": [],
        "roa_ccs_results": None,
        "operational_hedging_decision": None,
        "strategic_ccs_investment_decision": None
    }

    # 1. Data Preparation for DES (using baseline_cp)
    logging.info("Preparing DES data...")
    num_hours_des = global_sim_params['des_optimization_horizon_hours']
    
    # Load full year data first
    full_elec_demand_df = load_electricity_demand() 
    full_heat_demand_df = load_heat_demand()
    full_pv_factor_series = load_pv_generation_factor()

    # Slice to the desired horizon for DES
    elec_demand_series = full_elec_demand_df['electricity_demand_kw'].iloc[:num_hours_des]
    heat_demand_series = full_heat_demand_df['heat_demand_kwth'].iloc[:num_hours_des]
    pv_factor_series = full_pv_factor_series.iloc[:num_hours_des]
    
    pv_capacity = global_sim_params['pv_capacity_kw'] 
    pv_generation_series = pv_factor_series * pv_capacity
    
    current_market_params_original = market_params_func(baseline_carbon_price=baseline_cp) # Get market params for current CP
    
    # Create TOU price series for the DES horizon
    time_index_des = pd.date_range(start='2023-01-01', periods=num_hours_des, freq='h') # Create a dummy index for hour matching
    tou_price_list = []
    for hour_of_day in time_index_des.hour: # Iterate through hours of the day for the DES horizon
        if (10 <= hour_of_day < 15) or (18 <= hour_of_day < 21): # Peak
            tou_price_list.append(current_market_params_original['tou_tariffs_cny_per_kwh']['peak'])
        elif (23 <= hour_of_day) or (hour_of_day < 7): # Valley
            tou_price_list.append(current_market_params_original['tou_tariffs_cny_per_kwh']['valley'])
        else: # Flat
            tou_price_list.append(current_market_params_original['tou_tariffs_cny_per_kwh']['flat'])

    current_market_params = current_market_params_original.copy() # Avoid modifying the original dict from getter
    current_market_params['tou_tariffs_cny_per_kwh_series'] = tou_price_list

    # Prepare inputs for the DES model
    # Note: DES model might expect more scenario-based inputs if it's stochastic.
    # Currently, run_case_study.py treats DES as deterministic for a given baseline carbon price.
    des_data_inputs = {
        "time_horizon": range(num_hours_des), # Pyomo model expects 'time_horizon' with integer indices
        "elec_demand_kw": elec_demand_series.tolist(), # Convert to list for integer indexing by Pyomo
        "heat_demand_kwth": heat_demand_series.tolist(), # Convert to list
        "pv_avail_kw": pv_generation_series.tolist(), # Convert to list
        "chp_params": chp_params,
        "bess_params": bess_params,
        "market_params": current_market_params, # This contains the single carbon price for this run
        # "grid_params" key used in model, let's pass what it might expect
        "grid_params": {'max_import_kw': global_sim_params['grid_max_import_export_kw'], 
                        'max_export_kw': global_sim_params['grid_max_import_export_kw']},
        # The following are expected by the current des_model.py but not directly used in this deterministic setup
        # Adding dummy values or structure to prevent KeyErrors if the model tries to access them.
        # This indicates a potential mismatch between run_case_study.py's deterministic DES run
        # and des_model.py's stochastic formulation.
        'scenarios': ['s1'], # Dummy scenario
        'option_types': [opt['label'] for opt in financial_option_specs], # Use correct variable name
        'scenario_probabilities': {'s1': 1.0}, # Dummy probability
        'carbon_prices_scenario_cny_ton': {('s1', t): current_market_params['carbon_price_cny_per_ton'] for t in range(num_hours_des)}, # Dummy carbon prices
        'option_strike_prices_cny_ton': {opt['label']: opt['strike_price'] for opt in financial_option_specs}, # Use correct variable name
        'option_premiums_cny_contract': {opt['label']: 0 for opt in financial_option_specs}, # Dummy premiums, use correct var name
        'option_payoffs_cny_contract': {(opt['label'], 's1'): 0 for opt in financial_option_specs} # Dummy payoffs, use correct var name
    }

    # 2. DES Optimization
    logging.info("Building and solving DES model...")
    des_summary_dict = {"total_chp_co2_ton": 0, "error": "DES not run or failed early"} # Default in case of failure
    try:
        des_model = build_des_model(des_data_inputs)
        solver_results, des_model_solved = solve_des_model(des_model, solver_name='cbc')
        if solver_results.solver.termination_condition == 'optimal' or solver_results.solver.termination_condition == 'feasible':
            des_results_df, des_summary_dict = extract_results(des_model_solved, des_data_inputs)
            experiment_summary["des_operational_summary"] = des_summary_dict
            
            # Save DES operational results to CSV
            des_csv_path = os.path.join(current_experiment_path, f"{scenario_name}_des_operational_results.csv")
            des_results_df.to_csv(des_csv_path)
            logging.info(f"DES operational results saved to {des_csv_path}")
            
            plot_des_dispatch(des_results_df, current_experiment_path, file_prefix=scenario_name)
        else:
            logging.error(f"DES Solver failed for {scenario_name}. Status: {solver_results.solver.termination_condition}")
            des_summary_dict = {"total_chp_co2_ton": 0, "error": f"Solver status: {solver_results.solver.termination_condition}"}
            experiment_summary["des_operational_summary"] = des_summary_dict
    except Exception as e:
        logging.error(f"Error in DES optimization for {scenario_name}: {e}")
        des_summary_dict = {"total_chp_co2_ton": 0, "error": str(e)}
        experiment_summary["des_operational_summary"] = des_summary_dict

    # 3. Carbon Price Scenario Generation
    logging.info("Generating carbon price scenarios...")
    carbon_model_to_use = global_sim_params.get("carbon_price_model_type", "GBM")
    cp_scenarios_df = None
    n_scenarios = global_sim_params['num_carbon_price_scenarios']
    horizon_days = global_sim_params['carbon_scenario_horizon_days']
    scenario_start_time = datetime.datetime.strptime(os.path.basename(base_output_path).split('master_run_')[1], '%Y%m%d_%H%M%S')
    scenario_start_time = scenario_start_time.replace(microsecond=np.random.randint(100000, 999999))

    if carbon_model_to_use == "GARCH":
        hist_prices = generate_synthetic_historical_prices(days=500, initial_price=baseline_cp*0.9, mu=0.01, sigma=cp_volatility*1.1) 
        garch_fit = fit_garch_model(hist_prices)
        if garch_fit:
            cp_scenarios_df = generate_price_scenarios_garch(garch_fit, baseline_cp, horizon_days, n_scenarios, scenario_start_time)
            experiment_summary["carbon_price_scenario_summary"] = {"num_scenarios": n_scenarios, "horizon_days": horizon_days, "model_used": "GARCH"}
        else:
            logging.warning("GARCH fitting failed for {scenario_name}, falling back to GBM.")
            experiment_summary["carbon_price_scenario_summary"] = {"num_scenarios": n_scenarios, "horizon_days": horizon_days, "model_used": "GBM_fallback_due_to_GARCH_fit_failure"}
    
    if cp_scenarios_df is None: # Fallback or primary GBM
        cp_scenarios_df = generate_price_scenarios_gbm(
            initial_price=baseline_cp,
            drift=global_sim_params['carbon_price_gbm_drift'],
            volatility=cp_volatility,
            horizon_days=horizon_days,
            num_scenarios=n_scenarios,
            start_datetime=scenario_start_time
        )
        if experiment_summary["carbon_price_scenario_summary"] is None: # Only set if not already set by GARCH attempt
             experiment_summary["carbon_price_scenario_summary"] = {"num_scenarios": n_scenarios, "horizon_days": horizon_days, "model_used": "GBM"}

    if cp_scenarios_df is not None:
        cp_scenarios_path = os.path.join(current_experiment_path, "carbon_price_scenarios.csv")
        cp_scenarios_df.to_csv(cp_scenarios_path)
        plot_carbon_price_scenarios(cp_scenarios_df, current_experiment_path, file_prefix=scenario_name)
    else:
        logging.error(f"Failed to generate carbon price scenarios for {scenario_name}.")

    # 4. Financial Option Pricing
    logging.info("Pricing financial options...")
    priced_options_list = []
    risk_free_rate = global_sim_params['risk_free_rate'] 
    for spec in financial_option_specs:
        price = price_european_option(spot_price=baseline_cp, 
                                      strike_price=spec['strike_price'], 
                                      time_to_maturity_years=spec['time_to_maturity_years'], 
                                      risk_free_rate=risk_free_rate, 
                                      volatility=cp_volatility, 
                                      option_type=spec['option_type'])
        priced_options_list.append({"strike_price": spec['strike_price'],"time_to_maturity_years": spec['time_to_maturity_years'],"option_type": spec['option_type'],"label": spec['label'],"price": price})

    logging.info(f"Priced {len(priced_options_list)} financial options.")
    experiment_summary["financial_options_priced"] = priced_options_list
    display_option_prices(priced_options_list, current_experiment_path, file_prefix=scenario_name + "_options")

    # 5. Real Option Analysis (ROA) for CCS
    logging.info("Performing ROA for CCS investment...")
    project_life = roa_specific_details['project_lifetime_years']
    annual_cash_flow_per_ton_co2 = roa_specific_details.get('annual_cash_flow_per_ton_co2', 50) # Example default if not in roa_params
    
    # Check if des_summary_dict exists and contains the key
    if des_summary_dict and 'total_chp_co2_ton' in des_summary_dict:
        annual_co2_abated_des = des_summary_dict['total_chp_co2_ton'] * (365*24 / num_hours_des) 
    else:
        annual_co2_abated_des = 0 # Fallback if DES results are not available
        logging.warning(f"DES summary or 'total_chp_co2_ton' not found for {scenario_name}. Using 0 for ROA CO2 abatement.")

    # Wrapper for ccs_project_npv_at_node to match the signature expected by value_american_call_on_binomial_lattice
    # The lattice builder will call this with: stochastic_var_at_node, project_params, time_step_dt, remaining_steps
    def current_ccs_project_npv_at_node_wrapper(carbon_price_at_node, project_params_from_lattice, time_step_dt_from_lattice, remaining_steps_from_lattice):
        # We use roa_specific_details captured from the outer scope as the primary project_params.
        # project_params_from_lattice should be the same roa_specific_details if passed correctly to value_american_call_on_binomial_lattice.
        
        # The imported ccs_project_npv_at_node from roa_model.py expects specific keys in its project_params arg.
        # roa_specific_details (which is project_params_from_lattice here) should have what ccs_project_npv_at_node needs
        # (e.g. 'chp_co2_emission_ton_per_m3_gas', 'chp_annual_generation_kwh_assumed', etc.) 
        # as provided by get_roa_ccs_project_parameters(). The annual_co2_abated_des from DES output is not directly used by ccs_project_npv_at_node.
        current_project_details = project_params_from_lattice # This is roa_specific_details

        return ccs_project_npv_at_node( # Calling the imported function from roa_model
            carbon_price_at_node=carbon_price_at_node,
            project_params=current_project_details, # Pass the comprehensive dict
            time_step_dt=time_step_dt_from_lattice, 
            remaining_steps_in_option_life=remaining_steps_from_lattice
        )

    def current_ccs_investment_cost_func(time_step, project_params_from_lattice):
        # project_params_from_lattice should be roa_specific_details.
        return ccs_investment_cost_func(time_step, project_params_from_lattice) # Calling imported function
    
    trad_npv_ccs = -np.inf # Default if ROA cannot be run
    roa_option_value = 0
    try:
        roa_option_value, underlying_lattice, option_lattice = value_american_call_on_binomial_lattice(
            S0=baseline_cp,
            K=None, # Strike is effectively handled by investment_cost_func and underlying_value_func
            T=roa_specific_details['max_deferral_years'],
            r=risk_free_rate,
            sigma=cp_volatility,
            N=global_sim_params['roa_lattice_steps'],
            underlying_value_func=current_ccs_project_npv_at_node_wrapper, # Use the wrapper
            investment_cost_func=current_ccs_investment_cost_func, # Pass the function for investment cost
            project_params=roa_specific_details # Pass the project parameters dictionary
        )
        # For traditional NPV, call the wrapper with current carbon price and parameters as if investing now (time_step_dt and remaining_steps might be less relevant here or set to typicals)
        # The original ccs_project_npv_at_node expects project_params, time_step_dt, remaining_steps.
        # Let's use dt from the lattice setup for consistency if needed by the NPV func for any internal discounting/period logic.
        lattice_dt = roa_specific_details['max_deferral_years'] / global_sim_params['roa_lattice_steps']
        project_npv_if_invest_now_trad = current_ccs_project_npv_at_node_wrapper(baseline_cp, roa_specific_details, lattice_dt, global_sim_params['roa_lattice_steps'])
        investment_cost_now_trad = current_ccs_investment_cost_func(0, roa_specific_details) # time_step=0 for now
        trad_npv_ccs = project_npv_if_invest_now_trad - investment_cost_now_trad
        experiment_summary["roa_ccs_results"] = {"option_value": roa_option_value, "traditional_npv": trad_npv_ccs}
        display_roa_results(trad_npv_ccs, roa_option_value, current_experiment_path, file_prefix=scenario_name + "_roa")
        if underlying_lattice is not None and option_lattice is not None and hasattr(plot_roa_lattice, '__call__'):
            plot_roa_lattice(underlying_lattice, option_lattice, current_experiment_path, file_prefix=scenario_name + "_roa_lattice") 
    except Exception as e:
        logging.error(f"Error in ROA calculation for {scenario_name}: {e}")
        experiment_summary["roa_ccs_results"] = {"option_value": 0, "traditional_npv": trad_npv_ccs, "error": str(e)}

    # 6. Decision Making
    logging.info("Making operational and strategic decisions...")

    # Prepare operational parameters
    # risk_aversion_factor from global_sim_params might map to cvar_alpha_level if 1-alpha, or be used differently.
    # For now, using common defaults as per decision_logic.py, can be refined.
    operational_params_for_decision = {
        'cvar_alpha_level': global_sim_params.get('cvar_alpha_level', 0.95), # e.g. 0.95 for 95% CVaR
        'co2_tons_to_hedge_from_des': 'total_chp_co2_emissions_ton', # Correct key from des_summary_dict
        'hedge_decision_threshold_cvar_reduction_pct': global_sim_params.get('hedge_decision_threshold_cvar_reduction_pct', 0.05)
    }
    # Add risk_aversion_factor if it's meant to be used directly by the function, though not explicitly in its current signature's logic
    operational_params_for_decision['risk_aversion_factor'] = global_sim_params.get('risk_aversion_factor')


    op_decision_details = make_operational_hedging_decision(
        des_summary=des_summary_dict,
        priced_options=priced_options_list,
        carbon_price_scenarios_df=cp_scenarios_df,
        operational_params=operational_params_for_decision
    )
    experiment_summary["operational_hedging_decision"] = op_decision_details

    # Prepare inputs for strategic decision
    roa_results_for_decision = {"option_value": roa_option_value, "traditional_npv": trad_npv_ccs}
    
    # strategic_params needs npv_hurdle_rate_pct. global_sim_params has 'npv_hurdle_rate' (absolute).
    # We need to either convert it or assume decision_logic can handle absolute if pct is not given.
    # For now, let's see if we can define a percentage based on investment cost.
    # The investment_cost is in roa_specific_details.
    investment_cost_for_hurdle_calc = roa_specific_details.get('investment_cost_cny', 0)
    npv_hurdle_rate_abs = global_sim_params.get('npv_hurdle_rate', 0)
    npv_hurdle_rate_pct_calc = (npv_hurdle_rate_abs / investment_cost_for_hurdle_calc) if investment_cost_for_hurdle_calc > 0 else 0.0

    strategic_params_for_decision = {
        'npv_hurdle_rate_pct': global_sim_params.get('npv_hurdle_rate_pct', npv_hurdle_rate_pct_calc), # Use calculated if not in global
        'min_option_premium_to_defer_pct': global_sim_params.get('min_option_premium_to_defer_pct', 0.05),
        'investment_cost_key_in_roa_params': 'investment_cost_cny' # key to find investment cost within roa_specific_details
    }
    market_outlook_for_decision = {
        'current_carbon_price': baseline_cp,
        'roa_ccs_project_parameters': roa_specific_details # This dict contains investment_cost_cny etc.
    }

    strat_decision_details = make_strategic_investment_decision(
        roa_results_ccs=roa_results_for_decision,
        strategic_params=strategic_params_for_decision,
        market_outlook=market_outlook_for_decision
    )
    experiment_summary["strategic_ccs_investment_decision"] = strat_decision_details

    # Save experiment summary JSON and TXT
    summary_json_path = os.path.join(current_experiment_path, "experiment_summary_data.json")
    with open(summary_json_path, 'w') as f_json:
        json.dump(experiment_summary, f_json, indent=4, cls=NpEncoder)
    logging.info(f"Saved experiment summary to {summary_json_path}")

    summary_txt_path = os.path.join(current_experiment_path, "experiment_summary.txt")
    with open(summary_txt_path, 'w') as f_txt:
        f_txt.write(json.dumps(experiment_summary, indent=4, cls=NpEncoder))
    logging.info(f"Saved text experiment summary to {summary_txt_path}")

    return experiment_summary, op_decision_details, strat_decision_details

class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types. """
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, pd.Timestamp): return obj.isoformat()
        elif isinstance(obj, datetime.datetime): return obj.isoformat()
        return super(NpEncoder, self).default(obj)

def main():
    logging.info("Starting Master Case Study Run...")
    experiment_configs = []
    # 只运行一个我们期望产生正E_NPV的情景
    base_cps = [350] # 提高基准碳价
    volatilities = [0.30] # 提高波动率

    for cp in base_cps:
        for vol in volatilities:
            experiment_configs.append({"name": f"cp{cp}_vol{int(vol*100)}", "baseline_carbon_price": float(cp), "carbon_price_volatility": float(vol)})

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_run_path = os.path.join("experiment_logs", f"master_run_{timestamp_str}")
    os.makedirs(master_run_path, exist_ok=True)
    logging.info(f"Master run output directory: {master_run_path}")

    general_sim_params = get_simulation_parameters()
    chp_parameters_static = get_chp_parameters()
    bess_parameters_static = get_bess_parameters()
    financial_option_specs_static = get_financial_option_specs()
    # Pass the function for ROA details so it can be called inside run_single_experiment if needed
    # This is if roa_details might change per scenario or needs lazy loading.
    # If they are truly static like CHP/BESS, load once: roa_details_static = get_roa_ccs_project_parameters()

    aggregated_results = []
    for config in experiment_configs:
        summary, op_decision, strat_decision = run_single_experiment(
            config,
            master_run_path,
            general_sim_params,
            chp_parameters_static,
            bess_parameters_static,
            get_market_parameters, # Pass func to be called with baseline_cp inside loop
            financial_option_specs_static,
            get_roa_ccs_project_parameters # Pass func to be called inside loop
        )
        aggregated_results.append({
            "scenario_name": config["name"],
            "baseline_carbon_price": config["baseline_carbon_price"],
            "gbm_volatility": config["carbon_price_volatility"],
            "operational_hedging_action": op_decision[0] if isinstance(op_decision, tuple) and len(op_decision) > 0 else "Error", # op_decision is a tuple (action, details)
            "operational_hedging_option_label": op_decision[1].get('option_label', '') if isinstance(op_decision, tuple) and len(op_decision) > 1 and isinstance(op_decision[1], dict) else '',
            "strategic_ccs_investment_action": strat_decision.get('action', 'Error') # strat_decision is a dict
        })
    
    agg_df = pd.DataFrame(aggregated_results)
    agg_csv_path = os.path.join(master_run_path, "aggregated_scenario_results.csv")
    agg_df.to_csv(agg_csv_path, index=False)
    logging.info(f"Saved aggregated scenario results to {agg_csv_path}")

    logging.info("Master Case Study Run Completed.")

if __name__ == "__main__":
    main()
