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
    get_financial_instrument_specs, get_roa_ccs_project_parameters, get_simulation_parameters,
    get_collar_strategy_specs
)
from src.des_optimizer.des_model import build_des_model, solve_des_model, extract_results
from src.carbon_pricer.carbon_price_models import (
    generate_synthetic_historical_prices, fit_garch_model,
    generate_price_scenarios_gbm, generate_price_scenarios_garch,
    generate_price_scenarios_jump_diffusion, generate_price_scenarios_regime_switching
)
from src.financial_option_pricer.option_pricer import price_european_option
from src.financial_option_pricer.financial_instruments import get_futures_price, calculate_fair_swap_rate
from src.real_option_analyzer.roa_model import value_american_call_on_binomial_lattice, ccs_project_npv_at_node, ccs_investment_cost_func
from src.decision_controller.decision_logic import make_operational_hedging_decision, make_strategic_investment_decision
from src.results_analyzer.analysis import (
    plot_des_dispatch, plot_carbon_price_scenarios, 
    display_option_prices, display_roa_results, plot_roa_lattice
)
from src.des_optimizer.des_model_checker import run_all_checks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_single_experiment(config, base_output_path, global_sim_params, chp_params, bess_params, market_params_func, financial_instrument_specs, roa_details_func):
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
        "financial_instruments_evaluated": [],
        "roa_ccs_results": None,
        "operational_hedging_decision": None,
        "strategic_ccs_investment_decision": None,
        "des_model_checker_issues": None
    }

    # 1. Data Preparation for general simulation (not DES specific hourly demands yet)
    num_hours_des = global_sim_params['des_optimization_horizon_hours']
    # DES operational period in days, rounded up
    num_days_des_optimization = (num_hours_des + 23) // 24

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

    # Moved Section: 3. Carbon Price Scenario Generation (Moved up)
    logging.info("Generating carbon price scenarios for stochastic DES inputs...")
    carbon_model_to_use = config.get("carbon_price_model_type", global_sim_params.get("carbon_price_model_type", "GBM"))

    cp_scenarios_df = None
    n_scenarios_for_des = global_sim_params['num_carbon_price_scenarios']
    horizon_days_for_cp_scenarios = global_sim_params['carbon_scenario_horizon_days']
    
    # Initialize experiment_summary for carbon price scenarios
    # We will update "model_used" specifically based on what actually happens.
    experiment_summary["carbon_price_scenario_summary"] = {
        "num_scenarios": n_scenarios_for_des, 
        "horizon_days": horizon_days_for_cp_scenarios,
        "model_configured": carbon_model_to_use, # Store what was configured
        "model_used": None # Will be set below
    }
    
    # Ensure cp_scenarios cover at least the DES optimization period for option payoff calculation
    if horizon_days_for_cp_scenarios < num_days_des_optimization:
        logging.warning(f"Carbon scenario horizon ({horizon_days_for_cp_scenarios} days) is shorter than DES optimization period ({num_days_des_optimization} days). Payoffs for some options might be based on incomplete data. Adjust 'carbon_scenario_horizon_days'.")
        # Consider extending horizon_days_for_cp_scenarios or handling this case carefully.
        # For now, we proceed, but this is a potential issue for options maturing beyond the scenario horizon.

    # scenario_start_time for cp_scenarios_df should be consistent
    # It's used for naming/logging, actual DES model uses relative time steps 0..T-1
    # base_output_path is like 'experiment_logs/master_run_YYYYMMDD_HHMMSS'
    try:
        # Robustly extract timestamp for scenario generation
        base_folder_name = os.path.basename(base_output_path)
        # Example: base_folder_name = "master_run_test_models_20250529_174040"
        # Example: base_folder_name = "master_run_cp300_vol74_20250523_141627"

        parts = base_folder_name.split('_')
        if len(parts) >= 3: # Check if there are enough parts to form a timestamp
            # Attempt to form the timestamp from the last two parts
            timestamp_candidate = parts[-2] + "_" + parts[-1]
            scenario_gen_start_time = datetime.datetime.strptime(timestamp_candidate, '%Y%m%d_%H%M%S')
            logging.info(f"Successfully parsed master_run timestamp for scenario generation: {scenario_gen_start_time} from {base_folder_name}")
        else:
            # If not enough parts, raise an error to be caught by the except block
            raise ValueError(f"Folder name format '{base_folder_name}' does not allow easy timestamp extraction from last two parts.")
            
    except Exception as e:
        # Log detailed error including the parts if available
        parts_info = locals().get('parts', 'not available')
        logging.warning(f"Could not parse master_run timestamp from {base_output_path} (parts: {parts_info}). Using current time. Error: {e}")
        scenario_gen_start_time = datetime.datetime.now()
    
    # Add a bit of randomness to microseconds for different experiment runs if they start at the exact same second
    scenario_gen_start_time = scenario_gen_start_time.replace(microsecond=np.random.randint(100000, 999999))

    if carbon_model_to_use == "GBM":
        logging.info(f"Using GBM model for carbon price scenarios as configured for {scenario_name}.")
        cp_scenarios_df = generate_price_scenarios_gbm(
            initial_price=baseline_cp,
            drift=config.get('gbm_drift', global_sim_params['carbon_price_gbm_drift']), 
            volatility=cp_volatility, 
            horizon_days=horizon_days_for_cp_scenarios,
            num_scenarios=n_scenarios_for_des,
            start_datetime=scenario_gen_start_time
        )
        experiment_summary["carbon_price_scenario_summary"]["model_used"] = "GBM"
    elif carbon_model_to_use == "GARCH":
        logging.info(f"Using GARCH model for carbon price scenarios for {scenario_name}.")
        hist_prices_garch = generate_synthetic_historical_prices(
            days=config.get('garch_hist_days', 500), 
            initial_price=baseline_cp * config.get('garch_hist_initial_price_factor', 0.9), 
            mu=config.get('garch_hist_mu', 0.01), 
            sigma=config.get('garch_hist_sigma', cp_volatility * 1.1)
        )
        garch_fit_model = fit_garch_model(hist_prices_garch)
        if garch_fit_model:
            cp_scenarios_df = generate_price_scenarios_garch(
                garch_fit_model, baseline_cp, horizon_days_for_cp_scenarios, 
                n_scenarios_for_des, scenario_gen_start_time
            )
            experiment_summary["carbon_price_scenario_summary"]["model_used"] = "GARCH"
            logging.info(f"GARCH model successfully used for {scenario_name}.")
        else:
            logging.warning(f"GARCH fitting failed for {scenario_name}, falling back to GBM.")
            experiment_summary["carbon_price_scenario_summary"]["model_used"] = "GBM_fallback_garch_fit_failed"
            cp_scenarios_df = generate_price_scenarios_gbm( # Fallback
                initial_price=baseline_cp,
                drift=config.get('gbm_drift', global_sim_params['carbon_price_gbm_drift']),
                volatility=cp_volatility,
                horizon_days=horizon_days_for_cp_scenarios,
                num_scenarios=n_scenarios_for_des,
                start_datetime=scenario_gen_start_time
            )
    elif carbon_model_to_use == "JumpDiffusion":
        logging.info(f"Using actual JumpDiffusion model for {scenario_name}.")
        try:
            # The placeholder function generate_price_scenarios_jump_diffusion now internally calls GBM
            cp_scenarios_df = generate_price_scenarios_jump_diffusion(
                initial_price=baseline_cp,
                drift=config.get('jd_drift', global_sim_params.get('jd_drift', config.get('gbm_drift', global_sim_params['carbon_price_gbm_drift']))), # Fallback to GBM drift for placeholder
                volatility=config.get('jd_volatility', global_sim_params.get('jd_volatility', cp_volatility)), # Fallback to scenario volatility for placeholder
                jump_intensity=config.get('jd_jump_intensity', global_sim_params.get('jd_jump_intensity', 0.1)), # These params are for the actual JD model, but passed to placeholder
                jump_mean=config.get('jd_jump_mean', global_sim_params.get('jd_jump_mean', 0.0)),
                jump_std=config.get('jd_jump_std', global_sim_params.get('jd_jump_std', 0.15)),
                horizon_days=horizon_days_for_cp_scenarios,
                num_scenarios=n_scenarios_for_des,
                start_datetime=scenario_gen_start_time
            )
            if cp_scenarios_df is not None:
                experiment_summary["carbon_price_scenario_summary"]["model_used"] = "JumpDiffusion"
                logging.info(f"Actual JumpDiffusion model successfully generated scenarios for {scenario_name}.")
            else:
                # This case implies the JumpDiffusion placeholder function itself returned None
                logging.error(f"Actual JumpDiffusion model generation failed for {scenario_name} (returned None). Falling back to direct GBM call.")
                experiment_summary["carbon_price_scenario_summary"]["model_used"] = "GBM_fallback_jumpdiffusion_failed"
                cp_scenarios_df = generate_price_scenarios_gbm(
                    initial_price=baseline_cp, drift=config.get('gbm_drift', global_sim_params['carbon_price_gbm_drift']),
                    volatility=cp_volatility, horizon_days=horizon_days_for_cp_scenarios,
                    num_scenarios=n_scenarios_for_des,start_datetime=scenario_gen_start_time)
        except Exception as e_jd:
            logging.error(f"Error during actual JumpDiffusion model execution for {scenario_name}: {e_jd}. Falling back to direct GBM call.")
            experiment_summary["carbon_price_scenario_summary"]["model_used"] = "GBM_fallback_jumpdiffusion_exception"
            cp_scenarios_df = generate_price_scenarios_gbm(
                initial_price=baseline_cp, drift=config.get('gbm_drift', global_sim_params['carbon_price_gbm_drift']),
                volatility=cp_volatility, horizon_days=horizon_days_for_cp_scenarios,
                num_scenarios=n_scenarios_for_des,start_datetime=scenario_gen_start_time)

    elif carbon_model_to_use == "RegimeSwitching":
        logging.info(f"Using actual RegimeSwitching model for {scenario_name}.")
        default_rs_params1 = {'drift': 0.01, 'volatility': 0.10} 
        default_rs_params2 = {'drift': 0.05, 'volatility': 0.30}
        default_rs_trans_matrix = [[0.95, 0.05], [0.03, 0.97]] # P00, P01; P10, P11

        params_regime1 = config.get('rs_params_regime1', global_sim_params.get('rs_params_regime1', default_rs_params1))
        params_regime2 = config.get('rs_params_regime2', global_sim_params.get('rs_params_regime2', default_rs_params2))
        transition_matrix_list = config.get('rs_transition_matrix', global_sim_params.get('rs_transition_matrix', default_rs_trans_matrix))
        
        # Ensure transition_matrix is a numpy array
        try:
            transition_matrix = np.array(transition_matrix_list)
            if transition_matrix.shape != (2,2):
                logging.error(f"Transition matrix for {scenario_name} has incorrect shape {transition_matrix.shape}. Expected (2,2). Using default.")
                transition_matrix = np.array(default_rs_trans_matrix)
        except Exception as e_tm_conversion:
            logging.error(f"Error converting transition matrix for {scenario_name}: {e_tm_conversion}. Using default.")
            transition_matrix = np.array(default_rs_trans_matrix)

        try:
            cp_scenarios_df = generate_price_scenarios_regime_switching(
                initial_price=baseline_cp, 
                params_regime1=params_regime1, 
                params_regime2=params_regime2,
                transition_matrix=transition_matrix, 
                horizon_days=horizon_days_for_cp_scenarios,
                num_scenarios=n_scenarios_for_des, 
                start_datetime=scenario_gen_start_time
            )
            if cp_scenarios_df is not None:
                experiment_summary["carbon_price_scenario_summary"]["model_used"] = "RegimeSwitching"
                logging.info(f"Actual RegimeSwitching model successfully generated scenarios for {scenario_name}.")
            else:
                logging.error(f"Actual RegimeSwitching model generation failed for {scenario_name} (returned None). Falling back to direct GBM call.")
                experiment_summary["carbon_price_scenario_summary"]["model_used"] = "GBM_fallback_regimeswitching_failed"
                cp_scenarios_df = generate_price_scenarios_gbm(
                    initial_price=baseline_cp, drift=config.get('gbm_drift', global_sim_params['carbon_price_gbm_drift']),
                    volatility=cp_volatility, horizon_days=horizon_days_for_cp_scenarios,
                    num_scenarios=n_scenarios_for_des,start_datetime=scenario_gen_start_time)
        except Exception as e_rs:
            logging.error(f"Error during actual RegimeSwitching model execution for {scenario_name}: {e_rs}. Falling back to direct GBM call.")
            experiment_summary["carbon_price_scenario_summary"]["model_used"] = "GBM_fallback_regimeswitching_exception"
            cp_scenarios_df = generate_price_scenarios_gbm(
                initial_price=baseline_cp, drift=config.get('gbm_drift', global_sim_params['carbon_price_gbm_drift']),
                volatility=cp_volatility, horizon_days=horizon_days_for_cp_scenarios,
                num_scenarios=n_scenarios_for_des,start_datetime=scenario_gen_start_time)
    else:
        logging.error(f"Unknown carbon_price_model_type: {carbon_model_to_use} in config for {scenario_name}. Defaulting to GBM.")
        experiment_summary["carbon_price_scenario_summary"]["model_used"] = "GBM_fallback_unknown_model"
        cp_scenarios_df = generate_price_scenarios_gbm( # Fallback
            initial_price=baseline_cp,
            drift=config.get('gbm_drift', global_sim_params['carbon_price_gbm_drift']),
            volatility=cp_volatility,
            horizon_days=horizon_days_for_cp_scenarios,
            num_scenarios=n_scenarios_for_des,
            start_datetime=scenario_gen_start_time
        )
    
    # This general fallback for cp_scenarios_df being None is a final safety net.
    # Individual model sections should ideally handle their fallbacks and set model_used.
    if cp_scenarios_df is None:
        logging.critical(f"Carbon price scenario generation FAILED for model type {carbon_model_to_use} and no fallback succeeded for {scenario_name}. This is a critical error.")
        # Ensure model_used reflects the ultimate failure if not already set by a specific fallback.
        if experiment_summary["carbon_price_scenario_summary"]["model_used"] is None:
            experiment_summary["carbon_price_scenario_summary"]["model_used"] = f"Generation_Failed_For_{carbon_model_to_use}"
        
        experiment_summary["des_operational_summary"] = {"error": f"Carbon price scenario generation ultimately failed for model {carbon_model_to_use}."}
        summary_json_path = os.path.join(current_experiment_path, "experiment_summary_data.json")
        with open(summary_json_path, 'w') as f_json: json.dump(experiment_summary, f_json, indent=4, cls=NpEncoder)
        return experiment_summary, {}, {}

    # DEBUG: Print cp_scenarios_df info for specific scenario
    if scenario_name == "test_garch_model":
        logging.info(f"DEBUG: cp_scenarios_df for {scenario_name} (GARCH):")
        logging.info(f"Shape: {cp_scenarios_df.shape}")
        # Create a string representation for logging to avoid direct df object if too large
        try:
            desc_str = cp_scenarios_df.describe().to_string()
            head_str = cp_scenarios_df.head().to_string()
            logging.info(f"Describe:\n{desc_str}")
            logging.info(f"Head:\n{head_str}")
        except Exception as e_debug_print:
            logging.error(f"Error printing cp_scenarios_df debug info: {e_debug_print}")

    # Moved Section: 4. Financial Option Pricing (Moved up)
    logging.info("Pricing financial options for stochastic DES inputs...")
    priced_instruments_list = []
    risk_free_rate = global_sim_params['risk_free_rate'] 
    for spec in financial_instrument_specs:
        instrument_type = spec.get('instrument_type', 'option')
        
        if instrument_type == 'option':
            price = price_european_option(spot_price=baseline_cp, 
                                          strike_price=spec['strike_price'], 
                                          time_to_maturity_years=spec['time_to_maturity_years'], 
                                          risk_free_rate=risk_free_rate, 
                                          volatility=cp_volatility, 
                                          option_type=spec['option_type'])
            priced_instruments_list.append({**spec, "price": price, "premium": price})
        elif instrument_type == 'futures':
            price = get_futures_price(spot_price=baseline_cp,
                                        risk_free_rate=risk_free_rate,
                                        time_to_maturity_years=spec['time_to_maturity_years'])
            priced_instruments_list.append({**spec, "price": price, "premium": 0.0})
        elif instrument_type == 'swap':
            time_to_maturity_years = spec['time_to_maturity_years']
            # Maturity day index for cp_scenarios_df (0-indexed)
            # approx_days_to_maturity = int(round(time_to_maturity_years * 365))
            # Ensure we use the same horizon as the carbon price scenarios for consistency
            # The swap rate is determined by the expected price AT that maturity day.
            maturity_day_index = min(int(round(time_to_maturity_years * 365)), horizon_days_for_cp_scenarios) -1
            maturity_day_index = max(0, maturity_day_index) # Ensure not negative if ttm is very short

            if maturity_day_index < cp_scenarios_df.shape[0]:
                carbon_prices_at_maturity = cp_scenarios_df.iloc[maturity_day_index]
                fair_swap_rate = calculate_fair_swap_rate(carbon_prices_at_maturity)
                priced_instruments_list.append({**spec, "price": fair_swap_rate, "premium": 0.0, "fixed_rate": fair_swap_rate})
                logging.info(f"Calculated fair swap rate for {spec['name']} (maturity {time_to_maturity_years*12:.0f}M, day index {maturity_day_index+1}): {fair_swap_rate:.2f}")
            else:
                logging.warning(f"Maturity day index {maturity_day_index+1} for swap {spec['name']} is out of bounds for cp_scenarios_df (horizon {cp_scenarios_df.shape[0]} days). Skipping swap.")
                priced_instruments_list.append({**spec, "price": np.nan, "premium": 0.0, "error": "Maturity out of bounds"})

        else:
            logging.warning(f"Unknown instrument type: {instrument_type} for spec {spec.get('name')}. Skipping.")
            continue

    # Process Collar Strategies
    collar_strategy_specs = get_collar_strategy_specs() # Get collar definitions
    if collar_strategy_specs:
        logging.info(f"Processing {len(collar_strategy_specs)} collar strategy specifications...")
        # Create a quick lookup for already priced instruments by name
        priced_instrument_lookup = {inst['name']: inst for inst in priced_instruments_list}
        
        for collar_spec in collar_strategy_specs:
            put_name = collar_spec.get('put_to_buy_name')
            call_name = collar_spec.get('call_to_sell_name')
            collar_name = collar_spec.get('name', f"Collar_{put_name}__{call_name}")

            put_option = priced_instrument_lookup.get(put_name)
            call_option = priced_instrument_lookup.get(call_name)

            if put_option and call_option and put_option.get('instrument_type') == 'option' and call_option.get('instrument_type') == 'option':
                if put_option.get('option_type', '').lower() != 'put' or call_option.get('option_type', '').lower() != 'call':
                    logging.warning(f"Collar {collar_name}: Component {put_name} is not a Put or {call_name} is not a Call. Skipping.")
                    continue
                
                put_premium = put_option.get('premium', np.nan)
                call_premium = call_option.get('premium', np.nan)

                if not np.isnan(put_premium) and not np.isnan(call_premium):
                    net_collar_premium = put_premium - call_premium # Cost to buy put, revenue from selling call
                    
                    # Ensure critical details are present for the decision logic
                    if 'strike_price' not in put_option or 'strike_price' not in call_option:
                        logging.warning(f"Collar {collar_name}: Strike price missing for {put_name} or {call_name}. Skipping collar.")
                        continue

                    collar_details = {
                        **collar_spec, # Includes name and component names from definition
                        "instrument_type": "collar_strategy", # Override or ensure it's set
                        "price": net_collar_premium, # Net cost/premium of the collar strategy
                        "premium": net_collar_premium,
                        "put_strike_price": put_option['strike_price'],
                        "call_strike_price": call_option['strike_price'],
                        "put_premium_component": put_premium,
                        "call_premium_component": call_premium,
                        # Include time_to_maturity if consistent for both legs, or handle appropriately
                        # For simplicity, assume they share a maturity for now if defined that way
                        "time_to_maturity_years": put_option.get('time_to_maturity_years') 
                    }
                    priced_instruments_list.append(collar_details)
                    logging.info(f"Processed Collar Strategy '{collar_name}': Net Premium {net_collar_premium:.2f}")
                else:
                    logging.warning(f"Collar {collar_name}: Premium missing for component {put_name} or {call_name}. Skipping collar.")
            else:
                logging.warning(f"Collar {collar_name}: Component option {put_name} or {call_name} not found or not an option in priced_instruments. Skipping collar.")

    logging.info(f"Total evaluated financial instruments (including strategies): {len(priced_instruments_list)}")
    experiment_summary["financial_instruments_evaluated"] = priced_instruments_list

    # 1. Data Preparation for DES (Now after CP scenarios and option pricing)
    logging.info("Preparing DES data with stochastic inputs...")
    # Stochastic inputs for DES model
    des_model_scenarios = list(cp_scenarios_df.columns)
    num_model_scenarios = len(des_model_scenarios)
    des_scenario_probabilities = {s: 1.0/num_model_scenarios for s in des_model_scenarios}

    # Initialize with existing deterministic parts that are still needed
    des_data_inputs = {
        "time_horizon": range(num_hours_des),
        "elec_demand_kw": elec_demand_series.tolist(),
        "heat_demand_kwth": heat_demand_series.tolist(),
        "pv_avail_kw": pv_generation_series.tolist(),
        "chp_params": chp_params,
        "bess_params": bess_params,
        "market_params": current_market_params, # Contains TOU, gas price, but NOT carbon price for DES model
        "grid_params": {'max_import_kw': global_sim_params['grid_max_import_export_kw'], 
                        'max_export_kw': global_sim_params['grid_max_import_export_kw']},
        'option_list': [inst['name'] for inst in priced_instruments_list if inst['instrument_type'] == 'option'],
        'option_strike_prices_cny_ton': {inst['name']: inst['strike_price'] for inst in priced_instruments_list if inst['instrument_type'] == 'option'},
        
        # Stochastic parts to be filled
        'scenarios': des_model_scenarios,
        'scenario_probabilities': des_scenario_probabilities,
        'carbon_prices_scenario_cny_ton': {}, # To be filled
        'option_premiums_cny_contract': {},     # To be filled
        'option_payoffs_cny_contract': {},     # To be filled

        # CVaR Parameters for DES Model Objective
        'cvar_alpha_level': global_sim_params.get('cvar_alpha_level', 0.95), # Get from global_sim_params or default
        'lambda_cvar_weight': global_sim_params.get('lambda_cvar_weight', 0.0), # Get from global_sim_params or default to 0 (no CVaR)
        'max_option_contracts_limit_abs': global_sim_params.get('max_option_contracts_limit_abs', 250) # Added for option contract limits
    }
    
    # Fill carbon_prices_scenario_cny_ton
    # cp_scenarios_df index is DatetimeIndex, DES model uses integer time steps for hours.
    # We assume the daily carbon price applies to all hours of that day.
    for s_label in des_model_scenarios:
        for t_hour_idx in range(num_hours_des):
            day_idx = t_hour_idx // 24
            # Ensure day_idx is within the bounds of the generated scenario horizon
            if day_idx < len(cp_scenarios_df.index):
                des_data_inputs['carbon_prices_scenario_cny_ton'][(s_label, t_hour_idx)] = cp_scenarios_df[s_label].iloc[day_idx]
            else:
                # This case should ideally not happen if horizon_days_for_cp_scenarios is adequate.
                # Fallback: use the last available day's price or handle error.
                # Using last known price for robustness, though it might not be accurate for long DES horizons beyond CP scenario.
                logging.warning(f"Hour index {t_hour_idx} (day {day_idx}) exceeds cp_scenario horizon ({len(cp_scenarios_df.index)} days) for scenario {s_label}. Using last available carbon price.")
                des_data_inputs['carbon_prices_scenario_cny_ton'][(s_label, t_hour_idx)] = cp_scenarios_df[s_label].iloc[-1]

    # Fill option_premiums_cny_contract
    for inst_data in priced_instruments_list:
        if inst_data['instrument_type'] == 'option':
            des_data_inputs['option_premiums_cny_contract'][inst_data['name']] = inst_data.get('premium', 0.0)

    # Fill option_payoffs_cny_contract
    # Payoff is calculated based on the average carbon price over the option's life,
    # but relevant to the DES operational period.
    for inst_data in priced_instruments_list:
        if inst_data['instrument_type'] != 'option':
            continue
        
        opt_label = inst_data['name']
        strike_price = inst_data['strike_price']
        option_type = inst_data['option_type']
        # Maturity in days from years, relative to start of carbon price scenarios
        maturity_in_days_config = int(inst_data['time_to_maturity_years'] * 365)

        for s_label in des_model_scenarios:
            # Determine the number of days relevant for payoff calculation within this DES run.
            # THIS IS THE KEY CHANGE: Use the option's full maturity for payoff calculation
            # days_for_payoff_calc = min(maturity_in_days_config, num_days_des_optimization)
            days_for_payoff_calc = maturity_in_days_config
            
            # Ensure we don't try to read beyond the generated carbon price scenario horizon
            if days_for_payoff_calc > horizon_days_for_cp_scenarios:
                logging.warning(f"Option {opt_label} maturity ({days_for_payoff_calc} days) exceeds carbon scenario horizon ({horizon_days_for_cp_scenarios} days). Payoff will be based on available {horizon_days_for_cp_scenarios} days.")
                days_for_payoff_calc = horizon_days_for_cp_scenarios
            
            if days_for_payoff_calc <= 0:
                avg_carbon_price_for_payoff = baseline_cp # Or handle as 0 payoff if maturity is 0. For safety, use baseline_cp to avoid NaN if list is empty.
                payoff = 0 # An option with 0 days to maturity or for payoff calc has no time value / exposure.
            else:
                # Extract daily carbon prices for this scenario for the relevant payoff period.
                # cp_scenarios_df columns are s_labels, index is DatetimeIndex. iloc uses integer positions.
                # We need prices from day 0 up to days_for_payoff_calc-1
                relevant_daily_prices = cp_scenarios_df[s_label].iloc[0:days_for_payoff_calc]
                
                if relevant_daily_prices.empty:
                    # This might happen if days_for_payoff_calc is 0, or an issue with slicing.
                    # If empty due to days_for_payoff_calc = 0, payoff should be 0.
                    # If due to other issues, this is an unexpected state. Default to baseline_cp for safety.
                    logging.warning(f"No relevant daily prices found for payoff calc: opt {opt_label}, scenario {s_label}, days_for_payoff_calc {days_for_payoff_calc}. Defaulting avg price to baseline.")
                    avg_carbon_price_for_payoff = baseline_cp 
                else:
                    avg_carbon_price_for_payoff = relevant_daily_prices.mean()

            payoff = 0.0
            debug_strike_info = "N/A"

            if inst_data['instrument_type'] == 'collar_strategy':
                K_put=inst_data['put_strike_price']
                K_call=inst_data['call_strike_price']
                payoff = calculate_collar_payoff(
                    S=avg_carbon_price_for_payoff, 
                    K_put=K_put, 
                    K_call=K_call
                )
                debug_strike_info = f"PutK:{K_put}, CallK:{K_call}"
            elif option_type.lower() == 'call': 
                current_strike = strike_price
                payoff = max(0.0, avg_carbon_price_for_payoff - current_strike)
                debug_strike_info = f"Strike:{current_strike}"
            elif option_type.lower() == 'put':
                current_strike = strike_price
                payoff = max(0.0, current_strike - avg_carbon_price_for_payoff)
                debug_strike_info = f"Strike:{current_strike}"
            
            logging.info(f"DEBUG_PAYOFF_CALC: opt:{opt_label}, scen:{s_label}, type:{inst_data['instrument_type']}/{option_type}, avg_cp:{avg_carbon_price_for_payoff:.2f}, {debug_strike_info}, CALC_PAYOFF:{payoff:.4f}")
            
            # All other instrument_overall_type (futures, swap) will have payoff = 0.0, which is correct as they are not options.
            des_data_inputs['option_payoffs_cny_contract'][(opt_label, s_label)] = payoff
    
    # ---- START DEBUGGING BLOCK for test_gbm_model ----
    if scenario_name == "test_gbm_model":
        options_to_debug = ["Put_300_1M", "Call_300_1M", "Call_280_1M", "Collar_1M_P300"]
        logging.info(f"DEBUG_DES_INPUTS for {scenario_name}:")
        for opt_name_debug in options_to_debug:
            if opt_name_debug in des_data_inputs['option_premiums_cny_contract']:
                logging.info(f"  {opt_name_debug} - Premium: {des_data_inputs['option_premiums_cny_contract'][opt_name_debug]}")
            else:
                logging.info(f"  {opt_name_debug} - Premium: NOT FOUND")

            logging.info(f"  {opt_name_debug} - Payoffs (first 3 scenarios):")
            for s_idx, s_label_debug in enumerate(des_model_scenarios[:3]): # Log for first 3 scenarios
                payoff_val = des_data_inputs['option_payoffs_cny_contract'].get((opt_name_debug, s_label_debug), "NOT FOUND")
                
                # To provide context, find the average carbon price used for this payoff.
                # This requires knowing the maturity_in_days_config for this specific option.
                avg_cp_for_payoff_debug = "N/A"
                maturity_in_days_debug = "N/A"
                for inst_data_debug in priced_instruments_list:
                    if inst_data_debug['name'] == opt_name_debug:
                        maturity_in_days_debug = int(inst_data_debug['time_to_maturity_years'] * 365)
                        days_for_payoff_calc_debug = min(maturity_in_days_debug, horizon_days_for_cp_scenarios)
                        if days_for_payoff_calc_debug > 0:
                            relevant_daily_prices_debug = cp_scenarios_df[s_label_debug].iloc[0:days_for_payoff_calc_debug]
                            if not relevant_daily_prices_debug.empty:
                                avg_cp_for_payoff_debug = f"{relevant_daily_prices_debug.mean():.2f}"
                        break
                logging.info(f"    {s_label_debug}: Payoff = {payoff_val}, Avg CP for Payoff ({days_for_payoff_calc_debug}d) = {avg_cp_for_payoff_debug}")
    # ---- END DEBUGGING BLOCK ----

    # DEBUG: Print calculated payoffs for a specific option to verify
    if scenario_name == "test_garch_model": # Or any scenario you are testing
        target_option_to_debug = 'Call_280_1M'
        logging.info(f"DEBUG: Calculated payoffs for {target_option_to_debug} in scenario {scenario_name}:")
        payoffs_for_target_option = {}
        total_expected_payoff_debug = 0
        num_positive_payoff_scenarios = 0
        for s_idx, s_label_debug in enumerate(des_model_scenarios):
            payoff_val = des_data_inputs['option_payoffs_cny_contract'].get((target_option_to_debug, s_label_debug), None)
            payoffs_for_target_option[s_label_debug] = payoff_val
            if payoff_val is not None and payoff_val > 0:
                num_positive_payoff_scenarios += 1
            if payoff_val is not None:
                 total_expected_payoff_debug += des_scenario_probabilities[s_label_debug] * payoff_val
            # Print first 5, and any positive payoffs, or if payoff is unexpectedly zero for Call_280_1M
            if s_idx < 5 or payoff_val > 0 or (target_option_to_debug == 'Call_280_1M' and payoff_val == 0.0) :
                logging.info(f"  Scenario {s_label_debug}: Payoff = {payoff_val}, AvgPrice = {avg_carbon_price_for_payoff:.2f}, Strike = {strike_price}, Days = {days_for_payoff_calc}")
        logging.info(f"DEBUG: Summary for {target_option_to_debug}: Total expected payoff (calculated here) = {total_expected_payoff_debug}, Num scenarios with positive payoff = {num_positive_payoff_scenarios}/{len(des_model_scenarios)}")

    # ---- START MODIFICATION FOR test_gbm_model ----
    if scenario_name == "test_gbm_model":
        original_lambda = des_data_inputs.get('lambda_cvar_weight', 'Not Set')
        des_data_inputs['lambda_cvar_weight'] = 100.0
        logging.info(f"Applied MODIFICATION for {scenario_name}: lambda_cvar_weight changed from {original_lambda} to {des_data_inputs['lambda_cvar_weight']}")
        
        if 'Call_280_1M' in des_data_inputs.get('option_premiums_cny_contract', {}):
            original_premium = des_data_inputs['option_premiums_cny_contract']['Call_280_1M']
            des_data_inputs['option_premiums_cny_contract']['Call_280_1M'] /= 2
            logging.info(f"Applied MODIFICATION for {scenario_name}: 'Call_280_1M' premium changed from {original_premium} to {des_data_inputs['option_premiums_cny_contract']['Call_280_1M']}")
        else:
            logging.warning(f"MODIFICATION for {scenario_name}: 'Call_280_1M' not found in option_premiums_cny_contract. Premium not changed.")
    # ---- END MODIFICATION FOR test_gbm_model ----

    # 2. DES Optimization (Now uses stochastic inputs)
    logging.info("Building and solving STOCHASTIC DES model...")
    des_summary_dict = {"total_chp_co2_ton": 0, "error": "DES not run or failed early"} # Default in case of failure
    try:
        des_model = build_des_model(des_data_inputs)
        solver_results, des_model_solved = solve_des_model(des_model, solver_name='cbc')
        if solver_results.solver.termination_condition == 'optimal' or solver_results.solver.termination_condition == 'feasible':
            des_results_df, des_summary_dict = extract_results(des_model_solved, des_data_inputs)
            # Ensure ROA gets the CO2 tonnage under the key it expects
            if 'expected_total_chp_co2_emissions_ton' in des_summary_dict:
                des_summary_dict['total_chp_co2_ton'] = des_summary_dict['expected_total_chp_co2_emissions_ton']
            experiment_summary["des_operational_summary"] = des_summary_dict
            
            # Save DES operational results to CSV
            des_csv_path = os.path.join(current_experiment_path, f"{scenario_name}_des_operational_results.csv")
            des_results_df.to_csv(des_csv_path)
            logging.info(f"DES operational results saved to {des_csv_path}")
            
            plot_des_dispatch(des_results_df, current_experiment_path, file_prefix=scenario_name)

            # 在这里调用模型检查器
            logging.info(f"开始对场景 {scenario_name} 的DES模型结果进行合理性检查...")
            checker_issues = run_all_checks(
                dispatch_df=des_results_df, 
                summary_dict=des_summary_dict, 
                model_inputs=des_data_inputs 
            )
            if checker_issues:
                logging.warning(f"模型合理性检查发现问题 ({len(checker_issues)}条) - 场景 {scenario_name}:")
                # 将问题详情也记录到experiment_summary中，方便后续查看
                experiment_summary["des_model_checker_issues"] = []
                for issue_idx, issue_detail in enumerate(checker_issues):
                    log_message = f"  问题 {issue_idx+1}: {issue_detail}"
                    logging.warning(log_message)
                    experiment_summary["des_model_checker_issues"].append(log_message)
            else:
                logging.info(f"模型合理性检查通过 - 场景 {scenario_name}.")
                experiment_summary["des_model_checker_issues"] = ["所有检查均已通过。"]

        else:
            logging.error(f"DES Solver failed for {scenario_name}. Status: {solver_results.solver.termination_condition}")
            des_summary_dict = {"total_chp_co2_ton": 0, "error": f"Solver status: {solver_results.solver.termination_condition}"}
            experiment_summary["des_operational_summary"] = des_summary_dict
            experiment_summary["des_model_checker_issues"] = [f"DES求解失败，未进行模型检查。求解器状态: {solver_results.solver.termination_condition}"]
    except Exception as e:
        logging.error(f"Error in DES optimization for {scenario_name}: {e}")
        des_summary_dict = {"total_chp_co2_ton": 0, "error": str(e)}
        experiment_summary["des_operational_summary"] = des_summary_dict
        experiment_summary["des_model_checker_issues"] = [f"DES优化过程中发生错误，未进行模型检查: {str(e)}"]

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
        priced_instruments=priced_instruments_list,
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
    logging.info("Starting Master Case Study Run for Testing Carbon Price Models...")
    
    experiment_configs = [
        {
            "name": "test_gbm_model",
            "baseline_carbon_price": 300.0,
            "carbon_price_volatility": 0.80,
            "carbon_price_model_type": "GBM"
        },
        {
            "name": "test_garch_model",
            "baseline_carbon_price": 310.0,
            "carbon_price_volatility": 0.25,
            "carbon_price_model_type": "GARCH",
            "garch_hist_days": 400 
        },
        {
            "name": "test_jumpdiffusion_model",
            "baseline_carbon_price": 320.0,
            "carbon_price_volatility": 0.30, 
            "carbon_price_model_type": "JumpDiffusion",
            "jd_drift": 0.03,
            "jd_volatility": 0.25, 
            "jd_jump_intensity": 0.2,
            "jd_jump_mean": 0.05,
            "jd_jump_std": 0.15
        },
        {
            "name": "test_regimeswitching_model",
            "baseline_carbon_price": 330.0,
            "carbon_price_volatility": 0.35, 
            "carbon_price_model_type": "RegimeSwitching",
            "rs_params_regime1": {"drift": 0.01, "volatility": 0.15},
            "rs_params_regime2": {"drift": 0.04, "volatility": 0.30},
            "rs_transition_matrix": [[0.96, 0.04], [0.02, 0.98]]
        },
        {
            "name": "test_unknown_model_fallback",
            "baseline_carbon_price": 340.0,
            "carbon_price_volatility": 0.40,
            "carbon_price_model_type": "UNKNOWN_MODEL" 
        }
    ]

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_run_path = os.path.join("experiment_logs", f"master_run_test_models_{timestamp_str}")
    os.makedirs(master_run_path, exist_ok=True)
    logging.info(f"Master run output directory: {master_run_path}")

    general_sim_params = get_simulation_parameters()
    chp_parameters_static = get_chp_parameters()
    bess_parameters_static = get_bess_parameters()
    financial_instrument_specs_static = get_financial_instrument_specs()

    aggregated_results = []
    for config_item in experiment_configs: 
        config_item.setdefault("baseline_carbon_price", 300.0)
        config_item.setdefault("carbon_price_volatility", 0.20)
        
        summary, op_decision, strat_decision = run_single_experiment(
            config_item, 
            master_run_path,
            general_sim_params,
            chp_parameters_static,
            bess_parameters_static,
            get_market_parameters, 
            financial_instrument_specs_static,
            get_roa_ccs_project_parameters 
        )
        aggregated_results.append({
            "scenario_name": config_item["name"],
            "baseline_carbon_price": config_item["baseline_carbon_price"],
            "carbon_price_volatility": config_item["carbon_price_volatility"],
            "carbon_model_configured": config_item.get("carbon_price_model_type", "NotSet"),
            "carbon_model_actually_used": summary.get("carbon_price_scenario_summary", {}).get("model_used", "Error"),
            "operational_hedging_action": op_decision[0] if isinstance(op_decision, tuple) and len(op_decision) > 0 else "Error",
            "operational_hedging_option_label": op_decision[1].get('option_label', '') if isinstance(op_decision, tuple) and len(op_decision) > 1 and isinstance(op_decision[1], dict) else '',
            "strategic_ccs_investment_action": strat_decision.get('action', 'Error') 
        })
    
    agg_df = pd.DataFrame(aggregated_results)
    agg_csv_path = os.path.join(master_run_path, "aggregated_scenario_results_test_models.csv")
    agg_df.to_csv(agg_csv_path, index=False)
    logging.info(f"Saved aggregated scenario results to {agg_csv_path}")

    logging.info("Master Case Study Run (Test Models) Completed.")

if __name__ == "__main__":
    main()
