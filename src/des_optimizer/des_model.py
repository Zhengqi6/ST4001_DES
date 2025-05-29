from pyomo.environ import ConcreteModel, Set, Param, Var, Constraint, Objective, SolverFactory, NonNegativeReals, Reals, Binary, Expression, ConstraintList
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import logging

def build_des_model(data_inputs):
    """
    Builds the DES optimization Pyomo model, extended for stochastic optimization
    with carbon price scenarios and financial carbon options.

    Args:
        data_inputs (dict): A dictionary containing all necessary data:
            'time_horizon' (list/Set): List of time periods (e.g., hours for a day or week).
            'scenarios' (list/Set): List of carbon price scenarios.
            'option_types' (list/Set): List of available financial option types.
            
            'elec_demand_kw' (pd.Series): Electricity demand for each time period (same across scenarios).
            'heat_demand_kwth' (pd.Series): Heat demand for each time period (same across scenarios).
            'pv_avail_kw' (pd.Series): Available PV generation for each time period (same across scenarios).
            
            'chp_params' (dict): CHP parameters.
            'bess_params' (dict): BESS parameters.
            'market_params' (dict): Market parameters (natural_gas_price, tou_tariffs_cny_per_kwh_series, pv_feed_in_tariff_cny_per_kwh).
            'grid_params' (dict): Grid parameters.

            'scenario_probabilities' (dict): Probability for each scenario {s: prob_s}.
            'carbon_prices_scenario_cny_ton' (dict): Carbon price for each scenario and time {(s, t): price_st}.
            
            'option_strike_prices_cny_ton' (dict): Strike price for each option_type {opt: strike_k}.
            'option_premiums_cny_contract' (dict): Premium (cost) for each option_type {opt: premium_c}.
            'option_payoffs_cny_contract' (dict): Pre-calculated payoff for each option_type and scenario {(opt,s): payoff_os}.
                                                 (Payoff = max(0, scenario_avg_carbon_price_s - strike_k))
    """
    model = ConcreteModel(name="DES_Stochastic_Optimization_with_Options")

    # --- SETS ---
    model.T = Set(initialize=data_inputs['time_horizon'], ordered=True)
    model.S = Set(initialize=data_inputs['scenarios'], ordered=True) # Scenarios for carbon price
    model.OPT = Set(initialize=data_inputs['option_list'], ordered=True) # Types of financial options

    # --- PARAMETERS ---
    # Demand (assumed deterministic for now, can be made scenario-dependent if needed)
    # Data is expected to be a list or dict, indexable by model.T (0, 1, ...)
    model.p_elec_demand = Param(model.T, initialize=dict(enumerate(data_inputs['elec_demand_kw'])) if isinstance(data_inputs['elec_demand_kw'], list) else data_inputs['elec_demand_kw'])
    model.p_heat_demand = Param(model.T, initialize=dict(enumerate(data_inputs['heat_demand_kwth'])) if isinstance(data_inputs['heat_demand_kwth'], list) else data_inputs['heat_demand_kwth'])

    # PV (assumed deterministic)
    model.p_pv_avail = Param(model.T, initialize=dict(enumerate(data_inputs['pv_avail_kw'])) if isinstance(data_inputs['pv_avail_kw'], list) else data_inputs['pv_avail_kw'])

    # CHP
    model.p_chp_cap_e_kw = Param(initialize=data_inputs['chp_params']['capacity_kw'])
    model.p_chp_cap_h_kwth = Param(initialize=data_inputs['chp_params']['heat_capacity_kwth'])
    model.p_chp_gas_cons_m3_kwh_e = Param(initialize=data_inputs['chp_params']['gas_consumption_m3_per_kwh_e'])
    model.p_chp_co2_ton_m3_gas = Param(initialize=data_inputs['chp_params']['co2_emission_ton_per_m3_gas'])
    model.p_chp_e_to_h_ratio = Param(initialize=data_inputs['chp_params']['electricity_to_heat_ratio'])

    # BESS
    model.p_bess_cap_kwh = Param(initialize=data_inputs['bess_params']['capacity_kwh'])
    model.p_bess_max_p_kw = Param(initialize=data_inputs['bess_params']['power_kw'])
    model.p_bess_eff_ch = Param(initialize=data_inputs['bess_params']['charge_eff'])
    model.p_bess_eff_dis = Param(initialize=data_inputs['bess_params']['discharge_eff'])
    model.p_bess_soc_initial_kwh = Param(initialize=data_inputs['bess_params']['initial_soc_kwh'])
    model.p_bess_soc_min_kwh = Param(initialize=0.1 * data_inputs['bess_params']['capacity_kwh']) 
    model.p_bess_soc_max_kwh = Param(initialize=0.9 * data_inputs['bess_params']['capacity_kwh'])

    # Market (excluding carbon price here, as it's scenario-dependent)
    # tou_tariffs_cny_per_kwh_series is also expected to be list/dict indexed by model.T
    tou_tariffs_input = data_inputs['market_params']['tou_tariffs_cny_per_kwh_series']
    model.p_grid_price_buy_cny_kwh = Param(model.T, initialize=dict(enumerate(tou_tariffs_input)) if isinstance(tou_tariffs_input, list) else tou_tariffs_input)
    model.p_grid_price_sell_cny_kwh = Param(initialize=data_inputs['market_params']['pv_feed_in_tariff_cny_per_kwh'])
    model.p_gas_price_cny_m3 = Param(initialize=data_inputs['market_params']['natural_gas_price_cny_per_m3'])
    
    # Grid
    model.p_grid_max_import_kw = Param(initialize=data_inputs['grid_params'].get('max_import_kw', 10000)) 
    model.p_grid_max_export_kw = Param(initialize=data_inputs['grid_params'].get('max_export_kw', 10000))

    # Stochastic Parameters (Carbon Price and Options)
    model.p_scenario_prob = Param(model.S, initialize=data_inputs['scenario_probabilities'])
    # Assuming carbon price can vary per hour per scenario
    model.p_carbon_price_scen_cny_ton = Param(model.S, model.T, initialize=data_inputs['carbon_prices_scenario_cny_ton'])

    model.p_option_strike_cny_ton = Param(model.OPT, initialize=data_inputs['option_strike_prices_cny_ton'])
    model.p_option_premium_cny_contract = Param(model.OPT, initialize=data_inputs['option_premiums_cny_contract'])
    # Payoff = max(0, CarbonPrice_s_avg - Strike_opt). This should be calculated based on the relevant carbon price for the option period.
    # For simplicity, let's assume the payoff is pre-calculated against an average scenario carbon price relevant to the option's life.
    # If options are settled hourly against hourly carbon price, then p_option_payoff_cny_contract would need S and T indices.
    # For now, assume payoff is for the whole period for scenario s.
    model.p_option_period_payoff_cny_contract = Param(model.OPT, model.S, initialize=data_inputs['option_payoffs_cny_contract'])

    # New Parameters for CVaR
    model.p_cvar_alpha = Param(initialize=data_inputs.get('cvar_alpha_level', 0.95), within=Reals) # CVaR confidence level
    model.p_lambda_cvar = Param(initialize=data_inputs.get('lambda_cvar_weight', 0.0), within=NonNegativeReals) # Weight for CVaR in objective

    # --- VARIABLES ---
    # First-stage variables (Option Purchase) - decided before scenario realization
    # Define bounds for option contracts
    max_contracts = data_inputs.get('max_option_contracts_limit_abs', 250) # Default to +/- 250 if not provided
    def option_bounds_rule(m, opt):
        return (-max_contracts, max_contracts)
    
    model.v_buy_option_contracts = Var(model.OPT, domain=Reals, bounds=option_bounds_rule) # Number of option contracts to buy (positive) or sell (negative)

    # Second-stage variables (Operational) - scenario-dependent
    # PV
    model.v_pv_gen_e = Var(model.S, model.T, domain=NonNegativeReals) 
    model.v_pv_curtail = Var(model.S, model.T, domain=NonNegativeReals)

    # CHP
    model.v_chp_gen_e = Var(model.S, model.T, domain=NonNegativeReals) 
    model.v_chp_gen_h = Var(model.S, model.T, domain=NonNegativeReals) 
    model.v_chp_on = Var(model.S, model.T, domain=Binary)

    # BESS
    model.v_bess_ch_p = Var(model.S, model.T, domain=NonNegativeReals) 
    model.v_bess_dis_p = Var(model.S, model.T, domain=NonNegativeReals) 
    model.v_bess_soc_kwh = Var(model.S, model.T, domain=NonNegativeReals)

    # Grid
    model.v_grid_import_e = Var(model.S, model.T, domain=NonNegativeReals) 
    model.v_grid_export_e = Var(model.S, model.T, domain=NonNegativeReals) 

    # New Variables for CVaR calculation
    model.v_var_total_cost = Var(domain=Reals) # Value at Risk for total cost
    model.v_eta_total_cost_scen = Var(model.S, domain=NonNegativeReals) # Cost exceeding VaR for each scenario

    # --- CONSTRAINTS ---
    # Constraints now need to be indexed by scenario S as well

    # PV Generation
    def pv_generation_rule(m, s, t):
        return m.v_pv_gen_e[s,t] + m.v_pv_curtail[s,t] == m.p_pv_avail[t]
    model.c_pv_generation = Constraint(model.S, model.T, rule=pv_generation_rule)

    # CHP Operation
    def chp_max_elec_gen_rule(m, s, t):
        return m.v_chp_gen_e[s,t] <= m.p_chp_cap_e_kw * m.v_chp_on[s,t]
    model.c_chp_max_elec_gen = Constraint(model.S, model.T, rule=chp_max_elec_gen_rule)

    # Revised CHP Heat Production Logic
    # Constraint 1: Enforce E/H ratio if p_chp_e_to_h_ratio is defined
    def chp_e_h_ratio_rule(m, s, t):
        if m.p_chp_e_to_h_ratio > 0:
            # Assumes p_chp_e_to_h_ratio is E_gen / H_gen
            return m.v_chp_gen_h[s,t] * m.p_chp_e_to_h_ratio == m.v_chp_gen_e[s,t]
        else:
            return Constraint.Skip # No fixed ratio to enforce
    model.c_chp_e_h_ratio = Constraint(model.S, model.T, rule=chp_e_h_ratio_rule)

    # Constraint 2: Enforce maximum heat generation capacity based on v_chp_on
    def chp_max_heat_gen_rule(m, s, t):
        if m.p_chp_cap_h_kwth > 0:
            return m.v_chp_gen_h[s,t] <= m.p_chp_cap_h_kwth * m.v_chp_on[s,t]
        else:
            # If no heat capacity defined, heat generation should be zero (unless defined by ratio and E gen)
            # If ratio is also zero, then this path means H must be 0.
            if not (m.p_chp_e_to_h_ratio > 0):
                 return m.v_chp_gen_h[s,t] == 0
            return Constraint.Skip # Heat is determined by E/H ratio if H_cap is 0 but E/H ratio exists
    model.c_chp_max_heat_gen = Constraint(model.S, model.T, rule=chp_max_heat_gen_rule)

    def chp_min_elec_gen_rule(m, s, t):
        return m.v_chp_gen_e[s,t] >= 0.1 * m.p_chp_cap_e_kw * m.v_chp_on[s,t] 
    model.c_chp_min_elec_gen = Constraint(model.S, model.T, rule=chp_min_elec_gen_rule)

    # BESS Operation
    def bess_soc_rule(m, s, t):
        if t == m.T.first():
            return m.v_bess_soc_kwh[s,t] == m.p_bess_soc_initial_kwh + \
                                        m.v_bess_ch_p[s,t] * m.p_bess_eff_ch - \
                                        m.v_bess_dis_p[s,t] / m.p_bess_eff_dis
        return m.v_bess_soc_kwh[s,t] == m.v_bess_soc_kwh[s,m.T.prev(t)] + \
                                    m.v_bess_ch_p[s,t] * m.p_bess_eff_ch - \
                                    m.v_bess_dis_p[s,t] / m.p_bess_eff_dis
    model.c_bess_soc = Constraint(model.S, model.T, rule=bess_soc_rule)

    def bess_soc_min_rule(m, s, t):
        return m.v_bess_soc_kwh[s,t] >= m.p_bess_soc_min_kwh
    model.c_bess_soc_min = Constraint(model.S, model.T, rule=bess_soc_min_rule)

    def bess_soc_max_rule(m, s, t):
        return m.v_bess_soc_kwh[s,t] <= m.p_bess_soc_max_kwh
    model.c_bess_soc_max = Constraint(model.S, model.T, rule=bess_soc_max_rule)

    def bess_charge_power_rule(m, s, t):
        return m.v_bess_ch_p[s,t] <= m.p_bess_max_p_kw
    model.c_bess_charge_power = Constraint(model.S, model.T, rule=bess_charge_power_rule)

    def bess_discharge_power_rule(m, s, t):
        return m.v_bess_dis_p[s,t] <= m.p_bess_max_p_kw
    model.c_bess_discharge_power = Constraint(model.S, model.T, rule=bess_discharge_power_rule)
    
    # Grid Operation
    def grid_import_limit_rule(m, s, t):
        return m.v_grid_import_e[s,t] <= m.p_grid_max_import_kw
    model.c_grid_import_limit = Constraint(model.S, model.T, rule=grid_import_limit_rule)

    def grid_export_limit_rule(m, s, t):
        return m.v_grid_export_e[s,t] <= m.p_grid_max_export_kw
    model.c_grid_export_limit = Constraint(model.S, model.T, rule=grid_export_limit_rule)

    # Energy Balance
    def electricity_balance_rule(m, s, t):
        return (m.v_pv_gen_e[s,t] + 
                m.v_chp_gen_e[s,t] + 
                m.v_bess_dis_p[s,t] + 
                m.v_grid_import_e[s,t] == 
                m.p_elec_demand[t] + 
                m.v_bess_ch_p[s,t] + 
                m.v_grid_export_e[s,t])
    model.c_electricity_balance = Constraint(model.S, model.T, rule=electricity_balance_rule)

    def heat_balance_rule(m, s, t):
        return m.v_chp_gen_h[s,t] >= m.p_heat_demand[t]  # Restoring original strict constraint
    model.c_heat_balance = Constraint(model.S, model.T, rule=heat_balance_rule)

    # --- Expressions for cost components (per scenario) ---
    # These help in making the objective function cleaner
    def chp_fuel_consumption_scenario_rule(m, s):
        return sum(m.v_chp_gen_e[s,t] * m.p_chp_gas_cons_m3_kwh_e for t in m.T)
    model.e_chp_fuel_consumption_m3_scen = Expression(model.S, rule=chp_fuel_consumption_scenario_rule)

    def chp_co2_emissions_scenario_rule(m, s):
        return m.e_chp_fuel_consumption_m3_scen[s] * m.p_chp_co2_ton_m3_gas
    model.e_chp_co2_emissions_ton_scen = Expression(model.S, rule=chp_co2_emissions_scenario_rule)
    
    # If carbon price is hourly:
    # def carbon_cost_hourly_scenario_rule(m,s):
    #     return sum(m.v_chp_gen_e[s,t] * m.p_chp_gas_cons_m3_kwh_e * m.p_chp_co2_ton_m3_gas * m.p_carbon_price_scen_cny_ton[s,t] for t in m.T)
    # model.e_carbon_cost_gross_scen = Expression(model.S, rule=carbon_cost_hourly_scenario_rule)

    # If carbon price is an average for the period for scenario s, then payoff is also for that period.
    # Let's assume p_carbon_price_scen_cny_ton[s,t] is used and payoff is also precalculated based on this time-series.
    # For simplicity in this stage, let's assume the payoff p_option_period_payoff_cny_contract[opt,s]
    # is the total payoff for option 'opt' in scenario 's' over the entire period if one unit of option is held.
    # And the carbon cost is based on hourly emissions and hourly carbon prices.

    def gross_carbon_cost_scenario_rule(m, s):
        return sum(m.v_chp_gen_e[s,t] * m.p_chp_gas_cons_m3_kwh_e * m.p_chp_co2_ton_m3_gas * m.p_carbon_price_scen_cny_ton[s,t] for t in m.T)
    model.e_gross_carbon_cost_scen = Expression(model.S, rule=gross_carbon_cost_scenario_rule)

    def options_total_payoff_scenario_rule(m, s):
        return sum(m.v_buy_option_contracts[opt] * m.p_option_period_payoff_cny_contract[opt,s] for opt in m.OPT)
    model.e_options_total_payoff_scen = Expression(model.S, rule=options_total_payoff_scenario_rule)

    def net_carbon_cost_scenario_rule(m, s):
        return m.e_gross_carbon_cost_scen[s] - m.e_options_total_payoff_scen[s]
    model.e_net_carbon_cost_scen = Expression(model.S, rule=net_carbon_cost_scenario_rule)

    # Expression for Total Operational Cost per Scenario (excluding first-stage option purchase costs)
    def total_operational_cost_scenario_rule(m, s):
        grid_buy_cost_s = sum(m.v_grid_import_e[s,t] * m.p_grid_price_buy_cny_kwh[t] for t in m.T)
        grid_sell_revenue_s = sum(m.v_grid_export_e[s,t] * m.p_grid_price_sell_cny_kwh for t in m.T)
        chp_fuel_cost_s = m.e_chp_fuel_consumption_m3_scen[s] * m.p_gas_price_cny_m3
        # net_carbon_cost_s is already m.e_net_carbon_cost_scen[s]
        return grid_buy_cost_s - grid_sell_revenue_s + chp_fuel_cost_s + m.e_net_carbon_cost_scen[s]
    model.e_total_operational_cost_scen = Expression(model.S, rule=total_operational_cost_scenario_rule)

    # Expression for total first-stage option purchase cost
    def total_option_purchase_cost_rule(m):
        return sum(m.v_buy_option_contracts[opt] * m.p_option_premium_cny_contract[opt] for opt in m.OPT)
    model.e_total_option_purchase_cost = Expression(rule=total_option_purchase_cost_rule)

    # Constraints for CVaR calculation
    def eta_definition_rule(m, s):
        # Cost for scenario s INCLUDING first-stage option costs
        cost_for_cvar_s = m.e_total_operational_cost_scen[s] + m.e_total_option_purchase_cost
        return m.v_eta_total_cost_scen[s] >= cost_for_cvar_s - m.v_var_total_cost
    model.c_eta_definition_scen = Constraint(model.S, rule=eta_definition_rule)

    # Expression for CVaR of Total Cost
    def cvar_total_cost_rule(m):
        if (1 - m.p_cvar_alpha) <= 1e-6: # Avoid division by zero if alpha is 1 or very close
            # If alpha is 1, CVaR is effectively the max cost or average of worst-case scenarios,
            # This formulation might not be robust for alpha=1 directly. Consider specific handling or error.
            # For simplicity, return a large penalty or just the VaR if lambda is non-zero.
            # A more robust way for alpha=1 would be to take the max of total_operational_cost_scen.
            # However, typical CVaR usage has alpha < 1.
            logging.warning("CVaR alpha is very close to 1, CVaR calculation might be unstable or ill-defined with this formulation.")
            return m.v_var_total_cost # Fallback, or consider raising an error.
        return m.v_var_total_cost + (1 / (1 - m.p_cvar_alpha)) * sum(m.p_scenario_prob[s] * m.v_eta_total_cost_scen[s] for s in m.S)
    model.e_cvar_total_cost = Expression(rule=cvar_total_cost_rule)

    # --- OBJECTIVE FUNCTION --- Minimize (Expected Cost + Lambda * CVaR_Cost)
    def expected_total_cost_rule(m):
        # First-stage cost (Option Premiums) is now captured by model.e_total_option_purchase_cost
        # expected_operational_cost is sum(m.p_scenario_prob[s] * m.e_total_operational_cost_scen[s] for s in m.S)
        
        # Total objective
        return model.e_total_option_purchase_cost + sum(m.p_scenario_prob[s] * m.e_total_operational_cost_scen[s] for s in m.S) + m.p_lambda_cvar * model.e_cvar_total_cost
    
    model.objective = Objective(rule=expected_total_cost_rule, sense=pyo.minimize)

    return model

def solve_des_model(model, solver_name='cbc', tee=True):
    """Solves the Pyomo model and returns results."""
    # model.write("debug_des_model.lp", io_options={"symbolic_solver_labels": True})
    # print("！！！！！！！！ DEBUG: Model written to debug_des_model.lp ！！！！！！！！")
    solver = SolverFactory(solver_name)
    results = solver.solve(model, tee=tee)
    return results, model # Return both solver results and the solved model

def safe_val(var_component):
    if hasattr(var_component, 'value') and not callable(var_component.value):
        if var_component.value is None: return 0
        return var_component.value
    if callable(var_component):
        try:
            val = var_component() 
            return val if val is not None else 0
        except Exception:
            pass 
    try:
        py_val = pyo.value(var_component, exception_flag=False) 
        return py_val if py_val is not None else 0
    except Exception:
        return 0

def extract_results(model, data_inputs):
    """
    Extracts key results from the solved model.
    For stochastic models, it returns first-stage decisions, expected costs,
    and dispatch/cost details for the FIRST scenario as a sample.

    Args:
        model: The solved Pyomo model.
        data_inputs (dict): The input data dictionary.

    Returns:
        pd.DataFrame: DataFrame with detailed dispatch results for the FIRST scenario.
        dict: Dictionary with summary operational results (first-stage decisions, expected costs).
    """
    if not model.S:
        return pd.DataFrame(), {"error": "No scenarios in model"}

    # For dispatch details, pick the first scenario as a representative sample
    # All scenarios are available in model.S if specific scenario analysis is needed elsewhere
    s_label_sample = model.S.first() 
    logging.info(f"Extracting dispatch results for sample scenario: {s_label_sample}")

    # --- Part 1: Create Dispatch DataFrame (for the sample scenario) ---
    time_periods = list(model.T)
    elec_demand_list = [model.p_elec_demand[t] for t in model.T]
    heat_demand_list = [model.p_heat_demand[t] for t in model.T]
    pv_avail_list = [model.p_pv_avail[t] for t in model.T] 
    
    pv_gen_list = [safe_val(model.v_pv_gen_e[s_label_sample,t]) for t in model.T]
    pv_curtail_list = [safe_val(model.v_pv_curtail[s_label_sample,t]) for t in model.T]
    chp_gen_e_list = [safe_val(model.v_chp_gen_e[s_label_sample,t]) for t in model.T]
    chp_gen_h_list = [safe_val(model.v_chp_gen_h[s_label_sample,t]) for t in model.T]
    chp_on_list = [safe_val(model.v_chp_on[s_label_sample,t]) for t in model.T]
    bess_ch_p_list = [safe_val(model.v_bess_ch_p[s_label_sample,t]) for t in model.T]
    bess_dis_p_list = [safe_val(model.v_bess_dis_p[s_label_sample,t]) for t in model.T]
    bess_soc_kwh_list = [safe_val(model.v_bess_soc_kwh[s_label_sample,t]) for t in model.T]
    grid_import_e_list = [safe_val(model.v_grid_import_e[s_label_sample,t]) for t in model.T]
    grid_export_e_list = [safe_val(model.v_grid_export_e[s_label_sample,t]) for t in model.T]
    
    # Carbon prices for the sample scenario's dispatch
    carbon_prices_sample_scen_hourly = [model.p_carbon_price_scen_cny_ton[s_label_sample, t] for t in model.T]

    dispatch_df_data = {
        'time_step': time_periods,
        'elec_demand_kw': elec_demand_list, # Deterministic
        'heat_demand_kwth': heat_demand_list, # Deterministic
        'pv_available_kw': pv_avail_list, # Deterministic
        'carbon_price_cny_ton': carbon_prices_sample_scen_hourly, # For sample scenario
        'pv_gen_e_kw': pv_gen_list,
        'pv_curtail_kw': pv_curtail_list,
        'chp_gen_e_kw': chp_gen_e_list,
        'chp_gen_h_kwth': chp_gen_h_list,
        'chp_on_state': chp_on_list,
        'bess_charge_kw': bess_ch_p_list,
        'bess_discharge_kw': bess_dis_p_list,
        'bess_soc_kwh': bess_soc_kwh_list,
        'grid_import_kw': grid_import_e_list,
        'grid_export_kw': grid_export_e_list
    }
    dispatch_df = pd.DataFrame(dispatch_df_data).set_index('time_step')

    # --- Part 2: Create Summary Dictionary (First-stage decisions and Expected Values) ---
    summary_dict = {}

    # First-stage decisions (option purchases)
    optimal_options = {}
    for opt_label in model.OPT:
        optimal_options[str(opt_label)] = safe_val(model.v_buy_option_contracts[opt_label])
    summary_dict['optimal_option_purchase_contracts'] = optimal_options
    
    option_purchase_cost_total = sum(safe_val(model.v_buy_option_contracts[opt]) * model.p_option_premium_cny_contract[opt] for opt in model.OPT)
    summary_dict['option_purchase_cost_total_cny'] = option_purchase_cost_total

    # Overall model objective (Expected Total Cost)
    summary_dict['model_objective_expected_total_cost_cny'] = safe_val(model.objective)
    
    # Expected operational cost (derived from objective and first-stage cost)
    summary_dict['expected_operational_cost_cny'] = summary_dict['model_objective_expected_total_cost_cny'] - summary_dict['option_purchase_cost_total_cny']

    # Calculate expected values for other key metrics by averaging over scenarios
    exp_grid_buy_cost = 0
    exp_grid_sell_revenue = 0
    exp_chp_fuel_cost = 0
    exp_chp_fuel_consumption_m3 = 0
    exp_gross_carbon_cost = 0
    exp_option_period_payoff = 0 # This is expected payoff from purchased options
    exp_net_carbon_cost = 0
    exp_total_chp_co2_emissions_ton = 0
    
    # Physical quantities (summed over time for each scenario, then averaged)
    # These are useful if we want expected physical flows, not just costs
    exp_total_pv_gen_e_kwh = 0
    exp_total_pv_curtail_kwh = 0
    exp_total_chp_gen_e_kwh = 0
    exp_total_chp_gen_h_kwth_h = 0
    exp_total_bess_charge_kwh = 0
    exp_total_bess_discharge_kwh = 0
    exp_total_grid_import_kwh = 0
    exp_total_grid_export_kwh = 0

    logging.info("DEBUG_EXTRACT_RESULTS: Calculating expected option payoffs...")
    for s_idx, s in enumerate(model.S):
        prob_s = model.p_scenario_prob[s]

        # Costs for this scenario
        scen_grid_buy_cost = sum(safe_val(model.v_grid_import_e[s,t]) * model.p_grid_price_buy_cny_kwh[t] for t in model.T)
        scen_grid_sell_revenue = sum(safe_val(model.v_grid_export_e[s,t]) * model.p_grid_price_sell_cny_kwh for t in model.T) # p_grid_price_sell_cny_kwh is a single param
        scen_chp_fuel_consumption_m3 = safe_val(model.e_chp_fuel_consumption_m3_scen[s])
        scen_chp_fuel_cost = scen_chp_fuel_consumption_m3 * model.p_gas_price_cny_m3
        scen_gross_carbon_cost = safe_val(model.e_gross_carbon_cost_scen[s])
        
        # Detailed logging for option payoff calculation for this scenario
        scen_option_payoff_val = safe_val(model.e_options_total_payoff_scen[s])
        if s_idx < 3: # Log details only for the first few scenarios to avoid excessive output
            logging.info(f"  DEBUG_EXTRACT_RESULTS: Scenario {s} (Prob: {prob_s:.4f}):")
            logging.info(f"    model.e_options_total_payoff_scen[{s}] = {scen_option_payoff_val:.4f}")
            for opt_idx, opt_label_debug in enumerate(model.OPT):
                contracts_val = safe_val(model.v_buy_option_contracts[opt_label_debug])
                unit_payoff_val = model.p_option_period_payoff_cny_contract[opt_label_debug, s]
                total_contrib = contracts_val * unit_payoff_val
                if abs(contracts_val) > 1e-4 or abs(unit_payoff_val) > 1e-4 : # Log if significant
                    logging.info(f"      Opt {opt_idx}: {opt_label_debug} -> Contracts: {contracts_val:.2f}, UnitPayoff(buyer): {unit_payoff_val:.4f}, Contribution: {total_contrib:.4f}")
        
        # Expected Net Carbon Cost for this scenario (for checking)
        scen_net_carbon_cost_val = safe_val(model.e_net_carbon_cost_scen[s]) # Correct way
        logging.info(f"    model.e_net_carbon_cost_scen[{s}] (Gross - OptPayoff) = {scen_net_carbon_cost_val:.2f}") # Corrected line

        exp_option_period_payoff += prob_s * scen_option_payoff_val
        exp_grid_buy_cost += prob_s * scen_grid_buy_cost
        exp_grid_sell_revenue += prob_s * scen_grid_sell_revenue
        exp_chp_fuel_cost += prob_s * scen_chp_fuel_cost
        exp_chp_fuel_consumption_m3 += prob_s * scen_chp_fuel_consumption_m3
        exp_gross_carbon_cost += prob_s * scen_gross_carbon_cost
        exp_net_carbon_cost += prob_s * scen_net_carbon_cost_val
        exp_total_chp_co2_emissions_ton += prob_s * scen_chp_fuel_consumption_m3 * model.p_chp_co2_ton_m3_gas
        
        # Summing hourly physical quantities for scenario s, then weighting by probability
        exp_total_pv_gen_e_kwh += prob_s * sum(safe_val(model.v_pv_gen_e[s,t]) for t in model.T)
        exp_total_pv_curtail_kwh += prob_s * sum(safe_val(model.v_pv_curtail[s,t]) for t in model.T)
        exp_total_chp_gen_e_kwh += prob_s * sum(safe_val(model.v_chp_gen_e[s,t]) for t in model.T)
        exp_total_chp_gen_h_kwth_h += prob_s * sum(safe_val(model.v_chp_gen_h[s,t]) for t in model.T)
        exp_total_bess_charge_kwh += prob_s * sum(safe_val(model.v_bess_ch_p[s,t]) for t in model.T)
        exp_total_bess_discharge_kwh += prob_s * sum(safe_val(model.v_bess_dis_p[s,t]) for t in model.T)
        exp_total_grid_import_kwh += prob_s * sum(safe_val(model.v_grid_import_e[s,t]) for t in model.T)
        exp_total_grid_export_kwh += prob_s * sum(safe_val(model.v_grid_export_e[s,t]) for t in model.T)

    summary_dict['expected_grid_buy_cost_cny'] = exp_grid_buy_cost
    summary_dict['expected_grid_sell_revenue_cny'] = exp_grid_sell_revenue
    summary_dict['expected_chp_fuel_cost_cny'] = exp_chp_fuel_cost
    summary_dict['expected_chp_fuel_consumption_m3'] = exp_chp_fuel_consumption_m3
    summary_dict['expected_gross_carbon_cost_cny'] = exp_gross_carbon_cost
    summary_dict['expected_option_period_payoff_cny'] = exp_option_period_payoff
    summary_dict['expected_net_carbon_cost_cny'] = exp_net_carbon_cost
    summary_dict['expected_total_chp_co2_emissions_ton'] = exp_total_chp_co2_emissions_ton

    # Deterministic demands and PV availability (can be summed directly from dispatch_df if needed, or from params)
    # These are useful for context, already available in dispatch_df or model.p_...
    summary_dict['total_elec_demand_kwh'] = dispatch_df['elec_demand_kw'].sum() 
    summary_dict['total_heat_demand_kwth_h'] = dispatch_df['heat_demand_kwth'].sum()
    summary_dict['total_pv_available_kwh'] = dispatch_df['pv_available_kw'].sum()
    
    # Add expected physical quantities (these are new)
    summary_dict['expected_total_pv_gen_e_kwh'] = exp_total_pv_gen_e_kwh
    summary_dict['expected_total_pv_curtail_kwh'] = exp_total_pv_curtail_kwh
    summary_dict['expected_total_chp_gen_e_kwh'] = exp_total_chp_gen_e_kwh
    summary_dict['expected_total_chp_gen_h_kwth_h'] = exp_total_chp_gen_h_kwth_h
    summary_dict['expected_total_bess_charge_kwh'] = exp_total_bess_charge_kwh
    summary_dict['expected_total_bess_discharge_kwh'] = exp_total_bess_discharge_kwh
    summary_dict['expected_total_grid_import_kwh'] = exp_total_grid_import_kwh
    summary_dict['expected_total_grid_export_kwh'] = exp_total_grid_export_kwh
    
    # For compatibility with existing run_case_study structure that might look for these specific keys:
    # We fill them with expected values.
    summary_dict['total_operational_cost_cny'] = summary_dict['expected_operational_cost_cny']
    summary_dict['grid_buy_cost_cny'] = summary_dict['expected_grid_buy_cost_cny']
    summary_dict['grid_sell_revenue_cny'] = summary_dict['expected_grid_sell_revenue_cny']
    summary_dict['chp_fuel_cost_cny'] = summary_dict['expected_chp_fuel_cost_cny']
    summary_dict['gross_carbon_cost_cny'] = summary_dict['expected_gross_carbon_cost_cny']
    summary_dict['option_period_payoff_cny'] = summary_dict['expected_option_period_payoff_cny'] # Payoff from purchased options
    summary_dict['net_carbon_cost_cny'] = summary_dict['expected_net_carbon_cost_cny']
    summary_dict['total_chp_co2_emissions_ton'] = summary_dict['expected_total_chp_co2_emissions_ton']
    # scenario_probability and scenario_label_in_model are no longer single values in summary_dict.
    # If needed by run_case_study, these would need to be handled differently, e.g. by pickling full scenario results.

    # Extract VaR and CVaR values if they exist on the model (i.e., lambda_cvar_weight was > 0 or they were defined)
    if hasattr(model, 'v_var_total_cost') and hasattr(model, 'e_cvar_total_cost'):
        summary_dict['value_at_risk_total_cost_cny'] = safe_val(model.v_var_total_cost)
        summary_dict['conditional_value_at_risk_total_cost_cny'] = safe_val(model.e_cvar_total_cost)
    else:
        summary_dict['value_at_risk_total_cost_cny'] = None
        summary_dict['conditional_value_at_risk_total_cost_cny'] = None

    # Costs for the first sample scenario (or an average/representative scenario)
    s_sample = model.S.first() # Use the first scenario as a sample

    return dispatch_df, summary_dict


if __name__ == '__main__':
    # This __main__ section needs to be completely rewritten to use the new stochastic model structure.
    # It will involve:
    # 1. Generating carbon price scenarios (e.g., list of Series or dict of dicts for p_carbon_price_scen_cny_ton)
    # 2. Defining option types, their strike prices, and calculating their premiums and scenario-specific payoffs.
    # 3. Preparing the full data_inputs dictionary for build_des_model.
    # 4. Building, solving, and extracting results.
    # 5. Analyzing and printing/plotting stochastic results (e.g., option purchase, expected costs, scenario-wise dispatch).

    print("DES model (stochastic with options) defined. Run run_case_study.py for an example.")

    # Example structure for data_inputs (conceptual)
    # data_inputs_stochastic = {
    #     'time_horizon': list(range(24)), # e.g., 24 hours
    #     'scenarios': ['s1', 's2', 's3'], 
    #     'option_types': ['opt_call_200', 'opt_call_220'],
    #     'option_labels': {'opt_call_200': 'Call_200_1M', 'opt_call_220': 'Call_220_1M'}, # For readable results
        
    #     'elec_demand_kw': pd.Series(...), 
    #     'heat_demand_kwth': pd.Series(...),
    #     'pv_avail_kw': pd.Series(...),
        
    #     'chp_params': {...}, 
    #     'bess_params': {...},
    #     'market_params': {...}, # tou_tariffs_cny_per_kwh_series will be a pd.Series
    #     'grid_params': {},

    #     'scenario_probabilities': {'s1': 0.3, 's2': 0.4, 's3': 0.3},
    #     # carbon_prices_scenario_cny_ton: {(s, t): price_st}
    #     # Example: {'s1': {0: 190, 1: 192,...}, 's2': {0: 205, 1: 208,...}}
    #     # This needs to be compatible with Param(model.S, model.T, initialize=...)
    #     # So, a nested dict like {(scenario_label, time_idx): value}
    #     'carbon_prices_scenario_cny_ton': {('s1',0):190, ('s1',1):192, ('s2',0):205, ('s2',1):208 ...},
        
    #     'option_strike_prices_cny_ton': {'opt_call_200': 200, 'opt_call_220': 220},
    #     'option_premiums_cny_contract': {'opt_call_200': 5.0, 'opt_call_220': 2.5}, # Calculated by Black-Scholes
    #     # option_payoffs_cny_contract: {(opt, s): payoff_os}
    #     'option_payoffs_cny_contract': {('opt_call_200', 's1'): max(0, avg_price_s1 - 200), ...} 
    # }
    pass

# Example usage (illustrative, would be called from a main script)
if __name__ == '__main__':
    # --- 1. Load Data (using data_preparation utilities) ---
    from src.utils.data_preparation import (
        load_electricity_demand, load_heat_demand, load_pv_generation_factor,
        get_chp_parameters, get_bess_parameters, get_market_parameters, 
        get_simulation_parameters
    )

    sim_params = get_simulation_parameters()
    op_days = sim_params['operational_simulation_days']
    time_horizon_op = pd.date_range(start='2023-01-01', periods=24 * op_days, freq='h')

    elec_demand_full = load_electricity_demand()
    heat_demand_full = load_heat_demand()
    pv_factor_full = load_pv_generation_factor()

    # Slice data for the operational horizon
    elec_demand_op = elec_demand_full.loc[time_horizon_op, 'electricity_demand_kw']
    heat_demand_op = heat_demand_full.loc[time_horizon_op, 'heat_demand_kwth']
    pv_avail_op = pv_factor_full[time_horizon_op] * sim_params['pv_capacity_kwp']

    chp_params = get_chp_parameters()
    bess_params = get_bess_parameters()
    market_data = get_market_parameters()

    # Create TOU series for the operational horizon
    tou_price_series_op = pd.Series(index=time_horizon_op, dtype=float)
    for hour in time_horizon_op:
        if (10 <= hour.hour < 15) or (18 <= hour.hour < 21):
            tou_price_series_op[hour] = market_data['tou_tariffs_cny_per_kwh']['peak']
        elif (23 <= hour.hour) or (hour.hour < 7):
            tou_price_series_op[hour] = market_data['tou_tariffs_cny_per_kwh']['valley']
        else:
            tou_price_series_op[hour] = market_data['tou_tariffs_cny_per_kwh']['flat']
    
    market_params_op = {
        'tou_tariffs_cny_per_kwh_series': tou_price_series_op,
        'pv_feed_in_tariff_cny_per_kwh': market_data['pv_feed_in_tariff_cny_per_kwh'],
        'natural_gas_price_cny_per_m3': market_data['natural_gas_price_cny_per_m3'],
        'carbon_price_cny_per_ton': market_data['carbon_price_baseline_cny_per_ton'] 
    }

    grid_params_op = {
        'max_import_kw': 1000, # Example limit
        'max_export_kw': sim_params['pv_capacity_kwp'] # Example limit, can't export more than PV cap
    }

    model_inputs = {
        'time_horizon': list(time_horizon_op),
        'elec_demand_kw': elec_demand_op,
        'heat_demand_kwth': heat_demand_op,
        'pv_avail_kw': pv_avail_op,
        'chp_params': chp_params,
        'bess_params': bess_params,
        'market_params': market_params_op,
        'grid_params': grid_params_op
    }

    # --- 2. Build Model ---
    print("\nBuilding DES model...")
    des_model = build_des_model(model_inputs)
    print("Model built.")

    # --- 3. Solve Model ---
    # Ensure you have a solver like CBC or GLPK installed and in your PATH
    # Or Gurobi/CPLEX if you have licenses and they are configured
    print("\nSolving DES model (using CBC)...")
    # Solvers: glpk, cbc, gurobi, cplex
    solver_name = 'cbc' 
    try:
        solution_results, solved_model = solve_des_model(des_model, solver_name=solver_name, tee=True)
        print(f"Model solved with {solver_name}.")
        print("Solver status:", solution_results.solver.status)
        print("Solver termination condition:", solution_results.solver.termination_condition)

        # --- 4. Extract and Display Results ---
        if solution_results.solver.termination_condition == 'optimal' or solution_results.solver.termination_condition == 'feasible':
            results_summary = extract_results(solved_model, model_inputs)
            print("\n--- Operational Summary ---")
            for key, value in results_summary['scenario_details'][0]['costs'].items():
                print(f"{key}: {value:.2f}")
            
            print("\n--- Scenario Details ---")
            for scenario, details in results_summary['scenario_details'].items():
                print(f"\nScenario: {scenario}")
                print("Dispatch DataFrame:")
                print(details['dispatch_df'])
                print("Costs:")
                print(details['costs'])
                print("Emissions (ton):", details['emissions_ton'])
                print("Probability:", details['probability'])
        else:
            print(f"Solver did not find an optimal solution. Status: {solution_results.solver.status}, Condition: {solution_results.solver.termination_condition}")
    except Exception as e:
        print(f"An error occurred during model solving or result extraction: {e}")
        print("Please ensure a MILP solver (like CBC, GLPK, Gurobi, or CPLEX) is installed and accessible in your system PATH.")
        print("If using CBC/GLPK, they are often bundled with Pyomo or can be installed separately.")
        print("For Gurobi/CPLEX, ensure licenses are active and environment variables (e.g., GRB_LICENSE_FILE) are set.") 