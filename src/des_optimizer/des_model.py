from pyomo.environ import ConcreteModel, Set, Param, Var, Constraint, Objective, SolverFactory, NonNegativeReals, Reals, Binary, Expression
import pandas as pd
import numpy as np
import pyomo.environ as pyo

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
    model.OPT = Set(initialize=data_inputs['option_types'], ordered=True) # Types of financial options

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


    # --- VARIABLES ---
    # First-stage variables (Option Purchase) - decided before scenario realization
    model.v_buy_option_contracts = Var(model.OPT, domain=NonNegativeReals) # Number of option contracts to buy for each type

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

    def chp_heat_production_rule(m, s, t):
        if m.p_chp_e_to_h_ratio > 0:
            return m.v_chp_gen_h[s,t] == m.v_chp_gen_e[s,t] / m.p_chp_e_to_h_ratio
        elif m.p_chp_cap_h_kwth > 0: 
             return m.v_chp_gen_h[s,t] <= m.p_chp_cap_h_kwth * m.v_chp_on[s,t]
        else: 
            return m.v_chp_gen_h[s,t] == 0
    model.c_chp_heat_production = Constraint(model.S, model.T, rule=chp_heat_production_rule)
    
    def chp_min_elec_gen_rule(m, s, t):
        return m.v_chp_gen_e[s,t] >= 0.1 * m.p_chp_cap_e_kw * m.v_chp_on[s,t] 
    # model.c_chp_min_elec_gen = Constraint(model.S, model.T, rule=chp_min_elec_gen_rule)

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
        return m.v_chp_gen_h[s,t] >= m.p_heat_demand[t] 
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

    # --- OBJECTIVE FUNCTION --- Expected Cost Minimization
    def expected_total_cost_rule(m):
        # First-stage cost (Option Premiums)
        option_purchase_cost = sum(m.v_buy_option_contracts[opt] * m.p_option_premium_cny_contract[opt] for opt in m.OPT)
        
        # Expected second-stage costs (Operational Costs for each scenario)
        expected_operational_cost = 0
        for s in m.S:
            grid_buy_cost_scen = sum(m.v_grid_import_e[s,t] * m.p_grid_price_buy_cny_kwh[t] for t in m.T)
            grid_sell_revenue_scen = sum(m.v_grid_export_e[s,t] * m.p_grid_price_sell_cny_kwh for t in m.T)
            chp_fuel_cost_scen = m.e_chp_fuel_consumption_m3_scen[s] * m.p_gas_price_cny_m3
            
            # Net carbon cost for the scenario (gross emissions cost - option payoffs)
            net_carbon_cost_scen = m.e_net_carbon_cost_scen[s]
            
            scenario_total_op_cost = grid_buy_cost_scen - grid_sell_revenue_scen + chp_fuel_cost_scen + net_carbon_cost_scen
            expected_operational_cost += m.p_scenario_prob[s] * scenario_total_op_cost
            
        return option_purchase_cost + expected_operational_cost
    
    model.o_expected_total_cost = Objective(rule=expected_total_cost_rule, sense=1) # 1 for minimize

    return model

def solve_des_model(model, solver_name='cbc', tee=True):
    """Solves the Pyomo model and returns results."""
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
    Extracts key results from the solved model for the (first) scenario.
    Designed for use with run_case_study.py where DES is run for one effective scenario.

    Args:
        model: The solved Pyomo model.
        data_inputs (dict): The input data dictionary. Used for context if needed.

    Returns:
        pd.DataFrame: DataFrame with detailed dispatch results for the (first) scenario.
        dict: Dictionary with summary operational results for the (first) scenario.
    """
    if not model.S:
        # Should not happen if run_case_study.py provides 's1'
        return pd.DataFrame(), {"error": "No scenarios in model"}

    s_label = next(iter(model.S)) # Get the single scenario label (e.g., 's1')

    # --- Part 1: Create Dispatch DataFrame ---
    time_periods = list(model.T)
    elec_demand_list = [model.p_elec_demand[t] for t in model.T]
    heat_demand_list = [model.p_heat_demand[t] for t in model.T]
    pv_avail_list = [model.p_pv_avail[t] for t in model.T] 
    
    pv_gen_list = [safe_val(model.v_pv_gen_e[s_label,t]) for t in model.T]
    pv_curtail_list = [safe_val(model.v_pv_curtail[s_label,t]) for t in model.T]
    chp_gen_e_list = [safe_val(model.v_chp_gen_e[s_label,t]) for t in model.T]
    chp_gen_h_list = [safe_val(model.v_chp_gen_h[s_label,t]) for t in model.T]
    chp_on_list = [safe_val(model.v_chp_on[s_label,t]) for t in model.T]
    bess_ch_p_list = [safe_val(model.v_bess_ch_p[s_label,t]) for t in model.T]
    bess_dis_p_list = [safe_val(model.v_bess_dis_p[s_label,t]) for t in model.T]
    bess_soc_kwh_list = [safe_val(model.v_bess_soc_kwh[s_label,t]) for t in model.T]
    grid_import_e_list = [safe_val(model.v_grid_import_e[s_label,t]) for t in model.T]
    grid_export_e_list = [safe_val(model.v_grid_export_e[s_label,t]) for t in model.T]
    
    carbon_prices_scen_hourly = [model.p_carbon_price_scen_cny_ton[s_label, t] for t in model.T]

    dispatch_df_data = {
        'time_step': time_periods,
        'elec_demand_kw': elec_demand_list,
        'heat_demand_kwth': heat_demand_list,
        'pv_available_kw': pv_avail_list,
        'carbon_price_cny_ton': carbon_prices_scen_hourly,
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

    # --- Part 2: Create Summary Dictionary ---
    summary_dict = {}
    
    summary_dict['scenario_label_in_model'] = str(s_label)
    # Directly use probability from data_inputs for clarity in this single-scenario context
    if s_label in data_inputs.get('scenario_probabilities', {}):
        summary_dict['scenario_probability'] = data_inputs['scenario_probabilities'][s_label]
    else:
        summary_dict['scenario_probability'] = safe_val(model.p_scenario_prob[s_label]) # Fallback

    grid_buy_cost_scen = sum(grid_import_e_list[t_idx] * model.p_grid_price_buy_cny_kwh[t] for t_idx, t in enumerate(model.T))
    grid_sell_revenue_scen_corr = sum(grid_export_e_list) * model.p_grid_price_sell_cny_kwh

    chp_fuel_consumption_scen_val = safe_val(model.e_chp_fuel_consumption_m3_scen[s_label])
    chp_fuel_cost_scen = chp_fuel_consumption_scen_val * model.p_gas_price_cny_m3
    
    gross_carbon_cost_scen_val = safe_val(model.e_gross_carbon_cost_scen[s_label])
    option_payoff_scen_val = safe_val(model.e_options_total_payoff_scen[s_label])
    net_carbon_cost_scen_val = gross_carbon_cost_scen_val - option_payoff_scen_val

    total_operational_cost_scen = grid_buy_cost_scen - grid_sell_revenue_scen_corr + chp_fuel_cost_scen + net_carbon_cost_scen_val

    summary_dict['total_operational_cost_cny'] = total_operational_cost_scen
    summary_dict['grid_buy_cost_cny'] = grid_buy_cost_scen
    summary_dict['grid_sell_revenue_cny'] = grid_sell_revenue_scen_corr
    summary_dict['chp_fuel_cost_cny'] = chp_fuel_cost_scen
    summary_dict['chp_fuel_consumption_m3'] = chp_fuel_consumption_scen_val
    summary_dict['gross_carbon_cost_cny'] = gross_carbon_cost_scen_val
    summary_dict['option_period_payoff_cny'] = option_payoff_scen_val
    summary_dict['net_carbon_cost_cny'] = net_carbon_cost_scen_val
    
    summary_dict['total_chp_co2_emissions_ton'] = safe_val(model.e_chp_co2_emissions_ton_scen[s_label])

    optimal_options = {}
    for opt_label in model.OPT:
        optimal_options[str(opt_label)] = safe_val(model.v_buy_option_contracts[opt_label])
    summary_dict['optimal_option_purchase_contracts'] = optimal_options
    
    option_purchase_cost_total = sum(safe_val(model.v_buy_option_contracts[opt]) * model.p_option_premium_cny_contract[opt] for opt in model.OPT)
    summary_dict['option_purchase_cost_total_cny'] = option_purchase_cost_total
    
    # For single scenario DES, the model objective should effectively be the total operational cost for that scenario
    # plus any option purchase costs (which are 0 in current DES setup via run_case_study.py)
    summary_dict['model_objective_expected_total_cost_cny'] = total_operational_cost_scen + option_purchase_cost_total 
    # summary_dict['model_objective_expected_total_cost_cny'] = safe_val(model.o_expected_total_cost) # Old way

    # Add aggregated physical quantities from dispatch_df
    summary_dict['total_elec_demand_kwh'] = dispatch_df['elec_demand_kw'].sum() # Assuming 1-hour time steps
    summary_dict['total_heat_demand_kwth_h'] = dispatch_df['heat_demand_kwth'].sum() # Assuming 1-hour time steps
    summary_dict['total_pv_available_kwh'] = dispatch_df['pv_available_kw'].sum()
    summary_dict['total_pv_gen_e_kwh'] = dispatch_df['pv_gen_e_kw'].sum()
    summary_dict['total_pv_curtail_kwh'] = dispatch_df['pv_curtail_kw'].sum()
    summary_dict['total_chp_gen_e_kwh'] = dispatch_df['chp_gen_e_kw'].sum()
    summary_dict['total_chp_gen_h_kwth_h'] = dispatch_df['chp_gen_h_kwth'].sum()
    summary_dict['total_bess_charge_kwh'] = dispatch_df['bess_charge_kw'].sum()
    summary_dict['total_bess_discharge_kwh'] = dispatch_df['bess_discharge_kw'].sum()
    summary_dict['total_grid_import_kwh'] = dispatch_df['grid_import_kw'].sum()
    summary_dict['total_grid_export_kwh'] = dispatch_df['grid_export_kw'].sum()

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