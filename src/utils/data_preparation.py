import pandas as pd
import numpy as np

# Constants from the case study
PV_CAPACITY_KWP = 500
CHP_CAPACITY_KW = 200
CHP_HEAT_CAPACITY_KWTH = 250
CHP_GAS_CONSUMPTION_M3_PER_KWH_E = 0.25 # m3 of gas per kWh of electricity produced (example value)
CHP_CO2_EMISSION_TON_PER_M3_GAS = 0.002 # ton CO2 per m3 of gas (example value, typically ~0.0018-0.0022)

BESS_CAPACITY_KWH = 300
BESS_POWER_KW = 100
BESS_CHARGE_EFF = 0.95
BESS_DISCHARGE_EFF = 0.95

ANNUAL_ELECTRICITY_DEMAND_GWH = 2
ANNUAL_HEAT_DEMAND_GWH = 1.5

TOU_TARIFFS = {'peak': 1.2, 'flat': 0.7, 'valley': 0.3}  # CNY/kWh
PV_FEED_IN_TARIFF = 0.4  # CNY/kWh
NATURAL_GAS_PRICE = 2.5  # CNY/m3
CARBON_PRICE_BASELINE = 200  # CNY/ton CO2
CARBON_PRICE_VOLATILITY_ANNUAL = 0.20 # Example

RISK_FREE_RATE = 0.03

# CCS ROA Parameters
CCS_INVESTMENT_COST = 300000  # CNY
CCS_CAPTURE_EFFICIENCY = 0.90
CCS_OPEX_INCREASE_PER_KWH_CHP = 0.1 # CNY/kWh of CHP generation
CCS_PROJECT_LIFETIME_YEARS = 15
CCS_MAX_DEFERRAL_YEARS = 3

# Global Simulation Parameters (Centralized)
DES_OPTIMIZATION_HORIZON_HOURS = 24 * 7 # E.g., 7 days
NUM_CARBON_PRICE_SCENARIOS = 30
CARBON_SCENARIO_HORIZON_DAYS = 90
CARBON_PRICE_GBM_DRIFT = 0.08  # Annual drift for GBM
# CARBON_PRICE_VOLATILITY_ANNUAL is a scenario parameter, but a default could be here if needed
ROA_LATTICE_STEPS = 100
RISK_AVERSION_FACTOR = 0.5 # Example value for decision making
NPV_HURDLE_RATE = 50000 # Example CNY for CCS investment decision
GRID_MAX_IMPORT_EXPORT_KW = 1000 # Example grid constraint
CARBON_PRICE_MODEL_TYPE = "GBM" # Can be "GBM" or "GARCH"


def load_electricity_demand(raw_data_path="data/raw/electricity_demand.csv", processed_data_path="data/processed/electricity_demand_processed.csv"):
    """
    Loads and preprocesses electricity demand data.
    For the case study, we'll generate a synthetic hourly profile for a year.
    Actual implementation would load from NREL ResStock/ComStock or other sources.
    """
    # Placeholder: Generate a synthetic hourly profile for a year
    # Total hours in a year = 365 * 24 = 8760
    # Average hourly demand = (ANNUAL_ELECTRICITY_DEMAND_GWH * 1e6 kWh/GWh) / 8760 hours
    average_hourly_demand_kw = (ANNUAL_ELECTRICITY_DEMAND_GWH * 1e6) / 8760
    
    # Create a simple sinusoidal pattern with some randomness
    hours = np.arange(8760)
    # Normalize to 0-1, then scale to create variation around the average
    demand_pattern = (0.8 + 0.4 * np.sin(2 * np.pi * hours / 8760) + 
                      0.2 * np.sin(2 * np.pi * hours / (24*7)) + 
                      0.1 * np.random.randn(8760))
    
    synthetic_demand_kw = average_hourly_demand_kw * demand_pattern
    synthetic_demand_kw[synthetic_demand_kw < 0] = 0 # Ensure no negative demand
    
    # Create a DatetimeIndex
    timestamps = pd.date_range(start='2023-01-01', periods=8760, freq='h')
    load_profile_df = pd.DataFrame({'electricity_demand_kw': synthetic_demand_kw}, index=timestamps)
    
    # Save processed data (optional, could be done by a DVC pipeline)
    # load_profile_df.to_csv(processed_data_path)
    print("Generated synthetic electricity demand profile.")
    return load_profile_df

def load_heat_demand(raw_data_path="data/raw/heat_demand.csv", processed_data_path="data/processed/heat_demand_processed.csv"):
    """
    Loads and preprocesses heat demand data.
    Similar to electricity, generating a synthetic profile for now.
    """
    average_hourly_heat_demand_kwth = (ANNUAL_HEAT_DEMAND_GWH * 1e6) / 8760
    hours = np.arange(8760)
    # Heat demand typically higher in winter, lower in summer
    # More pronounced seasonal pattern for heat
    heat_demand_pattern = (0.7 + 0.6 * np.cos(2 * np.pi * (hours - 8760//2) / 8760) + # Peaking in winter
                           0.2 * np.sin(2 * np.pi * hours / (24*7)) + # Weekly variation
                           0.1 * np.random.randn(8760)) # Daily noise
                           
    synthetic_heat_demand_kwth = average_hourly_heat_demand_kwth * heat_demand_pattern
    synthetic_heat_demand_kwth[synthetic_heat_demand_kwth < 0] = 0

    timestamps = pd.date_range(start='2023-01-01', periods=8760, freq='h')
    heat_profile_df = pd.DataFrame({'heat_demand_kwth': synthetic_heat_demand_kwth}, index=timestamps)
    
    # heat_profile_df.to_csv(processed_data_path)
    print("Generated synthetic heat demand profile.")
    return heat_profile_df

def load_pv_generation_factor(raw_data_path="data/raw/pv_gis_data.csv", processed_data_path="data/processed/pv_generation_factor_processed.csv"):
    """
    Loads and preprocesses PV generation factor data (capacity factor per kWp).
    Generating a synthetic profile. Actual would use NSRDB, PVGIS etc.
    """
    hours = np.arange(8760)
    # Simple daily solar pattern, stronger in summer
    # Max capacity factor around 0.6-0.8 on a sunny day at noon
    daily_pattern = np.maximum(0, np.sin(2 * np.pi * (hours % 24 - 6) / 24)) # Sun up from 6am to 6pm
    seasonal_scaling = 0.6 + 0.4 * np.cos(2 * np.pi * (hours - 8760//2 + 8760//4) / 8760) # Peaking in summer
    
    pv_factor = daily_pattern * seasonal_scaling * 0.8 # Max factor of 0.8
    pv_factor[pv_factor < 0.01] = 0 # Clip small values
    pv_factor += 0.02 * np.random.rand(8760) # Add some noise

    timestamps = pd.date_range(start='2023-01-01', periods=8760, freq='h')
    pv_generation_factor_series = pd.Series(pv_factor, index=timestamps)
    
    # pv_generation_factor_series.to_csv(processed_data_path)
    print("Generated synthetic PV generation factor series.")
    return pv_generation_factor_series

def get_chp_parameters():
    return {
        'capacity_kw': CHP_CAPACITY_KW,
        'heat_capacity_kwth': CHP_HEAT_CAPACITY_KWTH,
        'gas_consumption_m3_per_kwh_e': CHP_GAS_CONSUMPTION_M3_PER_KWH_E,
        'co2_emission_ton_per_m3_gas': CHP_CO2_EMISSION_TON_PER_M3_GAS,
        'electricity_to_heat_ratio': CHP_CAPACITY_KW / CHP_HEAT_CAPACITY_KWTH if CHP_HEAT_CAPACITY_KWTH > 0 else 0 
        # Add other CHP params: ramp rates, min uptime/downtime, startup costs etc. if needed for detailed model
    }

def get_bess_parameters():
    return {
        'capacity_kwh': BESS_CAPACITY_KWH,
        'power_kw': BESS_POWER_KW,
        'charge_eff': BESS_CHARGE_EFF,
        'discharge_eff': BESS_DISCHARGE_EFF,
        'initial_soc_kwh': BESS_CAPACITY_KWH * 0.5 # Assume 50% initial SOC
        # Add other BESS params: min/max SOC, cycle life degradation model if needed
    }

def get_market_parameters(baseline_carbon_price=None):
    """
    Returns market parameters.
    Uses the provided baseline_carbon_price if given, otherwise defaults to the global constant.
    """
    effective_carbon_price = baseline_carbon_price if baseline_carbon_price is not None else CARBON_PRICE_BASELINE
    return {
        'tou_tariffs_cny_per_kwh': TOU_TARIFFS,
        'pv_feed_in_tariff_cny_per_kwh': PV_FEED_IN_TARIFF,
        'natural_gas_price_cny_per_m3': NATURAL_GAS_PRICE,
        'carbon_price_cny_per_ton': effective_carbon_price # Use the effective carbon price
    }

def get_financial_option_specs():
    """
    Generates a list of financial option product configurations.
    Each product is defined by its type (Call/Put), strike price, and maturity.
    Example: {'type': 'Call', 'strike_cny_ton': 180, 'maturity_months': 1, 'name': 'Call_180_1M'}
    """
    option_specs = [
        # 1-Month Maturity Options
        {"option_type": "Call", "strike_price": 280, "time_to_maturity_years": 1/12, "name": "Call_280_1M"},
        {"option_type": "Call", "strike_price": 300, "time_to_maturity_years": 1/12, "name": "Call_300_1M"},
        {"option_type": "Call", "strike_price": 330, "time_to_maturity_years": 1/12, "name": "Call_330_1M"},
        # 3-Month Maturity Options
        {"option_type": "Call", "strike_price": 280, "time_to_maturity_years": 3/12, "name": "Call_280_3M"},
        {"option_type": "Call", "strike_price": 300, "time_to_maturity_years": 3/12, "name": "Call_300_3M"},
        {"option_type": "Call", "strike_price": 330, "time_to_maturity_years": 3/12, "name": "Call_330_3M"},
    ]
    return option_specs

def get_roa_ccs_project_parameters():
    # Parameters for the CCS addition Real Option Analysis
    return {
        'investment_cost_cny': CCS_INVESTMENT_COST,
        'capture_efficiency': CCS_CAPTURE_EFFICIENCY,
        'opex_increase_per_kwh_chp_cny': CCS_OPEX_INCREASE_PER_KWH_CHP,
        'project_lifetime_years': CCS_PROJECT_LIFETIME_YEARS,
        'max_deferral_years': CCS_MAX_DEFERRAL_YEARS,
        'chp_annual_generation_kwh_assumed': CHP_CAPACITY_KW * 6000, 
        'chp_co2_emission_ton_per_m3_gas': CHP_CO2_EMISSION_TON_PER_M3_GAS,
        'chp_gas_consumption_m3_per_kwh_e': CHP_GAS_CONSUMPTION_M3_PER_KWH_E,
        'risk_free_rate': RISK_FREE_RATE,
        # ROA might use its own assumptions or take from general sim params
        'carbon_price_gbm_drift': CARBON_PRICE_GBM_DRIFT, 
        'carbon_price_gbm_volatility': CARBON_PRICE_VOLATILITY_ANNUAL, 
        'carbon_price_initial_cny_per_ton': CARBON_PRICE_BASELINE
    }

def get_simulation_parameters():
    """Returns a dictionary of general simulation parameters."""
    return {
        'pv_capacity_kw': PV_CAPACITY_KWP, # Corrected key to be consistent with run_case_study
        'risk_free_rate': RISK_FREE_RATE,
        'des_optimization_horizon_hours': DES_OPTIMIZATION_HORIZON_HOURS,
        'num_carbon_price_scenarios': NUM_CARBON_PRICE_SCENARIOS,
        'carbon_scenario_horizon_days': CARBON_SCENARIO_HORIZON_DAYS,
        'carbon_price_gbm_drift': CARBON_PRICE_GBM_DRIFT,
        'carbon_price_model_type': CARBON_PRICE_MODEL_TYPE,
        'roa_lattice_steps': ROA_LATTICE_STEPS,
        'risk_aversion_factor': RISK_AVERSION_FACTOR,
        'npv_hurdle_rate': NPV_HURDLE_RATE,
        'grid_max_import_export_kw': GRID_MAX_IMPORT_EXPORT_KW,
        # New CVaR related parameters for DES model objective
        'cvar_alpha_level': 0.90, # Changed from 0.95 to 0.90
        'lambda_cvar_weight': 10.0  # Keep lambda at 10.0
        # Note: 'carbon_price_gbm_volatility' is scenario-specific in run_case_study.py
        # 'operational_simulation_days' was an old key, replaced by des_optimization_horizon_hours
    }

# Example of how to use these functions:
if __name__ == '__main__':
    elec_demand = load_electricity_demand()
    print("Electricity Demand (first 5 rows):\n", elec_demand.head())
    
    heat_demand = load_heat_demand()
    print("Heat Demand (first 5 rows):\n", heat_demand.head())

    pv_factor = load_pv_generation_factor()
    print("PV Generation Factor (first 5 rows):\n", pv_factor.head())
    # pv_output_kw = pv_factor * PV_CAPACITY_KWP # PV_CAPACITY_KWP is global
    # Corrected to use value from sim_params if it was intended to be sourced from there
    sim_params_test = get_simulation_parameters()
    pv_output_kw = pv_factor * sim_params_test['pv_capacity_kw'] 
    print("PV Output kW (first 5 rows):\n", pv_output_kw.head())

    chp_params = get_chp_parameters()
    print("CHP Parameters:\n", chp_params)

    bess_params = get_bess_parameters()
    print("BESS Parameters:\n", bess_params)

    market_params = get_market_parameters()
    print("Market Parameters:\n", market_params)
    
    fin_options = get_financial_option_specs()
    print("Financial Option Specs:\n", fin_options)

    roa_params = get_roa_ccs_project_parameters()
    print("ROA CCS Project Parameters:\n", roa_params)

    sim_params = get_simulation_parameters() # Renamed here for clarity
    print("General Simulation Parameters:\n", sim_params)

    time_index = elec_demand.index
    tou_price_series = pd.Series(index=time_index, dtype=float)
    for hour in time_index:
        if (10 <= hour.hour < 15) or (18 <= hour.hour < 21):
            tou_price_series[hour] = market_params['tou_tariffs_cny_per_kwh']['peak']
        elif (23 <= hour.hour) or (hour.hour < 7):
            tou_price_series[hour] = market_params['tou_tariffs_cny_per_kwh']['valley']
        else:
            tou_price_series[hour] = market_params['tou_tariffs_cny_per_kwh']['flat']
    print("TOU Price Series (first 5 rows):\n", tou_price_series.head())