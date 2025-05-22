import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob # For finding the latest master run directory

# Assuming your project structure allows this import
# If ST4001 is the root and analyze_scenarios.py is at the root,
# and your source code is in ST4001/src, you might need to adjust Python path
# or run this script as a module if imports fail.
# For now, let's assume direct import works or will be adjusted.
try:
    from src.results_analyzer.analysis import (
        plot_operational_hedging_decisions_across_scenarios,
        plot_strategic_investment_decisions_across_scenarios
    )
except ImportError:
    print("Failed to import plotting functions. Ensure PYTHONPATH is set correctly or run as a module.")
    # As a fallback for isolated testing, one might redefine minimal versions here or skip plotting.
    # For this development, we assume the import will eventually work.
    def plot_operational_hedging_decisions_across_scenarios(*args, **kwargs):
        print("Plotting function (operational) not available due to import error.")
        return plt.figure()
    def plot_strategic_investment_decisions_across_scenarios(*args, **kwargs):
        print("Plotting function (strategic) not available due to import error.")
        return plt.figure()

def find_latest_master_run_dir(base_log_dir="experiment_logs") -> Path | None:
    """Finds the latest 'master_run_*' directory in the base log directory."""
    list_of_dirs = glob.glob(f"{base_log_dir}/master_run_*")
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    return Path(latest_dir)

def load_all_scenario_results(master_run_dir: Path) -> pd.DataFrame:
    """Loads results from all scenario subdirectories within a master run directory."""
    all_scenario_data = []

    # Iterate over subdirectories in the master_run_dir
    # These subdirectories are now named like 'cp150_vol15', 'cp200_vol20', etc.
    for scenario_subdir in master_run_dir.iterdir():
        if scenario_subdir.is_dir(): # Ensure it's a directory
            summary_file = scenario_subdir / "experiment_summary_data.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract relevant parameters and results
                    # The scenario parameters are now directly in 'scenario_parameters_applied'
                    params = data.get('scenario_parameters_applied', {})
                    op_decision = data.get('operational_hedging_decision', {})
                    strat_decision = data.get('strategic_ccs_investment_decision', {})

                    record = {
                        'scenario_name': scenario_subdir.name, # Use subdirectory name as scenario identifier
                        'baseline_carbon_price': params.get('baseline_carbon_price_for_des'),
                        'gbm_volatility': params.get('carbon_price_gbm_volatility'),
                        'operational_hedging_action': op_decision.get('decision'),
                        'operational_hedging_option_label': op_decision.get('details', {}).get('chosen_option_label'),
                        'strategic_ccs_investment_action': strat_decision.get('action')
                        # Add more fields as needed
                    }
                    all_scenario_data.append(record)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {summary_file}")
                except Exception as e:
                    print(f"Error processing file {summary_file}: {e}")
            else:
                print(f"Warning: {summary_file} not found in {scenario_subdir.name}")

    return pd.DataFrame(all_scenario_data)

if __name__ == "__main__":
    print("--- Starting Scenario Analysis ---")
    
    master_run_directory = find_latest_master_run_dir()
    
    if master_run_directory:
        print(f"Analyzing master run: {master_run_directory.name}")
        results_df = load_all_scenario_results(master_run_directory)
        
        if not results_df.empty:
            print("\nAggregated DataFrame head:")
            print(results_df.head())
            
            # Save aggregated results to CSV in the master run directory
            csv_output_path = master_run_directory / "aggregated_scenario_results.csv"
            results_df.to_csv(csv_output_path, index=False)
            print(f"\nAggregated results saved to: {csv_output_path}")

            # --- Generate and Save Plots in the master run directory ---
            try:
                # Plot for operational hedging vs. baseline carbon price
                fig_op_vs_price = plot_operational_hedging_decisions_across_scenarios(
                    results_df, 
                    x_param_name='baseline_carbon_price', 
                    decision_col_name='operational_hedging_action',
                    option_col_name='operational_hedging_option_label'
                )
                if fig_op_vs_price:
                    plot_op_price_path = master_run_directory / "operational_hedging_vs_carbon_price.png"
                    fig_op_vs_price.savefig(plot_op_price_path)
                    plt.close(fig_op_vs_price)
                    print(f"Plot saved: {plot_op_price_path}")

                # Plot for operational hedging vs. GBM volatility
                fig_op_vs_vol = plot_operational_hedging_decisions_across_scenarios(
                    results_df, 
                    x_param_name='gbm_volatility',
                    decision_col_name='operational_hedging_action',
                    option_col_name='operational_hedging_option_label'
                )
                if fig_op_vs_vol:
                    plot_op_vol_path = master_run_directory / "operational_hedging_vs_gbm_volatility.png"
                    fig_op_vs_vol.savefig(plot_op_vol_path)
                    plt.close(fig_op_vs_vol)
                    print(f"Plot saved: {plot_op_vol_path}")

                # Plot for strategic investment vs. baseline carbon price
                fig_strat_vs_price = plot_strategic_investment_decisions_across_scenarios(
                    results_df, 
                    x_param_name='baseline_carbon_price',
                    decision_col_name='strategic_ccs_investment_action'
                )
                if fig_strat_vs_price:
                    plot_strat_price_path = master_run_directory / "strategic_investment_vs_carbon_price.png"
                    fig_strat_vs_price.savefig(plot_strat_price_path)
                    plt.close(fig_strat_vs_price)
                    print(f"Plot saved: {plot_strat_price_path}")

                # Plot for strategic investment vs. GBM volatility
                fig_strat_vs_vol = plot_strategic_investment_decisions_across_scenarios(
                    results_df, 
                    x_param_name='gbm_volatility',
                    decision_col_name='strategic_ccs_investment_action'
                )
                if fig_strat_vs_vol:
                    plot_strat_vol_path = master_run_directory / "strategic_investment_vs_gbm_volatility.png"
                    fig_strat_vs_vol.savefig(plot_strat_vol_path)
                    plt.close(fig_strat_vs_vol)
                    print(f"Plot saved: {plot_strat_vol_path}")
                    
            except NameError as e:
                print(f"Plotting functions not available. Skipping plotting. Error: {e}")
            except Exception as e:
                print(f"An error occurred during plotting: {e}")
        else:
            print("No scenario data found to analyze or DataFrame is empty.")
    else:
        print("No 'master_run_*' directories found in 'experiment_logs'.")

    print("\n--- Scenario Analysis Complete ---") 