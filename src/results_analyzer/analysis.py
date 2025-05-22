import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os

def plot_des_dispatch(results_df, output_path, file_prefix='des_dispatch'):
    """Plots key electricity dispatch, heat dispatch, and BESS SOC from DES results and saves the plot."""
    if results_df.empty:
        print("Cannot plot DES dispatch: Results DataFrame is empty.")
        # Return an empty figure or handle as preferred if not saving
        # For consistency, we might still want to save a blank placeholder or log this.
        fig = plt.figure()
        plt.text(0.5, 0.5, "No DES data to plot", ha='center', va='center')
        if output_path:
            filepath = Path(output_path) / f"{file_prefix}_summary.png"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath)
            print(f"Saved placeholder DES dispatch plot to {filepath}")
        plt.close(fig)
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    title_suffix = file_prefix # Use file_prefix as part of the title
    
    # Electricity Dispatch
    elec_cols = ['elec_demand_kw', 'pv_gen_kw', 'chp_gen_e_kw', 'grid_import_e_kw', 'bess_dis_p_kw', 'grid_export_e_kw']
    # Ensure correct column names as per typical DES output
    # Example: 'P_pv_gen', 'P_chp_e', 'P_grid_buy', 'P_grid_sell', 'P_bs_dis'
    # We'll stick to the provided names for now, but this might need alignment with des_model.py output
    
    plot_cols_elec = [col for col in elec_cols if col in results_df.columns]
    if plot_cols_elec:
        results_df[plot_cols_elec].plot(ax=axes[0])
    axes[0].set_title(f'Electricity Dispatch - {title_suffix}')
    axes[0].set_ylabel('kW')
    axes[0].legend(loc='upper right')

    # Heat Dispatch
    heat_cols = ['heat_demand_kwth', 'chp_gen_h_kwth']
    plot_cols_heat = [col for col in heat_cols if col in results_df.columns]
    if plot_cols_heat and not results_df[plot_cols_heat].empty:
        results_df[plot_cols_heat].plot(ax=axes[1])
        axes[1].legend(loc='upper right')
    else:
        axes[1].text(0.5, 0.5, 'No heat dispatch data', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
    axes[1].set_title(f'Heat Dispatch - {title_suffix}')
    axes[1].set_ylabel('kWth')
    
    # BESS SOC
    if 'bess_soc_kwh' in results_df.columns and not results_df['bess_soc_kwh'].empty:
        results_df[['bess_soc_kwh']].plot(ax=axes[2])
        axes[2].legend(loc='upper right')
    else:
        axes[2].text(0.5, 0.5, 'No BESS SOC data', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
    axes[2].set_title(f'BESS State of Charge - {title_suffix}')
    axes[2].set_ylabel('kWh')
    
    plt.xlabel("Time (Hours)") # Assuming index is hours or time steps
    plt.tight_layout()
    
    if output_path:
        filepath = Path(output_path) / f"{file_prefix}_summary.png"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)
        print(f"Saved DES dispatch plot to {filepath}")
    plt.close(fig) # Close the figure to free memory

def plot_carbon_price_scenarios(scenarios_df, output_path, file_prefix='carbon_scenarios'):
    """Plots carbon price scenarios and saves the plot."""
    if scenarios_df is None or scenarios_df.empty:
        print("Cannot plot carbon price scenarios: DataFrame is empty or None.")
        fig = plt.figure()
        plt.text(0.5, 0.5, "No carbon price scenarios to plot", ha='center', va='center')
        if output_path:
            filepath = Path(output_path) / f"{file_prefix}.png"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath)
            print(f"Saved placeholder carbon price scenarios plot to {filepath}")
        plt.close(fig)
        return
    
    fig = plt.figure(figsize=(12, 7))
    for col in scenarios_df.columns:
        plt.plot(scenarios_df.index, scenarios_df[col], lw=1.0, alpha=0.7, label=col if len(scenarios_df.columns) <=10 else None)
    
    # If many scenarios, only label the first one to avoid clutter, or don't label any by default
    if len(scenarios_df.columns) > 10 and not scenarios_df.empty:
         # plt.plot(scenarios_df.index, scenarios_df.iloc[:,0], lw=1.0, alpha=0.7, label=scenarios_df.columns[0]) 
         pass # Avoid legend for too many lines, or customize
    elif not scenarios_df.empty:
        plt.legend(loc='upper left', bbox_to_anchor=(1,1))

    title = file_prefix.replace('_', ' ').title()
    plt.title(f'{title}')
    plt.xlabel('Date/Time')
    plt.ylabel('Carbon Price (CNY/ton)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output_path:
        filepath = Path(output_path) / f"{file_prefix}.png"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)
        print(f"Saved carbon price scenarios plot to {filepath}")
    plt.close(fig)

def display_option_prices(priced_options, output_path, file_prefix='option_prices'):
    """Displays financial option prices and saves them to a text file and a simple plot."""
    if not priced_options:
        print("No option prices to display.")
        if output_path:
            filepath_txt = Path(output_path) / f"{file_prefix}_summary.txt"
            os.makedirs(os.path.dirname(filepath_txt), exist_ok=True)
            with open(filepath_txt, 'w') as f:
                f.write("No option prices to display.\n")
            print(f"Saved empty option prices summary to {filepath_txt}")
        return

    header = f"{'Label':<20} {'Strike (CNY)':<15} {'Maturity (Y)':<15} {'Type':<8} {'Price (CNY/ton)':<15}\n"
    separator = "-" * 70 + "\n"
    content = "Financial Option Prices:\n" + separator + header + separator
    for opt in priced_options:
        content += f"{opt.get('label', 'N/A'):<20} {opt.get('strike_price', 0):<15.2f} {opt.get('time_to_maturity_years', 0):<15.4f} {opt.get('option_type', 'N/A'):<8} {opt.get('price', 0):<15.6f}\n"
    content += separator
    print(content)

    if output_path:
        filepath_txt = Path(output_path) / f"{file_prefix}_summary.txt"
        os.makedirs(os.path.dirname(filepath_txt), exist_ok=True)
        with open(filepath_txt, 'w') as f:
            f.write(content)
        print(f"Saved option prices summary to {filepath_txt}")

        # Create a simple bar plot of option prices by label
        labels = [opt.get('label', f'Opt{i}') for i, opt in enumerate(priced_options)]
        prices = [opt.get('price', 0) for opt in priced_options]
        if labels and prices:
            fig, ax = plt.subplots(figsize=(10, len(labels) * 0.5 + 2))
            ax.barh(labels, prices, color='skyblue')
            ax.set_xlabel('Price (CNY/ton)')
            ax.set_title(f'Financial Option Prices - {file_prefix}')
            plt.yticks(fontsize=8)
            plt.tight_layout()
            filepath_png = Path(output_path) / f"{file_prefix}_plot.png"
            plt.savefig(filepath_png)
            print(f"Saved option prices plot to {filepath_png}")
            plt.close(fig)

def display_roa_results(traditional_npv, roa_option_value, output_path, file_prefix='roa_results'):
    """Displays real option analysis results and saves them to a text file and a simple plot."""
    content = "Real Option Analysis Results (CCS Investment):\n"
    content += "-" * 60 + "\n"
    content += f"Traditional NPV: {traditional_npv:,.2f} CNY\n"
    content += f"ROA Value (Option Value to Defer/Invest): {roa_option_value:,.2f} CNY\n"
    expanded_npv = traditional_npv + roa_option_value # More accurately, ROA value is the total Expanded NPV if K is NPV without option.
                                                  # If K is investment cost, then E_NPV = NPV_passive + Option_Premium
                                                  # Here, roa_option_value IS the option premium/value.
    content += f"Expanded NPV (Traditional NPV + Option Value): {expanded_npv:,.2f} CNY\n" 
    content += "-" * 60 + "\n"
    print(content)

    if output_path:
        filepath_txt = Path(output_path) / f"{file_prefix}_summary.txt"
        os.makedirs(os.path.dirname(filepath_txt), exist_ok=True)
        with open(filepath_txt, 'w') as f:
            f.write(content)
        print(f"Saved ROA results summary to {filepath_txt}")

        # Create a simple bar plot comparing NPV and ROA value
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = ['Traditional NPV', 'ROA Value (Option Premium)', 'Expanded NPV']
        values = [traditional_npv, roa_option_value, expanded_npv]
        ax.bar(metrics, values, color=['lightcoral', 'lightskyblue', 'lightgreen'])
        ax.set_ylabel('Value (CNY)')
        ax.set_title(f'ROA vs. Traditional NPV - {file_prefix}')
        # Add text labels for values
        for i, v in enumerate(values):
            ax.text(i, v + (0.05 * max(abs(np.array(values)))) if v >= 0 else v - (0.15 * max(abs(np.array(values)))), 
                    f"{v:,.0f}", ha='center', va='bottom' if v >=0 else 'top')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        filepath_png = Path(output_path) / f"{file_prefix}_plot.png"
        plt.savefig(filepath_png)
        print(f"Saved ROA results plot to {filepath_png}")
        plt.close(fig)

def plot_roa_lattice(underlying_lattice, option_lattice, output_path, file_prefix='roa_lattice'):
    """Plots the binomial lattice for the underlying asset and the option values, and saves the plot."""
    if underlying_lattice is None or option_lattice is None:
        print("Cannot plot ROA lattice: Lattice data is missing.")
        # Optionally save a placeholder if needed
        return

    N_steps = len(underlying_lattice) -1 
    if N_steps <= 0:
        print("Cannot plot ROA lattice: Invalid number of steps.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    # Plot Underlying Asset Lattice
    ax_underlying = axes[0]
    for i in range(N_steps + 1):
        for j in range(i + 1):
            ax_underlying.text(i, j - i/2, f"{underlying_lattice[i][j]:.2f}", ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, pad=0.1))
            if i < N_steps:
                ax_underlying.plot([i, i+1], [j - i/2, j - (i+1)/2], 'b-', alpha=0.5) # Line to lower node
                ax_underlying.plot([i, i+1], [j - i/2, (j+1) - (i+1)/2], 'b-', alpha=0.5) # Line to upper node
    ax_underlying.set_title(f'Underlying Asset Value Lattice - {file_prefix}')
    ax_underlying.set_xlabel('Time Steps')
    ax_underlying.set_ylabel('Node Value (Normalized)')
    ax_underlying.set_xticks(range(N_steps + 1))
    ax_underlying.set_yticks([]) # Y-axis is illustrative of structure, not strict value scale

    # Plot Option Value Lattice
    ax_option = axes[1]
    for i in range(N_steps + 1):
        for j in range(i + 1):
            ax_option.text(i, j - i/2, f"{option_lattice[i][j]:.2f}", ha='center', va='center', fontsize=8, bbox=dict(facecolor='lightgreen', alpha=0.5, pad=0.1))
            if i < N_steps:
                ax_option.plot([i, i+1], [j - i/2, j - (i+1)/2], 'r-', alpha=0.5)
                ax_option.plot([i, i+1], [j - i/2, (j+1) - (i+1)/2], 'r-', alpha=0.5)
    ax_option.set_title(f'Option Value Lattice - {file_prefix}')
    ax_option.set_xlabel('Time Steps')
    ax_option.set_ylabel('Node Value (Normalized)')
    ax_option.set_xticks(range(N_steps + 1))
    ax_option.set_yticks([])

    plt.tight_layout()
    if output_path:
        filepath = Path(output_path) / f"{file_prefix}.png"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)
        print(f"Saved ROA lattice plot to {filepath}")
    plt.close(fig)

def plot_operational_hedging_decisions_across_scenarios(scenario_summary_df, 
                                                        x_param_name='baseline_carbon_price_for_des',
                                                        decision_col_name='operational_hedging_decision_action',
                                                        option_col_name='operational_hedging_option_label',
                                                        output_path=None, file_prefix='op_hedge_decisions'):
    """Plots operational hedging decisions across different scenarios and saves the plot."""
    if scenario_summary_df.empty:
        print("Cannot plot operational hedging decisions: DataFrame is empty.")
        # Create and save a placeholder plot
        fig = plt.figure()
        plt.text(0.5,0.5, "No data for operational hedging decisions", ha='center', va='center')
        if output_path:
            filepath = Path(output_path) / f"{file_prefix}.png"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath)
            print(f"Saved placeholder operational hedging plot to {filepath}")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    scenario_summary_df[x_param_name] = pd.to_numeric(scenario_summary_df[x_param_name], errors='coerce')
    sorted_df = scenario_summary_df.sort_values(by=x_param_name).copy() # Use .copy() to avoid SettingWithCopyWarning

    def create_decision_label(row):
        decision = row[decision_col_name]
        option = row[option_col_name]
        if decision == 'HedgeWithFinancialOption' and pd.notna(option) and option != '':
            return f"Hedge ({option})"
        return decision
    
    if decision_col_name in sorted_df.columns and option_col_name in sorted_df.columns:
        # Ensure the apply is on the DataFrame, not a slice if possible
        sorted_df.loc[:, 'full_decision_label'] = sorted_df.apply(create_decision_label, axis=1)
        y_plot_col = 'full_decision_label'
    elif decision_col_name in sorted_df.columns:
        y_plot_col = decision_col_name
    else:
        ax.text(0.5, 0.5, f"Decision column '{decision_col_name}' not found.", ha='center', va='center', transform=ax.transAxes)
        if output_path:
            filepath = Path(output_path) / f"{file_prefix}.png"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath) # Save the plot with the error message
            print(f"Saved operational hedging plot (with error) to {filepath}")
        plt.close(fig)
        return

    unique_decisions = sorted_df[y_plot_col].unique()
    decision_to_int = {decision: i for i, decision in enumerate(unique_decisions)}
    
    if not sorted_df[y_plot_col].map(decision_to_int).empty:
        ax.scatter(sorted_df[x_param_name], sorted_df[y_plot_col].map(decision_to_int), 
                   marker='o', s=100, alpha=0.7)
        ax.set_yticks(list(decision_to_int.values()))
        ax.set_yticklabels(list(decision_to_int.keys()))
    else:
        ax.text(0.5, 0.5, "No decision data to plot after mapping.", ha='center', va='center', transform=ax.transAxes)

    x_label = x_param_name.replace('_', ' ').capitalize()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Operational Hedging Decision")
    ax.set_title(f"Operational Hedging Decision vs. {x_label} - {file_prefix}")
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if output_path:
        filepath = Path(output_path) / f"{file_prefix}.png"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)
        print(f"Saved operational hedging decisions plot to {filepath}")
    plt.close(fig)

def plot_strategic_investment_decisions_across_scenarios(scenario_summary_df, 
                                                           x_param_name='baseline_carbon_price_for_des',
                                                           decision_col_name='strategic_ccs_investment_action',
                                                           output_path=None, file_prefix='strat_invest_decisions'):
    """Plots strategic CCS investment decisions across different scenarios and saves the plot."""
    if scenario_summary_df.empty:
        print("Cannot plot strategic investment decisions: DataFrame is empty.")
        fig = plt.figure()
        plt.text(0.5,0.5, "No data for strategic investment decisions", ha='center', va='center')
        if output_path:
            filepath = Path(output_path) / f"{file_prefix}.png"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath)
            print(f"Saved placeholder strategic investment plot to {filepath}")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    scenario_summary_df[x_param_name] = pd.to_numeric(scenario_summary_df[x_param_name], errors='coerce')
    sorted_df = scenario_summary_df.sort_values(by=x_param_name)
    
    if decision_col_name not in sorted_df.columns:
        ax.text(0.5, 0.5, f"Decision column '{decision_col_name}' not found.", ha='center', va='center', transform=ax.transAxes)
        if output_path:
            filepath = Path(output_path) / f"{file_prefix}.png"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath)
            print(f"Saved strategic investment plot (with error) to {filepath}")
        plt.close(fig)
        return

    unique_decisions = sorted_df[decision_col_name].unique()
    decision_to_int = {decision: i for i, decision in enumerate(unique_decisions)}
    
    if not sorted_df[decision_col_name].map(decision_to_int).empty:
        ax.scatter(sorted_df[x_param_name], sorted_df[decision_col_name].map(decision_to_int), 
                   marker='s', s=100, alpha=0.7, label='Strategic Decision')
        ax.set_yticks(list(decision_to_int.values()))
        ax.set_yticklabels(list(decision_to_int.keys()))
    else:
         ax.text(0.5, 0.5, "No decision data to plot after mapping.", ha='center', va='center', transform=ax.transAxes)

    x_label = x_param_name.replace('_', ' ').capitalize()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Strategic CCS Investment Decision")
    ax.set_title(f"Strategic CCS Investment Decision vs. {x_label} - {file_prefix}")
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    # ax.legend() # Only one series, legend might be redundant
    plt.tight_layout()
    if output_path:
        filepath = Path(output_path) / f"{file_prefix}.png"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)
        print(f"Saved strategic investment decisions plot to {filepath}")
    plt.close(fig)

def plot_stochastic_des_results(results_summary: dict, output_dir: Path, time_horizon_dt: pd.DatetimeIndex | None = None):
    """
    Plots key results from the stochastic DES optimization.

    Args:
        results_summary (dict): The dictionary returned by extract_stochastic_des_results.
        output_dir (Path): Directory to save the plots.
        time_horizon_dt (pd.DatetimeIndex | None): DatetimeIndex for x-axis of dispatch plots.
                                                If None, integer index will be used.
    """
    if not results_summary:
        print("Cannot plot stochastic DES results: Results summary is empty.")
        return

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Optimal Option Purchase Strategy
    optimal_options = results_summary.get('optimal_option_purchase', {})
    if optimal_options:
        fig_opts, ax_opts = plt.subplots(figsize=(10, 6))
        option_names = list(optimal_options.keys())
        option_quantities = list(optimal_options.values())
        # Filter out options with zero quantity for cleaner plot
        plot_data = {name: quant for name, quant in zip(option_names, option_quantities) if quant > 1e-3}
        if plot_data:
            ax_opts.bar(plot_data.keys(), plot_data.values(), color='skyblue')
            ax_opts.set_title('Optimal Carbon Option Purchase Strategy')
            ax_opts.set_ylabel('Number of Contracts')
            ax_opts.set_xlabel('Option Type')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / "optimal_option_strategy.png")
            print(f"Saved optimal option strategy plot to {output_dir / 'optimal_option_strategy.png'}")
        else:
            print("No options were purchased in the optimal strategy.")
        plt.close(fig_opts)

    # 2. Expected Total Cost Breakdown
    expected_total_cost = results_summary.get('expected_total_cost', 0)
    option_purchase_cost = results_summary.get('option_purchase_cost', 0)
        
    scenario_details = results_summary.get('scenario_details', {})
    num_scenarios = len(scenario_details)
    avg_grid_buy_cny = 0
    avg_grid_sell_cny = 0 # This is revenue, will be subtracted or shown as negative
    avg_chp_fuel_cny = 0
    avg_chp_carbon_cny = 0
    avg_total_scenario_cost_after_options = 0

    if num_scenarios > 0:
        avg_grid_buy_cny = sum(s.get('grid_buy_cost_cny',0) for s in scenario_details.values()) / num_scenarios
        avg_grid_sell_cny = sum(s.get('grid_sell_revenue_cny',0) for s in scenario_details.values()) / num_scenarios
        avg_chp_fuel_cny = sum(s.get('chp_fuel_cost_cny',0) for s in scenario_details.values()) / num_scenarios
        avg_chp_carbon_cny = sum(s.get('chp_carbon_cost_cny',0) for s in scenario_details.values()) / num_scenarios
        avg_total_scenario_cost_after_options = sum(s.get('total_operational_cost_after_options_cny',0) for s in scenario_details.values()) / num_scenarios

    fig_costs, ax_costs = plt.subplots(figsize=(12, 7))
    cost_components = {
        'Option Purchase Cost': option_purchase_cost,
        'Avg Grid Electricity Purchase Cost': avg_grid_buy_cny,
        'Avg CHP Fuel Cost': avg_chp_fuel_cny,
        'Avg CHP Carbon Cost (Post-Hedge)': avg_chp_carbon_cny,
        'Avg Grid Electricity Sell Revenue': -avg_grid_sell_cny # Show as negative bar (revenue)
    }
    
    # Filter out zero/NaN components for clarity
    plot_cost_components = {k: v for k, v in cost_components.items() if pd.notna(v) and abs(v) > 1e-3}

    if plot_cost_components:
        colors = ['red' if v < 0 else 'dodgerblue' for v in plot_cost_components.values()]
        bars = ax_costs.bar(plot_cost_components.keys(), plot_cost_components.values(), color=colors)
        ax_costs.set_ylabel('Cost (CNY)')
        ax_costs.set_title('Expected Total Cost Breakdown (Post Hedging)')
        plt.xticks(rotation=30, ha='right')
        # Add data labels
        for bar in bars:
            yval = bar.get_height()
            ax_costs.text(bar.get_x() + bar.get_width()/2.0, yval + np.sign(yval)*0.01*abs(yval), 
                        f'{yval:,.0f}', ha='center', va='bottom' if yval >=0 else 'top')
        
        # Display total expected cost
        ax_costs.text(0.95, 0.95, f'Total Expected Cost: {expected_total_cost:,.0f} CNY', 
                    transform=ax_costs.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / "expected_total_cost_breakdown.png")
        print(f"Saved expected total cost breakdown plot to {output_dir / 'expected_total_cost_breakdown.png'}")
    else:
        print("No cost components to plot for stochastic DES results.")
    plt.close(fig_costs)

    # 3. CVaR of Total Costs (if available)
    cvar_total_costs = results_summary.get('cvar_total_costs')
    if cvar_total_costs is not None:
        # This is usually a single value, can be part of a summary text file or a simple bar plot if compared to other metrics
        print(f"CVaR of Total Costs: {cvar_total_costs:,.2f} CNY")
        # Could add to a summary TXT file if one is created for stochastic results

    # 4. Dispatch variables (e.g., expected CHP generation, Grid Buy/Sell under different scenarios)
    # This requires more detailed scenario-specific dispatch data, which might be too verbose for a summary plot.
    # If 'dispatch_summary_per_scenario' is part of results_summary, it could be processed.
    # Example: Plotting expected electricity generation by source across scenarios
    if 'expected_dispatch' in results_summary and isinstance(results_summary['expected_dispatch'], pd.DataFrame):
        expected_dispatch_df = results_summary['expected_dispatch']
        if not expected_dispatch_df.empty:
            fig_disp, ax_disp = plt.subplots(figsize=(14, 7))
            # Select relevant columns for plotting (e.g., generation sources, grid interaction)
            plot_cols_disp = [col for col in ['P_chp_e', 'P_pv_gen', 'P_grid_buy', 'P_bs_dis', 'P_load'] if col in expected_dispatch_df.columns]
            if plot_cols_disp:
                if time_horizon_dt is not None and len(time_horizon_dt) == len(expected_dispatch_df):
                    expected_dispatch_df.index = time_horizon_dt
                expected_dispatch_df[plot_cols_disp].plot(ax=ax_disp, kind='line') # or kind='area' for stacked area plot
                ax_disp.set_title('Expected System Dispatch (Averaged Across Scenarios)')
                ax_disp.set_ylabel('Power (kW or kWh average)')
                ax_disp.set_xlabel('Time Step')
                ax_disp.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(output_dir / "expected_system_dispatch.png")
                print(f"Saved expected system dispatch plot to {output_dir / 'expected_system_dispatch.png'}")
            else:
                print("No relevant columns for expected dispatch plot.")
            plt.close(fig_disp)
        else:
            print("Expected dispatch DataFrame is empty.")
    print(f"Stochastic DES results plotting completed in {output_dir}")

if __name__ == '__main__':
    idx = pd.date_range('2023-01-01', periods=5, freq='h')
    dummy_des_res = pd.DataFrame({
        'elec_demand_kw': [100,110,105,120,90],
        'pv_gen_kw': [0,20,50,30,0],
        'chp_gen_e_kw': [80,70,60,90,80],
        'grid_import_e_kw': [20,20,0,0,10],
        'bess_dis_p_kw': [0,0,0,0,0],
        'grid_export_e_kw': [0,0,5,0,0],
        'heat_demand_kwth': [50,55,52,60,45],
        'chp_gen_h_kwth': [50,55,52,60,45],
        'bess_soc_kwh': [150,140,130,120,110]
    }, index=idx)
    
    fig_des = plot_des_dispatch(dummy_des_res, title_suffix='(Test Data)')
    if fig_des and fig_des.get_axes(): plt.show()
    else: print("DES plot not generated.")

    dummy_scenarios = pd.DataFrame({
        f's{i}': np.linspace(200, 200 + np.random.randint(-50,50), 20) for i in range(12) # Test more than 10 scenarios
    }, index=pd.date_range('2023-01-01', periods=20, freq='D'))
    
    fig_scenarios = plot_carbon_price_scenarios(dummy_scenarios)
    if fig_scenarios and fig_scenarios.get_axes(): plt.show()
    else: print("Scenario plot not generated.")

    dummy_options = [
        {'label': 'Call_200_1M', 'strike_price': 200, 'time_to_maturity_years': 1/12, 'option_type': 'call', 'price': 5.25},
        {'label': 'Call_220_3M', 'strike_price': 220, 'time_to_maturity_years': 3/12, 'option_type': 'call', 'price': 3.10}
    ]
    display_option_prices(dummy_options)
    
    dummy_roa = {'option_value': 65000.75, 'traditional_npv': 10000.50} # Test without threshold
    display_roa_results(dummy_roa['traditional_npv'], dummy_roa['option_value'], '')
    dummy_roa_with_thresh = {'option_value': 75000, 'traditional_npv': 15000, 'investment_threshold_carbon_price': 245.0}
    display_roa_results(dummy_roa_with_thresh['traditional_npv'], dummy_roa_with_thresh['option_value'], '') 

    dummy_scenario_summary = pd.DataFrame({
        'baseline_carbon_price_for_des': [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310],
        'operational_hedging_decision_action': ['NoHedge', 'HedgeWithFinancialOption', 'NoHedge', 'HedgeWithFinancialOption', 'NoHedge', 'HedgeWithFinancialOption', 'NoHedge', 'HedgeWithFinancialOption', 'NoHedge', 'HedgeWithFinancialOption', 'NoHedge', 'HedgeWithFinancialOption'],
        'operational_hedging_option_label': ['Call_200_1M', 'Call_220_3M', 'Call_200_1M', 'Call_220_3M', 'Call_200_1M', 'Call_220_3M', 'Call_200_1M', 'Call_220_3M', 'Call_200_1M', 'Call_220_3M', 'Call_200_1M', 'Call_220_3M']
    })
    
    fig_operational_hedging = plot_operational_hedging_decisions_across_scenarios(dummy_scenario_summary)
    if fig_operational_hedging and fig_operational_hedging.get_axes(): plt.show()
    else: print("Operational Hedging Decision plot not generated.") 

    dummy_strategic_summary = pd.DataFrame({
        'baseline_carbon_price_for_des': [180, 200, 220, 240, 260, 280, 300],
        'strategic_ccs_investment_action': ['DEFER', 'DEFER', 'DEFER_BUT_MONITOR', 'INVEST_NOW', 'INVEST_NOW', 'INVEST_NOW', 'INVEST_NOW']
    })
    fig_strategic_investment = plot_strategic_investment_decisions_across_scenarios(dummy_strategic_summary)
    if fig_strategic_investment and fig_strategic_investment.get_axes(): plt.show()
    else: print("Strategic Investment Decision plot not generated.") 