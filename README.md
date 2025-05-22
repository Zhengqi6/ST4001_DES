# Low-Carbon Distributed Energy System Optimization with Carbon Options and Real Option Analysis

## 1. Project Overview

This project implements a comprehensive simulation framework to analyze the optimization of low-carbon Distributed Energy Systems (DES). It specifically focuses on incorporating carbon pricing mechanisms, financial carbon options for operational hedging, and Real Option Analysis (ROA) for strategic investments, such as Carbon Capture and Storage (CCS).

The primary goal is to translate the theoretical framework and implementation strategies удовольствия in `ST4001.md` into a functional software system. This system facilitates research into the strategic role of carbon-aware decision-making in DES planning and operation.

For a detailed project design, theoretical background, and methodology, please refer to **[ST4001.md](ST4001.md)**.
For a roadmap of potential future enhancements, see **[FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md)**.

## 2. Core Functionality

The system is designed with a modular architecture, with key functionalities including:

*   **DES Optimization (`src/des_optimizer`)**: Models and optimizes the physical operation of a DES (e.g., a campus microgrid with PV, CHP, BESS) to meet energy demands at minimum cost, considering carbon emissions.
*   **Carbon Price Modeling (`src/carbon_pricer`)**: Generates carbon price scenarios using models like Geometric Brownian Motion (GBM) and GARCH to simulate market uncertainties.
*   **Financial Carbon Option Pricing (`src/financial_option_pricer`)**: Prices European financial carbon options (e.g., based on EUA futures) using the Black-Scholes model, based on the generated carbon price scenarios.
*   **Real Option Analysis (`src/real_option_analyzer`)**: Evaluates the value of managerial flexibility in strategic DES investments (e.g., deferring CCS installation) using binomial lattice models.
*   **Integrated Decision Logic (`src/decision_controller`)**: Implements logic for:
    *   Operational hedging decisions using financial carbon options (based on CVaR).
    *   Strategic investment decisions based on ROA results.
*   **Scenario Simulation & Analysis**:
    *   `run_case_study.py`: Orchestrates end-to-end simulation runs for single or multiple scenarios, varying parameters like baseline carbon price and volatility.
    *   `analyze_scenarios.py`: Aggregates results from multiple scenario runs, generates comparative plots, and saves summary data.
*   **Experiment Logging (`src/utils/experiment_logger.py`)**: Manages logging for each experiment run, creating a unique directory for outputs (logs, data, plots).

## 3. Technical Stack

*   **Primary Language**: Python 3.9+
*   **Core Libraries**:
    *   Optimization: Pyomo
    *   Data Handling: Pandas, NumPy
    *   Time Series/Financial Modeling: Statsmodels (for Arch - GARCH models)
    *   Plotting: Matplotlib
*   **Solver**: CBC (used by default for DES optimization via Pyomo)

A detailed list of dependencies can be found in `requirements.txt`.

## 4. Current Data Sources and Parameters

It is important to note that the current version of this project primarily relies on:

*   **Hardcoded parameters and constants**: Defined at the beginning of `src/utils/data_preparation.py`. These include technical specifications for DES components (PV, CHP, BESS), base market prices (electricity, gas), baseline carbon price, and parameters for CCS ROA.
*   **Synthetically generated time-series data**:
    *   Electricity and heat load profiles are synthetically generated within `src/utils/data_preparation.py` to mimic typical hourly patterns.
    *   PV generation factor is also synthetically generated.
*   These parameters and synthetic data are used by the functions in `src/utils/data_preparation.py` (e.g., `get_simulation_parameters()`, `load_electricity_demand()`) which are then called by `run_case_study.py`.

This approach ensures reproducibility for the current set of experiments. Future enhancements aim to connect to real-world, dynamic data sources (see `FUTURE_ENHANCEMENTS.md`).

## 5. How to Run

### Prerequisites

1.  Ensure Python 3.9+ is installed.
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure a MILP solver compatible with Pyomo is installed and accessible (e.g., CBC, GLPK). CBC is often bundled with Pyomo or easily installable.

### Running a Multi-Scenario Case Study

The main script to execute is `run_case_study.py`. It is configured to run a set of scenarios by default (e.g., varying baseline carbon prices and volatilities).

```bash
python run_case_study.py
```

This will:
1.  Create a main experiment directory in `experiment_logs/` named `master_run_YYYYMMDD_HHMMSS/`.
2.  Inside this main directory, it will create subdirectories for each scenario (e.g., `cp150_vol15/`, `cp200_vol20/`).
3.  Each scenario subdirectory will contain:
    *   `experiment_log.txt`: Detailed logs of the scenario run.
    *   `experiment_summary.txt`: A human-readable summary of key inputs and results.
    *   `experiment_summary_data.json`: A JSON file with detailed scenario results, including decision logic outputs.
    *   `des_operational_results.csv`: CSV file with DES dispatch details.
    *   `carbon_price_scenarios.csv`: CSV file with generated carbon price paths for that scenario.
    *   `des_dispatch_plot.png`: Plot of the DES operational dispatch.
    *   `carbon_price_scenarios_plot.png`: Plot of the generated carbon price scenarios.

### Analyzing Results Across Scenarios

After running `run_case_study.py` for multiple scenarios, you can use `analyze_scenarios.py` to aggregate results and generate comparative plots.

```bash
python analyze_scenarios.py
```

This script will:
1.  Automatically find the latest `master_run_...` directory in `experiment_logs/`.
2.  Load `experiment_summary_data.json` from each scenario subdirectory within that master run.
3.  Aggregate these summaries into a Pandas DataFrame.
4.  Save the aggregated data to `aggregated_scenario_results.csv` and `aggregated_scenario_results.xlsx` within the master run directory.
5.  Generate and save plots visualizing how decisions (e.g., operational hedging, strategic investment) vary across different scenario parameters (e.g., `operational_hedging_vs_baseline_carbon_price.png`). These plots will also be saved in the master run directory.

## 6. Project Structure

\`\`\`
ST4001/
├── experiment_logs/ # Output directory for all simulation runs
│   └── master_run_YYYYMMDD_HHMMSS/
│       ├── scenario_name_1/
│       │   ├── experiment_summary_data.json
│       │   ├── des_operational_results.csv
│       │   └── ... (other logs, plots, csvs for scenario 1)
│       ├── scenario_name_2/
│       │   └── ...
│       ├── aggregated_scenario_results.csv
│       ├── aggregated_scenario_results.xlsx
│       └── ... (cross-scenario analysis plots)
├── src/                    # Source code
│   ├── des_optimizer/      # DES physical modeling and optimization
│   ├── carbon_pricer/      # Carbon price scenario generation
│   ├── financial_option_pricer/ # Financial carbon option pricing
│   ├── real_option_analyzer/ # Real Option Analysis (ROA)
│   ├── decision_controller/  # Integrated decision logic
│   ├── results_analyzer/   # Output processing, analysis, plotting
│   └── utils/              # Utility functions (data prep, logger)
├── SeniorThesis (Copy)/    # LaTeX template for thesis (User managed)
├── analyze_scenarios.py    # Script to analyze multi-scenario results
├── run_case_study.py       # Main script to run case studies
├── FUTURE_ENHANCEMENTS.md  # Roadmap for future development
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── ST4001.md               # Detailed project proposal and design document
\`\`\`

## 7. Contribution and Development

Please refer to `ST4001.md` for the detailed design and `FUTURE_ENHANCEMENTS.md` for potential areas of contribution.
 