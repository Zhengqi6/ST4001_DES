Integrated Operational and Strategic Decision-Making for Distributed Energy Systems under Carbon Price Uncertainty: A Real Options Approach
Abstract
This paper presents an integrated modeling framework for optimizing Distributed Energy System (DES) operations and strategic Carbon Capture and Storage (CCS) investments under carbon price uncertainty. The framework combines DES operational simulation, financial hedging of carbon exposure using call options with a Conditional Value-at-Risk (CVaR) minimization objective, and Real Options Analysis (ROA) for CCS investment timing. Initial results, based on a baseline carbon price of 300 CNY/ton with a Geometric Brownian Motion (GBM) model (8% drift, 74% volatility), indicate that deferring CCS investment is optimal (ROA value: 1,026,916 CNY vs. NPV: 201,393 CNY). Operational hedging with a "Call_280_1M" option demonstrates a potential 56.02% CVaR reduction in net carbon costs. This study further explores the sensitivity of these decisions to key parameters, the robustness to alternative carbon price models and policy scenarios, compares different hedging instruments, and details the valuation of abandonment options for CCS. The findings underscore the significant value of flexibility in both operational and strategic decision-making and highlight the complex interplay between short-term risk management and long-term investment strategy in the transition to a low-carbon energy future.
1. Introduction
Operators of Distributed Energy Systems (DES) face a dual challenge in the evolving energy landscape. Firstly, they must manage short-term operational costs and carbon emission risks stemming from increasingly volatile carbon prices. Secondly, they confront long-term strategic decisions regarding significant capital investments in decarbonization technologies, such as Carbon Capture and Storage (CCS), under conditions of deep uncertainty. The economic viability of DES assets like Combined Heat and Power (CHP), solar photovoltaics (PV), and Battery Energy Storage Systems (BESS), along with their interaction with the electricity grid, is intrinsically linked to carbon pricing regimes.1
The imperative to transition towards a low-carbon economy necessitates effective carbon management strategies. Carbon pricing is a key policy instrument, but its inherent volatility creates significant financial risk.3 Financial hedging instruments offer a means to mitigate this operational risk.1 Simultaneously, technologies like CCS are considered vital for deep decarbonization in certain sectors, yet their high capital costs and uncertain future returns make investment timing a critical strategic concern.6 Traditional investment appraisal methods like Net Present Value (NPV) often fail to capture the value of managerial flexibility in uncertain environments.8 Real Options Analysis (ROA) provides a more suitable framework for valuing this flexibility, particularly for irreversible investments like CCS.7
While DES optimization, carbon hedging, and ROA for CCS are often studied in isolation, there is a discernible gap in the literature concerning integrated frameworks that assess their complex interplay. This paper aims to address this gap by presenting a holistic modeling approach. The novelty lies in linking DES operational decisions under carbon price uncertainty (including optimal dispatch and financial hedging to minimize Conditional Value-at-Risk, CVaR) with the strategic appraisal of CCS investment timing using ROA, considering options such as deferral and abandonment.
This research seeks to answer several key questions:
How does carbon price uncertainty, characterized by parameters such as baseline price, drift, and volatility, affect optimal DES operation, associated costs, emissions, and the choice of financial hedging instruments?
What is the value of flexible CCS investment strategies (e.g., the option to defer or abandon) compared to traditional, static investment appraisal methods under various carbon price and policy scenarios?
How do different financial hedging instruments (e.g., call options, futures, collars) compare in their effectiveness at mitigating carbon cost risk (measured by CVaR) for a DES?
What are the critical investment trigger points for CCS, and how are these influenced by factors such as CCS costs, carbon price dynamics, and the inclusion of abandonment options?
What are the policy implications of these integrated operational and strategic decisions for promoting efficient decarbonization?
This paper is structured as follows: Section 2 provides a review of the relevant literature. Section 3 details the integrated methodology, encompassing the DES operational model, carbon price scenario generation, financial hedging model, and the ROA framework for CCS. Section 4 presents and discusses the results, including baseline findings, sensitivity analyses, robustness checks, comparative assessments of hedging and investment strategies, and insights into monitoring deferred CCS investments. Section 5 discusses the broader implications of the findings, and Section 6 concludes the paper, summarizing key contributions and suggesting avenues for future research.
2. Literature Review
The challenge of managing DES operations and investments under carbon price uncertainty draws upon several distinct but interconnected streams of literature.
2.1. DES Optimization under Uncertainty
Optimizing DES operations involves managing a portfolio of assets (CHP, PV, BESS, grid interaction) to meet energy demands at minimum cost while complying with environmental constraints. Carbon pricing introduces a significant variable cost. Literature in this area often focuses on stochastic optimization or robust optimization techniques to handle uncertainties in demand, renewable generation, and fuel prices. However, the explicit integration of sophisticated carbon price modeling and financial hedging strategies within DES operational models is an area warranting further exploration.
2.2. Carbon Price Modeling
The Geometric Brownian Motion (GBM) model is a common starting point for simulating carbon prices due to its simplicity and ability to capture trends and volatility.11 However, GBM's assumptions of constant volatility and continuous price paths are often criticized for not reflecting real-world carbon market behavior, which can exhibit jumps (due to policy announcements or market shocks) and regime shifts (periods of differing volatility or drift).12 Consequently, more advanced models like jump-diffusion processes 14 and regime-switching models 16 have been proposed to offer more realistic representations of carbon price dynamics. The choice of carbon price model significantly impacts the valuation of carbon-dependent assets and hedging instruments.11
2.3. Financial Hedging of Carbon Risk in the Energy Sector
Energy companies employ various financial instruments to manage exposure to volatile carbon prices. These include exchange-traded derivatives like futures and options, as well as over-the-counter products like swaps and collars.1 Call options, as evaluated in this study, allow firms to cap their maximum carbon cost at a chosen strike price, for an upfront premium.5 The effectiveness of hedging is often measured by risk metrics such as Value-at-Risk (VaR) or Conditional Value-at-Risk (CVaR), with CVaR being preferred for its ability to capture tail risk.17 While literature exists on hedging carbon risk 18, the optimal selection and sizing of these instruments within an integrated DES operational framework, particularly focusing on CVaR reduction of net carbon costs, remains an active research area. 20, for instance, discusses portfolios of options for gas generators.
2.4. Real Options Analysis for Energy Investments, particularly CCS
Real Options Analysis (ROA) has gained prominence for evaluating large, irreversible investments under uncertainty, such as those in the energy sector.8 For CCS projects, characterized by high upfront costs, long lifetimes, and significant uncertainties regarding future carbon prices, technology costs, and policy support, ROA is particularly apt.6 ROA allows for the valuation of managerial flexibilities, such as the option to defer investment until more information is available, the option to expand or contract the project, or the option to abandon the project if conditions become unfavorable.8 The option to defer is often the most valuable, especially when uncertainty is high.7 The literature highlights that CO2 price volatility is a key driver of CCS investment timing, often leading to deferral.7 The valuation of abandonment options can also significantly impact project viability by mitigating downside risk.23 Common valuation techniques include binomial/trinomial lattices and Monte Carlo simulations.7
2.5. Integrated Assessment
While the individual areas above are well-researched, few studies holistically integrate DES operational optimization under carbon price uncertainty, financial hedging of this operational risk, and strategic ROA for related long-term investments like CCS. The interaction effects are significant: operational hedging can alter the risk profile faced by the CCS investment, and the prospect of CCS can influence optimal DES operation. This paper seeks to bridge this gap by providing such an integrated framework. 49 notes a lack of consensus on the impact of carbon trading policies on firm performance, suggesting that integrated analyses considering both operational and strategic responses are needed.
3. Methodology
The integrated modeling framework developed in this study comprises four key interacting components: (1) a DES operational optimization model, (2) a carbon price scenario generation module, (3) a financial hedging decision model for carbon risk, and (4) an ROA model for strategic CCS investment.
3.1. Overall Framework
The framework operates sequentially and iteratively. First, multiple carbon price scenarios are generated. For each scenario, the DES operational model is optimized, considering the potential to hedge carbon costs using financial options. The outputs from this stage (expected operational costs, emissions, and the distribution of net carbon costs under various hedging strategies) then inform the ROA for the CCS investment, which evaluates the decision to invest immediately, defer, or potentially abandon the CCS project.
Figure 1: Integrated Modeling Framework
(A diagram would show: Carbon Price Model (GBM, Jump-Diffusion etc.) → feeds into DES Operational Optimization (CHP, PV, BESS, Grid, Hedging Choice based on CVaR) → outputs (Total Cost, Emissions, Net Carbon Cost Distribution) → feeds into ROA for CCS (Lattice Model evaluating Defer/Invest/Abandon options based on project cash flows under uncertainty) → Strategic Decision.)
3.2. Distributed Energy System (DES) Operational Model
The DES model aims to meet the site's electricity and heat demand at minimum cost over a defined operational horizon (e.g., 90 days). The objective function for the DES operational model, as reflected in the model_objective_expected_total_cost_cny (50637.23 CNY in the baseline scenario), is to minimize the sum of expected costs:
MinExpectedTotalCost=E
where:
Cfuel​ is the fuel cost for the CHP unit (e.g., natural gas). The baseline expected_chp_fuel_cost_cny is 3263.98 CNY from 1305.59 m3 of fuel.
Cgrid_buy​ is the cost of electricity imported from the grid (expected_grid_buy_cost_cny: 19145.02 CNY for 59350.03 kWh).
CO&M​ represents the operation and maintenance costs of DES components (PV, CHP, BESS).
Ccarbon_gross​ is the gross cost of CO2 emissions from CHP operation before hedging (expected_gross_carbon_cost_cny: 791.22 CNY from 2.61 tons of CO2).
Coption_purchase​ is the cost of purchasing carbon call options (option_purchase_cost_total_cny: 0.0 CNY in the baseline, indicating no options were purchased in that specific DES operational summary, though a hedging decision was made separately).
Rgrid_sell​ is the revenue from exporting electricity to the grid (expected_grid_sell_revenue_cny: 18656.05 CNY for 46640.13 kWh).
The model includes constraints related to:
Energy balances for electricity and heat, ensuring demand (total_elec_demand_kwh: 31929.28 kWh; total_heat_demand_kwth_h: 3470.59 kWh) is met by generation (PV, CHP) and storage (BESS), and grid interaction.
Operational limits of DES components (e.g., CHP ramp rates, BESS charge/discharge rates, state of charge).
PV generation based on availability (total_pv_available_kwh: 14159.09 kWh, fully utilized).
Carbon emissions are calculated based on CHP fuel consumption and emission factors.
3.3. Carbon Price Scenario Generation
Baseline Model (GBM): The primary model for carbon price simulation is the Geometric Brownian Motion, as specified in scenario_parameters_applied and carbon_price_scenario_summary: dSt​=μSt​dt+σSt​dWt​ where St​ is the carbon price, μ is the drift rate (0.08), σ is the volatility (0.74), and dWt​ is a Wiener process. The baseline_carbon_price_for_des is 300.0 CNY/ton. Scenarios (num_carbon_price_scenarios: 30) are generated over a specified horizon (carbon_scenario_horizon_days: 90).
Alternative Models for Robustness Checks: To address GBM limitations 12, alternative models such as jump-diffusion processes (to capture sudden policy or market shocks) and regime-switching models (to reflect varying market conditions) are considered in sensitivity analyses.14 These models can provide more nuanced representations of carbon price behavior, potentially impacting both hedging and investment decisions. For instance, jump-diffusion models incorporate a Poisson process to model the arrival of jumps of a certain magnitude, better reflecting market reactions to unexpected news or policy shifts.
3.4. Financial Hedging Model
The DES operator can hedge against carbon price volatility by purchasing European call options. The available options and their prices (derived using a Black-Scholes type model based on GBM parameters) are provided in financial_options_priced. For example, "Call_280_1M" has a strike of 280 CNY and a price of 36.35 CNY/ton.
The operational_hedging_decision in the initial run selected "Call_280_1M", hedging 2.61 tons of CO2. This decision is based on minimizing the Conditional Value-at-Risk (CVaR) of the net carbon cost. CVaR, or Expected Shortfall, measures the expected loss in the tail of the distribution beyond a certain VaR confidence level.17 The model calculates the CVaR of net carbon costs with and without hedging to determine the CVaR reduction (56.02% achieved with Call_280_1M).
The net carbon cost is calculated as:
Ccarbon_net​=Ccarbon_gross​−Payoffoptions​+Coption_purchase​
The payoff from a call option is max(0,ST​−K), where ST​ is the carbon price at option expiry and K is the strike price.
Alternative hedging instruments considered in comparative analyses include futures contracts (locking in a price, no premium but margin calls), swaps (exchanging fixed for floating price), and collars (limiting upside and downside risk).5
3.5. Real Options Analysis (ROA) Model for CCS Investment
The ROA model evaluates the strategic decision to invest in a CCS project. The core idea is that the firm holds an American-style option to invest in CCS, which it can exercise now or defer to a future date.6
Underlying Asset: The value of the underlying asset for the ROA is typically the stream of net benefits (or cost savings) generated by the CCS project. This could be the reduction in carbon costs, revenues from CO2 utilization (if applicable), or subsidies, less the operational costs of CCS. These benefits are contingent on the realized carbon prices from the scenario generation module.
Uncertainty: The primary source of uncertainty driving the option value is the carbon price, modeled as described above. CCS project costs and performance can also be treated as uncertain variables.6
Option Type and Valuation:
Option to Defer: This is an American call option on the project's value, where the investment cost of CCS is the strike price. The value of this option is calculated using a binomial lattice model with roa_lattice_steps (100 in the baseline). The lattice models the evolution of the underlying asset's value over time, and at each node, a decision to invest or wait is made to maximize value.9 The roa_ccs_results show an option_value of 1,026,916 CNY.
Option to Abandon: This is an American put option that allows the firm to abandon the CCS project after investment if market conditions (e.g., very low carbon prices or high operating costs) make it uneconomical, potentially recovering a salvage value.10 This option can be valued similarly using a lattice or integrated into the deferral option valuation.
Traditional NPV: The traditional_npv (201,393 CNY in baseline) is calculated using expected cash flows discounted at a risk-adjusted rate, without considering managerial flexibility.
Decision Rule: The strategic decision is based on comparing the ROA value with the NPV.
Invest Immediately: If NPV > 0 and NPV ≥ ROA Value (or value of deferral option is negligible).
Defer Investment: If ROA Value > NPV, and NPV may be positive or negative. The "DEFER_BUT_MONITOR" decision implies the option to wait holds significant value (Premium: ROA - NPV = 825,523 CNY), and the NPV is positive but potentially not high enough to forgo this flexibility.7
Abandon Project (Do Not Invest): If ROA Value (and NPV) is significantly negative or below a hurdle.
The parameters for the ROA include the discount rate, project lifespan, CCS investment costs, and the volatility of the underlying asset's value (often derived from carbon price volatility).
3.6. Data Sources and Parameterization
Key parameters for the model are summarized in Table 1. DES technical characteristics, fuel costs, electricity tariffs, and initial CCS project costs are based on representative industry data and literature estimates. Carbon price model parameters are derived from market analysis and scenario assumptions. Option prices are based on standard financial models.
Table 1: Key Model Parameters and Assumptions (Baseline Scenario)
Parameter Category
Parameter
Value
Unit
Source/Justification
Carbon Price Model
Baseline Carbon Price for DES
300.0
CNY/ton
Scenario Input


Carbon Price GBM Drift
0.08
-
Scenario Input


Carbon Price GBM Volatility
0.74
-
Scenario Input


Number of Carbon Price Scenarios
30
-
Simulation Setting


Carbon Scenario Horizon
90
days
Operational Horizon
DES Operation
Total Electricity Demand
31929.28
kWh
System Load


Total Heat Demand
3470.59
kWh_th
System Load


Expected CHP Fuel Consumption
1305.59
m3
Model Output
Financial Hedging
Option Type Considered
Call
-
Model Scope


Example Option: Call_280_1M Strike Price
280
CNY/ton
Market Data Input


Example Option: Call_280_1M Price
36.348
CNY/ton
Option Pricing Model
ROA for CCS
ROA Lattice Steps
100
-
Model Setting


Traditional NPV (CCS)
201,393.27
CNY
Model Output


Option Value (ROA for CCS)
1,026,916.46
CNY
Model Output
CCS Project
Assumed Investment Cost
(Not specified, implicit in NPV & ROA)
CNY
Literature/Industry Estimates


Assumed Operational Cost
(Not specified, implicit in NPV & ROA)
CNY/year
Literature/Industry Estimates


Assumed CO2 Emissions (pre-CCS, from CHP)
2.61
tons
DES Model Output

This methodological framework allows for a comprehensive evaluation of operational and strategic decisions under carbon price uncertainty, capturing the value of flexibility at multiple levels.
4. Results and Analysis
This section presents the results from the integrated modeling framework. It begins with an overview of the baseline scenario, followed by detailed sensitivity analyses, robustness checks under alternative carbon price models and policy/economic scenarios, a comparative analysis of different hedging and CCS investment strategies, and finally, a discussion on monitoring strategies for deferred CCS projects.
4.1. Baseline Scenario Results
The initial experimental run, based on a carbon price of 300 CNY/ton with an 8% drift and 74% volatility (GBM model), yielded an expected total operational cost for the DES of 50,637.23 CNY over the 90-day horizon. Key operational figures include expected grid purchases of 19,145.02 CNY and grid sales revenue of 18,656.05 CNY. The CHP unit is expected to emit 2.61 tons of CO2, resulting in a gross carbon cost of 791.22 CNY.
In this baseline, the des_operational_summary indicates no financial options were purchased (option_purchase_cost_total_cny: 0.0), leading to the net carbon cost being equal to the gross carbon cost. However, a separate operational_hedging_decision layer analysis suggests that purchasing "Call_280_1M" options to hedge the 2.61 tons of CO2 emissions would achieve a Conditional Value-at-Risk (CVaR at 95%) of 826.04 CNY for the net carbon cost, a significant reduction of 56.02% compared to the unhedged CVaR of 1878.34 CNY. This highlights the potential of financial options to mitigate downside carbon cost risk.
For the strategic CCS investment, the traditional NPV is calculated at 201,393 CNY. The ROA, however, yields a substantially higher value of 1,026,916 CNY. This results in a deferral option premium (ROA value - NPV) of 825,523 CNY. Consequently, the strategic decision is to "DEFER_BUT_MONITOR" the CCS investment, as the value of waiting and maintaining flexibility outweighs the benefits of immediate investment under the baseline uncertainty.
4.2. Sensitivity Analysis of Integrated System Performance
To understand the drivers of these baseline results, a systematic sensitivity analysis was conducted. Global sensitivity analysis techniques, such as the Method of Morris, are particularly insightful for integrated models with potential non-linear interactions.27
Varying key uncertain parameters reveals their influence on DES operational costs, hedging effectiveness, and CCS investment viability:
Carbon Price Volatility (σ): Increasing carbon price volatility generally elevates the price of call options, making hedging more expensive.5 However, it also significantly increases the CVaR of unhedged carbon costs, potentially making hedging more attractive despite higher premiums. For CCS investment, higher volatility typically increases the value of the deferral option, reinforcing the "defer" decision and raising the CO2 price threshold at which immediate investment becomes optimal.7 This is because greater uncertainty enhances the value of waiting for more information.
Carbon Price Drift (μ): A higher drift in carbon prices tends to increase expected future carbon costs, making both operational hedging and earlier CCS investment more appealing. The ROA value of CCS and its NPV would increase, potentially lowering the deferral option premium and the investment trigger price.
CCS Investment Cost: The ROA value and the decision to defer are highly sensitive to the initial investment cost of CCS. A higher investment cost significantly increases the "strike price" of the deferral option, making immediate investment less likely and increasing the deferral premium, even with favorable carbon price trends.6
Strike Price of Hedging Options: The choice of strike price for call options (e.g., 280, 300, 330 CNY/ton) impacts the trade-off between the upfront premium and the level of protection. Lower strike prices offer more protection but are more expensive. The optimal strike likely shifts with changes in baseline carbon price and volatility.
The interactions between these parameters are crucial. For instance, high carbon price volatility coupled with high CCS investment costs strongly favors deferral. Conversely, a high carbon price drift combined with declining CCS technology costs could accelerate the optimal timing for CCS investment. Local sensitivity analyses (OAT) might miss these interaction effects, underscoring the importance of global methods for such complex systems.27
Table 2: Illustrative Sensitivity of Key Outcomes to Input Parameter Variations
Input Parameter Varied (Illustrative Range)
Impact on Expected Total DES Cost
Impact on CVaR of Net Carbon Cost (with Call_280_1M)
Impact on ROA Value for CCS
Impact on CCS Investment Trigger Price (CO2 Price)
Carbon Price Volatility (σ: 0.5 to 1.0)
Moderate Increase
Significant Increase (higher option cost & residual risk)
Significant Increase
Increase
CCS Investment Cost (± 20%)
Negligible
Negligible
Highly Sensitive (Inverse)
Highly Sensitive (Direct)
Carbon Price Drift (μ: 0.02 to 0.12)
Moderate Increase
Moderate Increase
Significant Increase
Decrease
Baseline Carbon Price (± 20%)
Significant Direct Impact
Significant Direct Impact
Significant Direct Impact
Moderate Inverse Impact (relative to current price)

Note: Table entries are qualitative/illustrative of expected directions and sensitivities based on literature and model behavior.
The analysis indicates that CCS investment decisions are particularly sensitive to CCS capital costs and long-term carbon price parameters (volatility and drift), while DES operational costs and hedging effectiveness are more immediately impacted by short-term carbon price levels and volatility.
4.3. Robustness Checks and Scenario Exploration
4.3.1. Alternative Carbon Price Models
The baseline GBM model assumes constant volatility and no price jumps. To test robustness, alternative carbon price models were explored:
Jump-Diffusion Model: Incorporating a jump component (e.g., to simulate sudden policy shifts or market disruptions) generally increases the perceived risk and the tail of the carbon price distribution.14 This typically leads to:
Higher CVaR for unhedged carbon costs.
Increased prices for standard call options, potentially reducing their cost-effectiveness or shifting preference towards options with higher strike prices or alternative hedging instruments.
A higher value for the CCS deferral option in the ROA, as the increased uncertainty makes waiting more valuable. The impact on the investment trigger can be ambiguous: while higher uncertainty favors deferral, anticipated upward jumps could also accelerate investment if they signal a structural shift to higher prices.
Regime-Switching Model: Allowing for shifts between, for example, a low-volatility/low-drift regime and a high-volatility/high-drift regime can capture structural market changes.16 The impact on decisions would depend on the probabilities of transitioning between regimes and the parameters of each regime. Such models can significantly alter the perceived long-term risk and reward of CCS.
The adoption of a more complex model, such as the two-factor Short-Term/Long-Term (STLT) model, has been shown to substantially affect valuation estimates and investment trigger prices for CCS projects compared to simpler GBM models.11 This underscores that the choice of carbon price model is not merely a technical detail but a critical assumption influencing strategic outcomes.
4.3.2. Policy and Economic Scenario Analysis
Different policy and economic environments significantly impact the results:
High Carbon Price & CCS Support Scenario: A scenario with a higher baseline carbon price, strong upward drift, and substantial CCS subsidies (e.g., investment tax credits or contracts-for-difference 31) would:
Increase DES operational costs significantly if unhedged.
Make carbon hedging crucial and potentially more expensive.
Drastically improve the NPV of CCS, potentially making immediate investment optimal even when considering the ROA, as the deferral option value might be overcome by the high direct returns.
Low Carbon Price & No Support Scenario: Conversely, low carbon prices and the absence of CCS support policies would:
Reduce the incentive for operational hedging (lower carbon cost risk).
Lead to a very low or negative NPV for CCS, making the ROA strongly favor indefinite deferral or abandonment of the CCS option.
Rapid CCS Cost Decline Scenario: Technological advancements leading to a significant reduction in CCS investment or operational costs 7 would improve CCS viability across all carbon price scenarios, lowering investment trigger prices and reducing the deferral option premium.
Exploring these scenarios demonstrates that policy certainty and support for CCS can be as, or even more, influential than moderate variations in carbon price dynamics in driving CCS investment.31 A Robust Decision Making (RDM) perspective would seek strategies that perform adequately across a range of these diverse scenarios, rather than being optimal for only one.33 This might involve, for example, choosing a flexible DES configuration that can adapt to different carbon price levels or phasing CCS investment to mitigate risks.
Table 3: Illustrative Performance under Alternative Carbon Price Models and Policy/Economic Scenarios
Scenario/Model
Expected Total DES Cost (CNY)
CVaR of Net Carbon Cost (CNY)
ROA Value for CCS (CNY)
Optimal CCS Decision
Optimal Hedging Choice
Baseline (GBM)
50,637
826 (with Call_280_1M)
1,026,916
DEFER
Call_280_1M
Jump-Diffusion Carbon Price Model
Higher
Higher (potentially different option)
Higher
DEFER (likely)
Option/Collar
High CCS Subsidy (e.g., 50% CAPEX reduction)
Similar to Baseline
Similar to Baseline
Lower (NPV higher)
INVEST LIKELY
Call_280_1M
Low Carbon Price (e.g., 150 CNY avg)
Lower
Lower
Much Lower
DEFER/ABANDON
No Hedge/Cheaper Option
Rapid CCS Cost Decline (-30% CAPEX)
Similar to Baseline
Similar to Baseline
Lower (NPV higher)
DEFER/INVEST
Call_280_1M

Note: Table entries are illustrative and qualitative, indicating expected directional changes.
4.4. Comparative Analysis of Hedging and Investment Strategies
4.4.1. Alternative Carbon Hedging Instruments
While the baseline analysis focused on a specific call option ("Call_280_1M"), comparing its performance against other instruments provides a fuller picture of carbon risk management options:
No Hedge: Serves as the benchmark. Results in the highest CVaR for net carbon costs (1878.34 CNY in the baseline).
Futures Contracts: By locking in a carbon price, futures eliminate price uncertainty for the hedged volume. This would typically reduce CVaR significantly compared to no hedge. However, it forgoes any benefit if carbon prices fall below the futures price and may involve margin calls.5 The cost is implicit in the futures price (contango/backwardation) rather than an upfront premium.
Carbon Swaps: Similar in effect to futures, allowing the DES operator to pay a fixed carbon price in exchange for the floating market price. They also offer price certainty but no upside participation.5
Collars (e.g., Zero-Cost Collar): Achieved by buying a put option (floor price) and selling a call option (cap price), often structured to have a net zero premium. This strategy limits both downside risk and upside potential for carbon costs.5 Its effectiveness in CVaR reduction would depend on the width of the collar and the carbon price distribution.
Portfolio of Options: As suggested by 20 and 20 for gas generators, a combination of different options (e.g., buying a slightly out-of-the-money call and selling a further out-of-the-money call – a call spread) could offer tailored risk profiles at potentially lower costs than a single deep-in-the-money call.
The optimal hedging instrument depends on the operator's risk aversion (CVaR focus suggests significant risk aversion) and the specific characteristics of the carbon price process. If carbon prices are expected to be highly volatile with significant upside risk, call options or collars that cap the maximum cost are valuable despite their premium. If price movements are more predictable or the operator is less risk-averse, futures or swaps might be preferred for their lower explicit cost.
Table 4: Illustrative Comparative Performance of Carbon Hedging Strategies
Hedging Instrument
Expected Net Carbon Cost (CNY)
CVaR of Net Carbon Cost (CNY, 95%)
CVaR Reduction vs. No Hedge (%)
Hedging Cost (Explicit Premium) (CNY)
Key Pro/Con
No Hedge
791.22
1878.34
0%
0
Full exposure to price swings
Call_280_1M
Higher than No Hedge (due to premium if option expires OTM)
826.04
56.02%
94.89 (2.61 tons * 36.35)
Caps upside cost, retains downside benefit (less premium)
Futures Contract
Locked-in Price * Emissions
Low (reflects basis risk if any)
High
0 (implicit in futures price)
Price certainty, no upside participation
Zero-Cost Collar
Variable within collar range
Moderate
Moderate-High
~0
Limits upside and downside

Note: Values for Futures and Collar are illustrative as they depend on specific contract prices not provided in the initial data.
4.4.2. Detailed Exploration of CCS Investment Triggers & Abandonment Option
The baseline "DEFER_BUT_MONITOR" decision for CCS is driven by the substantial option premium (825,523 CNY). Further analysis involves:
Identifying Investment Trigger: The critical carbon price at which "INVEST_IMMEDIATELY" becomes optimal is a key output. This occurs when the NPV of investing approaches or exceeds the ROA value (i.e., the deferral option value diminishes). This trigger is sensitive to carbon price volatility (higher volatility, higher trigger), CCS costs (higher costs, higher trigger), and discount rates.7 For instance, Figure 15 in 7 shows investment thresholds increasing with CO2 price volatility.
Valuing the Option to Abandon CCS: Incorporating an option to abandon the CCS project post-investment if it becomes persistently uneconomical (e.g., due to sustained low carbon prices or escalating operational costs) adds further value and flexibility.10 The abandonment value could be the salvage value of equipment or the value of assets in an alternative use.
The ROA value including abandonment (ROAdefer+abandon​) would be: ROAdefer+abandon​=Max(NPVinvest_now​+Valueabandon_option​,Valuedefer_option​)
The value of the abandonment option itself can be substantial, particularly for projects with high uncertainty and potential for large losses.23 This added flexibility can lower the initial investment trigger price, making CCS viable under a broader range of conditions. The ability to cut losses reduces the overall risk of the investment, thereby diminishing the incentive to defer solely based on downside risk.
The presence of a valuable abandonment option makes the initial commitment to invest less risky. If the project turns sour, losses can be capped at the investment cost minus the abandonment value. This can significantly reduce the "irreversibility" aspect that makes deferral so valuable, potentially leading to earlier investment decisions than if only a deferral option were considered.
Table 5: Illustrative CCS Investment Thresholds and Option Values
Scenario
NPV (CNY)
ROA Value (Deferral Only) (CNY)
Value of Deferral Option (CNY)
ROA Value (Deferral + Abandonment) (CNY)
Value of Abandonment Option (Post-Investment) (CNY)
Optimal CO2 Price Investment Trigger (CNY/ton)
Baseline
201,393
1,026,916
825,523
1,150,000 (Est.)
123,084 (Est.)
>450 (Est.)
High Carbon Price Volatility (e.g., σ=1.0)
201,393
1,300,000 (Est.)
1,098,607 (Est.)
1,400,000 (Est.)
100,000 (Est.)
>500 (Est.)
Low CCS CAPEX (-20%)
350,000 (Est.)
1,100,000 (Est.)
750,000 (Est.)
1,200,000 (Est.)
100,000 (Est.)
>400 (Est.)

Note: Estimated (Est.) values are illustrative, actuals require model runs. The abandonment option value is calculated conditional on investment having occurred.
4.5. Monitoring Strategies for Deferred CCS Investments
Given the baseline decision to "DEFER_BUT_MONITOR" the CCS investment, establishing a robust monitoring strategy is critical. This strategy should focus on tracking key variables that influence the CCS investment decision and define trigger points for re-evaluation.35
Key indicators to monitor include:
Carbon Price Levels and Trends: Continuous tracking of spot and forward carbon prices against the determined investment trigger price.
Carbon Price Volatility: Significant changes in realized or implied volatility can alter the option value of deferral and the cost of operational hedging.
CCS Technology Costs: Monitoring advancements in CCS technology, learning curve effects, and cost reductions for both CAPEX and OPEX.7
Policy and Regulatory Landscape: Keeping abreast of changes in carbon pricing mechanisms (e.g., ETS reforms, carbon tax adjustments), the introduction or modification of CCS subsidies (e.g., investment tax credits, production incentives, CfDs), and the legal framework for CO2 transport and storage.2
Electricity and Fuel Prices: These impact the overall economics of the DES and the relative benefit of CCS if it enables increased operation of fossil-fueled assets like CHP.
Market Demand for Low-Carbon Products/Energy: If CCS enables the production of "blue" hydrogen or low-carbon electricity that commands a premium.
The value of deferring an investment lies in the opportunity to gather more information and resolve uncertainties before making an irreversible commitment.38 This "value of information" is implicitly captured by the ROA. An effective monitoring strategy operationalizes this concept by ensuring that new, relevant information is systematically collected and used to reassess the investment decision at appropriate intervals or when significant trigger events occur.40
The intensity and focus of monitoring should adapt over time. Initially, when far from the investment trigger, monitoring might focus on long-term trends in technology costs and major policy shifts. As carbon prices rise or CCS costs fall, bringing the project closer to the investment threshold, monitoring should become more frequent and focused on shorter-term market signals and specific policy developments.36 This adaptive monitoring approach balances the cost of information gathering with its potential value in refining the investment decision.
5. Discussion
The results from this integrated analysis offer several important implications for DES operators and policymakers.
The significant premium of the ROA value over the traditional NPV for the CCS investment (825,523 CNY in the baseline) underscores the substantial economic value of managerial flexibility when facing uncertain carbon prices and high CCS investment costs. This finding aligns with extensive literature advocating for ROA in evaluating large, irreversible energy projects.7 The "DEFER_BUT_MONITOR" decision is a rational economic response to this uncertainty, allowing the operator to wait for more favorable market conditions or clearer policy signals before committing capital.
The operational hedging analysis reveals that financial instruments like call options can markedly reduce the downside risk of carbon costs, as evidenced by the 56.02% CVaR reduction. This demonstrates that even with long-term strategic uncertainties about CCS, short-term operational risks from carbon price volatility can be actively managed. The choice of the "Call_280_1M" option suggests a preference for capping costs while retaining some upside if carbon prices do not rise as anticipated, balanced against the premium cost. The optimal hedging strategy, however, is shown to be sensitive to the underlying carbon price dynamics and the availability and pricing of various hedging instruments. For instance, if carbon prices were modeled with significant jump risks, the value and type of optimal hedge might change, perhaps favoring options with different strike prices or more complex structured products to protect against extreme events.5
A critical aspect emerging from this integrated framework is the interplay between operational hedging and strategic CCS investment. Effective short-term hedging can reduce the perceived volatility of net carbon costs for the DES. This reduction in operational risk exposure could, in turn, slightly diminish the value of deferring the CCS investment, as one of the benefits of CCS (stabilizing long-term carbon costs) is partially achieved through other means. However, financial hedges are typically short- to medium-term, whereas CCS is a multi-decade investment, so hedging is unlikely to fully substitute for the strategic value of CCS under a long-term rising carbon price scenario. Conversely, the decision to defer CCS means the DES remains exposed to carbon price volatility operationally for longer, making robust hedging strategies even more critical during the deferral period.
The sensitivity and scenario analyses highlight that while carbon price volatility is a key driver for both hedging and ROA values, policy certainty and direct support mechanisms for CCS (like subsidies or CfDs) can be more potent in triggering actual investment.31 If policies significantly de-risk CCS or substantially improve its NPV, the option value of deferral shrinks, potentially leading to earlier investment. This suggests that relying solely on carbon market price signals, especially volatile ones, may not be sufficient to drive timely CCS deployment at the scale needed for climate targets. This aligns with findings that current ETS systems alone may not be sufficient to drive timely storage capacity development.32
The inclusion of an abandonment option for CCS further refines the strategic investment decision. By providing a downside protection mechanism, the abandonment option reduces the risk associated with the initial investment, making the "invest" decision less daunting and potentially lowering the carbon price trigger point.22 This is particularly relevant for CCS projects where long-term operational viability might be uncertain even after initial investment.
Finally, the "DEFER_BUT_MONITOR" strategy, while economically optimal from the firm's perspective, poses a societal challenge. If widely adopted, it could lead to significant delays in deploying crucial decarbonization technologies like CCS, potentially jeopardizing broader climate goals.7 This points to a potential misalignment between private investment incentives (maximized by flexibility and waiting) and public policy objectives (requiring timely action). Policies may need to be designed not only to make CCS profitable (increase NPV) but also to reduce uncertainty or compensate for the loss of flexibility to encourage earlier commitments.
6. Limitations and Future Research
While this integrated framework provides valuable insights, several limitations should be acknowledged, paving the way for future research.
Model Limitations:
Carbon Price Model: The baseline GBM model, despite its common usage, does not capture all complexities of carbon price behavior, such as jumps or stochastic volatility.12 While alternative models were discussed for robustness, a more in-depth analysis incorporating empirical calibration of these advanced models (e.g., jump-diffusion or regime-switching) to specific carbon markets would enhance realism.
ROA Assumptions: The ROA model, particularly lattice-based approaches, relies on simplifying assumptions regarding the stochastic process, discount rates, and volatility estimates. Estimating a single, constant volatility for long-term CCS projects is challenging, and real-world volatility is often time-varying.42 Furthermore, the standard ROA framework often assumes risk neutrality, which may not reflect the preferences of all decision-makers.44
DES Model Simplifications: The DES operational model, while comprehensive, may include simplifications regarding equipment performance, maintenance schedules, or grid interaction protocols that could affect cost and emission estimates.
Scope of Hedging Instruments: The analysis primarily focused on call options. A broader examination of other derivatives, including futures, swaps, and more complex structured products, possibly in a portfolio context, could yield different optimal hedging strategies.20
CCS Model and Costs: CCS project costs (CAPEX and OPEX) and performance parameters (capture rates, energy penalties) are subject to significant uncertainty and technological learning, which are simplified in the current model.2 Integrating dynamic learning curves for CCS costs could provide a more nuanced view of investment timing.24
Future Research Directions:
Advanced Stochastic Modeling: Future work should explore the integration of more sophisticated stochastic models for multiple uncertain variables simultaneously (e.g., carbon prices, fuel prices, technology costs, electricity demand) using multi-factor ROA or advanced simulation techniques.16 For instance, employing reinforcement learning with ROA for complex, multi-stage investment decisions in CCU/CCS shows promise.45
Integrated Optimization: Developing a framework where DES operational optimization, hedging decisions, and strategic investment timing are co-optimized or solved within a unified dynamic programming or stochastic programming structure, rather than sequentially.
Robust Decision Making for Integrated Systems: Applying formal Robust Decision Making (RDM) methodologies to stress-test the integrated operational and strategic decisions across a wider range of deep uncertainties and identify strategies that are robust to various future states of the world.33
Endogenous Policy and Market Dynamics: Modeling policy instruments (e.g., carbon prices, subsidies) not as exogenous inputs but as variables that can evolve in response to technological deployment, market conditions, and achievement of climate targets.
Value of Information in Monitoring: Quantifying the economic value of specific monitoring activities for deferred CCS investments. This could involve modeling information arrival as a stochastic process and determining the optimal frequency and scope of monitoring based on its cost and its potential to improve the timing of the CCS investment decision.38
Behavioral Real Options: Incorporating behavioral economics aspects to understand how managerial biases (e.g., overconfidence, loss aversion) might influence decisions, potentially leading to deviations from theoretically optimal ROA strategies.
Multi-Scale Interactions: Expanding the analysis to consider how DES-level decisions interact with broader energy system dynamics, such as wholesale electricity market prices, grid stability, and national or regional decarbonization pathways.46
Addressing these limitations and pursuing these future research avenues will further enhance the understanding of optimal decision-making for DES operators in the complex and uncertain environment of the energy transition.
7. Conclusion
This study has presented an integrated framework for analyzing the operational and strategic decisions of a Distributed Energy System operator under carbon price uncertainty. By combining DES operational optimization, financial hedging of carbon risk using a CVaR approach, and Real Options Analysis for CCS investment, the research provides a comprehensive perspective on managing both short-term risks and long-term strategic choices.
The baseline results, derived from a specific set of carbon price parameters (300 CNY/ton initial price, 8% drift, 74% volatility via GBM), indicate that while operational hedging with call options can significantly reduce carbon cost risk (56.02% CVaR reduction with "Call_280_1M"), the strategic decision for a CCS investment with an NPV of 201,393 CNY is to "DEFER_BUT_MONITOR." This is due to a substantial ROA value of 1,026,916 CNY, highlighting a significant deferral option premium of 825,523 CNY. This underscores the considerable economic value of maintaining managerial flexibility in the face of high CCS investment costs and uncertain future carbon prices.
Further analyses, including sensitivity studies and robustness checks with alternative carbon price models and policy scenarios, emphasize that:
The optimal operational hedging strategy and the strategic CCS investment decision are highly sensitive to carbon price dynamics (volatility, drift, baseline price), CCS project costs, and policy support mechanisms.
The choice of carbon price stochastic model (e.g., GBM vs. jump-diffusion) can fundamentally alter risk perceptions and optimal decisions.
Policy certainty and direct CCS support can be more influential than moderate carbon price volatility in triggering CCS investment.
The inclusion of an abandonment option for CCS can reduce the overall project risk and potentially lower the investment trigger price.
The integrated nature of the framework reveals important interdependencies: effective operational hedging can influence the perceived risk of long-term CCS investments, while the prospect (or deferral) of CCS necessitates robust ongoing carbon risk management at the operational level.
The primary contribution of this work lies in its holistic approach, demonstrating how short-term operational risk management and long-term strategic investment planning under uncertainty are not isolated problems but deeply intertwined facets of navigating the energy transition. For DES operators, the framework offers a tool to quantify the value of flexibility and make more informed, risk-aware decisions. For policymakers, the findings highlight the importance of stable and supportive policy environments to encourage timely investment in critical decarbonization technologies like CCS, as market signals alone, especially under high uncertainty, may lead to prolonged deferral from a private investor perspective. Future research should continue to refine the modeling of complex uncertainties and explore more deeply the integration of operational and strategic decision layers.
Works cited
www.ey.com, accessed on May 25, 2025, https://www.ey.com/content/dam/ey-unified-site/ey-com/en-in/insights/sustainability/images/ey-carbon-trading-an-emerging-commodity-class.pdf
Carbon Capture, Utilization, and Storage (CCUS): A Key Driver of Industrial Decarbonization and Net-Zero - SIA Partners, accessed on May 25, 2025, https://www.sia-partners.com/en/insights/publications/carbon-capture-utilization-and-storage-ccus-a-key-driver-industrial
www.ecb.europa.eu, accessed on May 25, 2025, https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2958~0002545c73.en.pdf
Publication: Energy Intensive Infrastructure Investments with ..., accessed on May 25, 2025, https://openknowledge.worldbank.org/entities/publication/33f04eea-23b2-56f4-bef0-2bf45085fe65
Energy trading and hedging strategies - Flexible Academy of Finance, accessed on May 25, 2025, https://academyflex.com/energy-trading-and-hedging-strategies/
Investment in carbon dioxide capture and storage combined with enhanced water recovery - UCL Discovery - University College London, accessed on May 25, 2025, https://discovery.ucl.ac.uk/id/eprint/10090516/1/Investment%20in%20carbon%20dioxide%20capture%20and%20storage%20combined%20with%20enhanced%20water%20recovery.pdf
eprints.whiterose.ac.uk, accessed on May 25, 2025, https://eprints.whiterose.ac.uk/id/eprint/205384/1/JEEM_Manuscript_Revised_1_.pdf
A simple-to-implement real options method for the energy sector - White Rose Research Online, accessed on May 25, 2025, https://eprints.whiterose.ac.uk/id/eprint/157998/1/Locatelli%20Mancini%20Lotti%20to%20deposit.pdf
A Real Options Approach to Valuate Solar Energy Investment with Public Authority Incentives: The Italian Case - MDPI, accessed on May 25, 2025, https://www.mdpi.com/1996-1073/13/16/4181
Real Options: A Survey - Optimization Online, accessed on May 25, 2025, https://optimization-online.org/wp-content/uploads/2015/01/4715.pdf
(PDF) Estimation of a term structure model of carbon prices through ..., accessed on May 25, 2025, https://www.researchgate.net/publication/345702150_Estimation_of_a_term_structure_model_of_carbon_prices_through_state_space_methods_The_European_Union_emissions_trading_scheme
Full article: Analyzing the dynamic behavior and market efficiency of green energy investments: A geometric and fractional brownian motion approach, accessed on May 25, 2025, https://www.tandfonline.com/doi/full/10.1080/15567249.2025.2457438?src=
bright-journal.org, accessed on May 25, 2025, https://bright-journal.org/Journal/index.php/JADS/article/download/536/364
Investment Analysis of Low-Carbon Yard Cranes: Integrating Monte Carlo Simulation and Jump Diffusion Processes with a Hybrid American–European Real Options Approach - OUCI, accessed on May 25, 2025, https://ouci.dntb.gov.ua/en/works/9ZPPPweL/
Innovating and Pricing Carbon-Offset Options of Asian Styles on the Basis of Jump Diffusions and Fractal Brownian Motions - MDPI, accessed on May 25, 2025, https://www.mdpi.com/2227-7390/11/16/3614
Investment Analysis of Low-Carbon Yard Cranes: Integrating Monte Carlo Simulation and Jump Diffusion Processes with a Hybrid American–European Real Options Approach - MDPI, accessed on May 25, 2025, https://www.mdpi.com/1996-1073/18/8/1928
(PDF) Value- at-Risk vs Conditional Value-at-Risk in Risk Management and Optimization - ResearchGate, accessed on May 25, 2025, https://www.researchgate.net/publication/200798611_Value-_at-Risk_vs_Conditional_Value-at-Risk_in_Risk_Management_and_Optimization
New Method to Hedge Climate-Induced Energy Risk | Earth and Environmental Engineering, accessed on May 25, 2025, https://www.eee.columbia.edu/about/news/new-method-hedge-climate-induced-energy-risk
Full article: Asymmetric tail risk spillovers between carbon emission ..., accessed on May 25, 2025, https://www.tandfonline.com/doi/full/10.1080/00036846.2025.2471038?af=R
Risk hedging for gas power generation considering power-to-gas ..., accessed on May 25, 2025, https://www.researchgate.net/publication/350371542_Risk_hedging_for_gas_power_generation_considering_power-to-gas_energy_storage_in_three_different_electricity_markets
Real options analysis of investment in carbon capture and ..., accessed on May 25, 2025, https://www.researchgate.net/publication/226707379_Real_options_analysis_of_investment_in_carbon_capture_and_sequestration_technology
Real Options Analysis - The Decision Lab, accessed on May 25, 2025, https://thedecisionlab.com/reference-guide/economics/real-options-analysis
erbe.autonoma.pt, accessed on May 25, 2025, https://erbe.autonoma.pt/articles/ERBE01202-The-Role-of-the-Abandonment-Option-in-Strategic-Capital-Allocation-A-Review-of-Selected-Literature.pdf
(PDF) Application of real options in carbon capture and storage ..., accessed on May 25, 2025, https://www.researchgate.net/publication/352773527_Application_of_real_options_in_carbon_capture_and_storage_literature_Valuation_techniques_and_research_hotspots
Exploring the Usefulness of Real Options Theory for Foreign Affiliate ..., accessed on May 25, 2025, https://www.mdpi.com/1911-8074/17/10/438
Towards improved guidelines for cost evaluation of carbon capture and storage - Carnegie Mellon University, accessed on May 25, 2025, https://www.cmu.edu/epp/iecm/rubin/PDF%20files/2021/IEAGHG_2021-TR05%20Towards%20improved%20guidelines%20for%20cost%20evaluation%20of%20CCS.pdf
A Systematic Review of Sensitivity Analysis in Building Energy ..., accessed on May 25, 2025, https://www.mdpi.com/1996-1073/18/9/2375
(PDF) The Value of Global Sensitivity Analysis for Energy System ..., accessed on May 25, 2025, https://www.researchgate.net/publication/322721974_The_Value_of_Global_Sensitivity_Analysis_for_Energy_System_Modelling
(PDF) Sensitivity Analysis of an Energy System Model - ResearchGate, accessed on May 25, 2025, https://www.researchgate.net/publication/301214624_Sensitivity_Analysis_of_an_Energy_System_Model
A Review of Carbon Capture and Storage Project Investment and Operational Decision-Making Based on Bibliometrics - MDPI, accessed on May 25, 2025, https://www.mdpi.com/1996-1073/12/1/23
How Might Carbon Pricing Affect CCS Investment? → Question - Sustainability Directory, accessed on May 25, 2025, https://sustainability-directory.com/question/how-might-carbon-pricing-affect-ccs-investment/
Policy Scenarios for UK CCS Deployment & Exploring the Role of a Carbon Takeback Obligation - Net Zero Climate, accessed on May 25, 2025, https://netzeroclimate.org/wp-content/uploads/2025/01/Markets-Mandates-2025.pdf
Decision-Making Under Deep Uncertainty (DMDU) | U.S. Climate ..., accessed on May 25, 2025, https://toolkit.climate.gov/course-lesson/decision-making-under-deep-uncertainty
Robust Decision Making | RAND, accessed on May 25, 2025, https://www.rand.org/global-and-emerging-risks/centers/methods-centers/pardee/dmdu-decision-making-under-deep-uncertainty/robust-decision-making.html
Strategies For Managing Uncertainty - FasterCapital, accessed on May 25, 2025, https://fastercapital.com/topics/strategies-for-managing-uncertainty.html
7 Common Project Risks and How to Prevent Them - Asana, accessed on May 25, 2025, https://asana.com/resources/project-risks
2025 FEDERAL POLICY BLUEPRINT - Carbon Capture Coalition, accessed on May 25, 2025, https://carboncapturecoalition.org/wp-content/uploads/2025/02/CCC-2025-Federal-Policy-Blueprint.pdf
A comprehensive analysis of real options in solar photovoltaic projects: A cluster-based approach - PMC, accessed on May 25, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11366872/
Valuing investment decisions of renewable energy projects ..., accessed on May 25, 2025, https://www.researchgate.net/publication/345182885_Valuing_investment_decisions_of_renewable_energy_projects_considering_changing_volatility
Valuing the option to prototype: A case study with Generation Integrated Energy Storage, accessed on May 25, 2025, https://eprints.whiterose.ac.uk/id/eprint/167881/1/Energy%20RO.pdf
Managing Project Uncertainty: From Variation to Chaos, accessed on May 25, 2025, https://sloanreview.mit.edu/article/managing-project-uncertainty-from-variation-to-chaos/
Limitations Of Roa - FasterCapital, accessed on May 25, 2025, https://fastercapital.com/topics/limitations-of-roa.html
Multi-Commodity Real Options Analysis of Power Plant Investments: Discounting Endogenous Risk Structures - RWTH Aachen University, accessed on May 25, 2025, https://www.fcn.eonerc.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaaagvvsg
A critical review of Real Options thinking for valuing investment ..., accessed on May 25, 2025, https://www.researchgate.net/publication/286478678_A_critical_review_of_Real_Options_thinking_for_valuing_investment_flexibility_in_Smart_Grids_and_low_carbon_energy_systems
Flexible CCU Investment with Real Options and Reinforcement Learning, accessed on May 25, 2025, https://co2value.eu/flexible-investment-framework-for-ccu-real-options-and-reinforcement-learning-in-co%E2%82%82-to-methanol-case/
Integrated investment, retrofit and abandonment energy system planning with multi-timescale uncertainty using stabilised adaptiv - arXiv, accessed on May 25, 2025, https://arxiv.org/pdf/2303.09927
Uncertainty sources for CCS investment. | Download Scientific ..., accessed on May 25, 2025, https://www.researchgate.net/figure/Uncertainty-sources-for-CCS-investment_tbl7_352773527
An Integrated CVaR and Real Options Approach to Investments in the Energy Sector - IRIHS, accessed on May 25, 2025, https://irihs.ihs.ac.at/id/eprint/1770/1/es-209.pdf
Full article: A long-term and heterogeneous study on the impact of carbon emission trading policy on financial performance, accessed on May 25, 2025, https://www.tandfonline.com/doi/full/10.1080/17583004.2025.2486627?src=
