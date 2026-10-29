# MatRisk AI — Short Viva Preparation Notes

Use these notes to practise in your own words. Read the actual Python functions and run the project before presenting. Do not memorise metrics without understanding what was tested.

## 1. Explain the project in 30 seconds

“My project explores how the condition of a material can affect the financial risk of an asset made from it. I used the sample data supplied with the assignment to build a first working version. It predicts a few material properties, compares commodity models with and without material-financial features, estimates bridge-deficiency risk, runs material-shock simulations and calculates the carbon and cost impact of substituting materials. I also made a small dashboard and a decision game to show the results.”

## 2. Why connect material science with finance?

A material can get weaker, corrode, or become more expensive to source. That may lead to repairs, loss of asset life, higher insurance loss, or a change in commodity demand. The point is to provide financial analysts with extra physical context, not to claim that one model can solve every part of the problem.

## 3. What datasets did you use?

I used the six provided CSV files: 5,500 material records, about 22,952 commodity observations, 5,000 bridge records, about 22,952 cross-domain rows, 5,280 monthly element-price rows and 2,000 historical failure records. They are supplied sample data; I have not independently verified them as production feeds.

## 4. Why did you use ExtraTrees for the material model?

The sample file provides a formula and some structure descriptors, but it does not provide atomic coordinates for every material. ExtraTrees is a practical tree-ensemble baseline for learning non-linear patterns from fixed features. A crystal graph neural network would require full crystal structures and a different data pipeline, so I did not claim that the baseline is a CGNN.

## 5. What is the difference between a PINN and the physics check here?

A PINN includes physical equations in the model's training loss. In this prototype, I perform physical bounds checks on the supplied data and derive Poisson's ratio from predicted bulk and shear moduli after prediction. This makes the displayed elastic relationship consistent, but it is post-processing and not a trained PINN.

## 6. How did you prevent future information leaking into commodity prediction?

The target is the next trading day's log return, created by shifting the return within each commodity. Models train on dates before 1 January 2023 and are evaluated on dates from 2023 onward. The features are merged on the same date and commodity. A production experiment should still use repeated walk-forward tests, realistic costs and a deeper timestamp audit.

## 7. What did the commodity results show?

Directional accuracy was only about 50.1% to 50.4% across the four variants. The full feature model had a toy Sharpe ratio of about 0.006, well below the brief's requested target above 0.5. My conclusion is that this sample run does not establish a strong trading signal. This is an important negative result, not something to hide.

## 8. Why does the bridge classifier have such a high ROC-AUC?

The sample holdout is small and the positive deficiency class is rare, around 3.2%. A high ROC-AUC is only a ranking measure and may reflect how the supplied labels were generated. I would check the sample for label rules and leakage and validate against real, dated inspection data before using this result for decisions.

## 9. Explain PD, LGD, EAD and expected loss

- **PD:** probability of default within a defined time horizon.
- **LGD:** fraction of exposure lost if default happens.
- **EAD:** exposure outstanding when default happens.
- **Expected loss:** PD × LGD × EAD.

In my prototype, the classifier output is a bridge-deficiency probability proxy, not a calibrated loan-default PD. The financial number is therefore called an expected-loss proxy.

## 10. What does the stress simulator do?

It samples 10,000 loss outcomes around a scenario-specific mean and uncertainty. For example, the steel-quality scenario uses a 6% mean impact assumption in the teaching configuration. The model then reports the simulated mean loss, VaR and TVaR. The impact rates are assumptions, not fitted causal effects; changing them changes the output.

## 11. What is the ESG calculation?

For a chosen material quantity, I compare emissions from the original material with emissions from the substitute. Avoided emissions equal the emissions difference. Incremental cost is quantity multiplied by the green premium. Cost per tonne avoided is incremental cost divided by avoided tonnes of CO2-equivalent emissions. Supplier-specific lifecycle boundaries must be checked before real reporting.

## 12. What is not fully implemented yet?

The current version is a prototype. It does not include a trained CGNN, a PINN training loss, a WGAN-GP generator, time-to-failure survival model, live 100K-plus material/API ingestion, full National Bridge Inventory pipeline, a fully validated commodity trading strategy, full SHAP/PDP for every model, or production-grade CI/DVC/MLflow controls. These are documented as future work rather than claimed as complete.

## 13. Questions you should be able to answer from the code

1. What is the prediction target in each model?
2. Why was a time split used for commodity returns but a grouped split for material formulas?
3. What changes when `mqi_21d_trend` is removed from the model?
4. Why is `expected_loss_proxy_M` not a bank-approved expected-loss estimate?
5. Which assumptions control the Monte Carlo loss distribution?
6. How does the ESG code calculate cost per tonne avoided?
7. What is the difference between the sample result and an externally validated result?

## 14. Before your review

- Create your own virtual environment and run the project start-to-finish.
- Change one modelling choice (for example, tree depth or one feature group), retrain, and write down what changes.
- Capture screenshots from your own local dashboard run.
- Replace placeholders with your details and your own observations.
- Follow your internship's rules on AI assistance and attribution. Do not describe a component as implemented unless you can show the code and explain it.
