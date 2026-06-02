# Evaluating Explanation Reliability in ICU Mortality Risk Prediction

This repository contains the analysis workflow for a clinical machine learning project on **ICU in-hospital mortality risk prediction** using the **PhysioNet/Computing in Cardiology Challenge 2012 dataset**.

The central goal is not only to train accurate mortality prediction models, but to evaluate whether their explanations are **stable, clinically plausible, robust to false signals, subgroup-aware, and supported by independent clinical evidence**.

## Overview

Machine learning models can achieve strong predictive performance while relying on unstable or clinically questionable signals. In high-stakes clinical settings such as intensive care, model explanations should therefore be evaluated separately from discrimination metrics such as AUROC or AUPRC.

This project proposes an **explanation reliability framework** for ICU mortality prediction. The framework combines:

| Component | Purpose |
|---|---|
| External validation | Test whether model performance generalizes to a held-out cohort |
| Calibration analysis | Evaluate whether predicted risks match observed mortality rates |
| Clinical support analysis | Check whether model-associated features align with independent mortality evidence |
| Explanation stability | Measure whether top-ranked features are stable across cross-validation folds |
| Method agreement | Compare explanations from different attribution methods |
| False-signal sensitivity | Test whether explanations are vulnerable to noise, permuted controls, or spurious proxies |
| Subgroup audit | Evaluate threshold-dependent prediction behavior across patient subgroups |
| Temporal trajectory analysis | Assess whether mortality risk is associated with first-48-hour deterioration patterns |

The project is framed as a **reliability audit for clinical ML explanations**, rather than as a claim that one mortality prediction model is sufficient for clinical deployment.

## Research Question

Can ICU mortality prediction models be evaluated not only by predictive performance, but also by whether their explanations are reproducible, clinically supported, resistant to false signals, subgroup-aware, and consistent with temporal deterioration patterns?

## Dataset

The project uses the **PhysioNet/Computing in Cardiology Challenge 2012 ICU dataset**.

| Cohort | Role | Number of patients | Notes |
|---|---:|---:|---|
| Set A + Set B | Development cohort | 8,000 | Used for preprocessing design, model development, cross-validation, feature analysis, and internal tuning |
| Set C | External test cohort | 4,000 | Held out for final external evaluation |

The outcome is **binary in-hospital mortality**.

## External Validation Design

Sets A and B were combined to form the development cohort, while Set C was reserved as the external test cohort.

Set C was not used for:

- Feature selection
- Preprocessing parameter estimation
- Hyperparameter tuning
- Decision threshold selection
- Explanation metric design
- Temporal model checkpoint selection

This separation was used to reduce optimistic bias and evaluate whether model performance and explanation behavior generalized beyond the development cohort.

## Tabular Feature Construction

Raw ICU time-series measurements were converted into patient-level tabular features. For repeated variables, summary statistics were computed, including:

- Mean
- Minimum
- Maximum
- Standard deviation
- First value
- Last value
- Measurement count

Static variables such as age, sex, height, weight, and ICU admission type were retained when available. Missingness indicators were added. Outcome-related and post-outcome variables were excluded to reduce leakage risk.

The final tabular feature matrix contained **243 numeric features**.

## Models

Three main tabular mortality prediction models were trained and externally evaluated.

| Model | Main preprocessing / tuning |
|---|---|
| Logistic Regression | Mean imputation, standardization, 5-fold grid search using ROC-AUC |
| XGBoost | 5-fold ROC-AUC grid search over tree depth, learning rate, estimators, subsampling, and column sampling |
| Multilayer Perceptron | Internal stratified development split, selected using validation ROC-AUC and Brier score |

The selected XGBoost configuration used:

| Hyperparameter | Value |
|---|---:|
| max_depth | 4 |
| learning_rate | 0.05 |
| n_estimators | 200 |
| subsample | 0.8 |
| colsample_bytree | 1.0 |

Logistic regression and XGBoost used a threshold of **0.50**. The multilayer perceptron threshold was selected using validation F1-score.

## External-Test Performance

External-test results on Set C showed that XGBoost had the strongest overall discrimination-calibration profile among the tabular models.

| Model | AUROC | AUPRC | F1-score | Brier score |
|---|---:|---:|---:|---:|
| XGBoost | 0.875 | 0.584 | 0.448 | 0.087 |
| Multilayer Perceptron | 0.867 | 0.546 | 0.539 | 0.091 |
| Logistic Regression | 0.863 | 0.548 | 0.503 | 0.150 |

Key interpretation:

- XGBoost achieved the highest AUROC, highest AUPRC, and lowest Brier score.
- The multilayer perceptron achieved the highest F1-score.
- Logistic regression had competitive discrimination but weaker calibration.
- Discrimination alone was not sufficient for evaluating model reliability.

## Clinical Support Analysis

To evaluate whether model-associated features were supported by independent clinical evidence, the project performed:

- Development-cohort differential analysis
- Standardized mean difference analysis
- Welch two-sample t-tests
- Benjamini-Hochberg false-discovery-rate correction
- Held-out effect-size direction validation in Set C
- Univariate Cox proportional hazards support analysis

Differential features were selected using:

- False discovery rate `< 0.05`
- Absolute standardized mean difference `>= 0.20`

The differential analysis selected **91 mortality-associated features**. Among these selected features, held-out Set C effect-size validation showed same-direction replication.

The top-30 logistic regression and XGBoost feature rankings overlapped in **12 features**, with a Jaccard overlap of **0.25**.

Shared top-ranked features included:

- `GCS_last`
- `BUN_last`
- `static_Age`
- `Lactate_last`
- `Urine_mean`
- `GCS_mean`
- `Urine_total`
- `Creatinine_median`
- `PaO2_median`
- `HR_mean`
- `GCS_std`
- `Temp_median`

All 12 shared features showed held-out same-direction evidence and CoxPH support. Ten of the 12 were also differential-selected. `GCS_std` and `Temp_median` were not differential-selected but were supported by held-out direction and CoxPH association.

## Explanation Reliability Framework

The explanation reliability analysis evaluated whether model explanations were stable and consistent.

### Cross-Fold Stability

Cross-fold explanation stability was measured as the mean pairwise Jaccard overlap of top-20 features across five stratified folds.

| Model | Mean top-20 Jaccard stability | SD |
|---|---:|---:|
| Logistic Regression | 0.428 | 0.077 |
| Linear SVM | 0.427 | 0.071 |
| XGBoost | 0.417 | 0.065 |

Interpretation: explanations were moderately stable within each model, but stability was not high enough to treat any single feature ranking as definitive.

### Attribution Method Agreement

Method agreement was measured as the Jaccard overlap between top-20 feature sets from different attribution methods.

| Method pair | Top-20 Jaccard overlap |
|---|---:|
| Logistic Regression vs Linear SVM | 0.429 |
| Logistic Regression vs XGBoost gain | 0.081 |
| Logistic Regression vs XGBoost permutation | 0.053 |
| Linear SVM vs XGBoost gain | 0.081 |
| Linear SVM vs XGBoost permutation | 0.026 |
| XGBoost gain vs XGBoost permutation | 0.176 |

Interpretation: agreement was highest between the two coefficient-based models, but much lower between linear and tree-based explanation methods.

### Shared Explanation Features

Features selected by at least two attribution methods included:

- `GCS_last`
- `PaCO2_count`
- `Urine_mean`
- `WBC_mean`
- `pH_median`
- `BUN_last`
- `WBC_max`
- `static_Age`
- `GCS_std`
- `Platelets_min`
- `Urine_total`
- `pH_last`

These features mapped to neurological, renal, metabolic or acid-base, hematologic, measurement-process, and static demographic domains.

## False-Signal Sensitivity

False-signal sensitivity was evaluated by injecting three types of artificial features:

| Injected feature type | Purpose |
|---|---|
| Gaussian noise features | Test whether random noise appears important |
| Permuted real-feature controls | Test whether broken real-feature associations appear important |
| Spurious proxy features | Positive-control stress test for shortcut learning |

Main result:

| Injected signal type | False explanation behavior |
|---|---|
| Gaussian noise | False explanation rate 0.0 |
| Permuted real-feature controls | False explanation rate 0.0 |
| Spurious proxy features | False explanation rate 1.0 |

Interpretation: attribution methods did not elevate random or permuted controls, but they were highly vulnerable to label-correlated spurious proxy features. This should be interpreted as a **shortcut-learning stress test**, not as causal evidence or proof that a real clinical spurious variable was present.

## AIF360-Style Subgroup Audit

A post-hoc subgroup audit was performed on external Set C using formula-equivalent AIF360-style metrics.

Important note: AIF360 itself was not available in the recorded execution environment, so the analysis used **formula-equivalent AIF360-style metric implementations**. No fairness mitigation method was applied.

Subgroups evaluated:

- Age group
- Sex
- ICU admission type

Metrics included:

| Metric | Interpretation |
|---|---|
| Statistical parity difference | Difference in predicted high-risk assignment rate |
| Disparate impact | Ratio of predicted high-risk assignment rates |
| Equal opportunity difference | Difference in true positive rates |
| Average odds difference | Average of TPR and FPR differences |

Because the positive label represented predicted high mortality risk, these metrics were interpreted as differences in **high-risk assignment and error rates**, not as favorable-outcome fairness metrics.

### External-Test Subgroup Performance

| Subgroup | n | Mortality | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| Age <=65 | 1,894 | 10.93% | 0.878 | 0.520 |
| Age >65 | 2,106 | 17.95% | 0.850 | 0.569 |
| Female | 1,768 | 16.12% | 0.860 | 0.558 |
| Male | 2,228 | 13.47% | 0.872 | 0.545 |
| Cardiac Surgery Recovery Unit | 874 | 5.26% | 0.896 | 0.488 |
| Coronary Care Unit | 602 | 11.79% | 0.829 | 0.450 |
| Medical ICU | 1,376 | 21.15% | 0.833 | 0.597 |
| Surgical ICU | 1,148 | 15.42% | 0.867 | 0.528 |

Age and ICU admission type showed larger subgroup variation than sex. ICU-type contrasts were especially important because observed mortality prevalence and model behavior differed substantially across ICU admission groups.

## Temporal Sequence Modeling

Temporal analysis was performed as complementary trajectory evidence using first-48-hour ICU measurements.

The time-series data were represented as:

- 36 dynamic ICU variables
- 12 four-hour bins
- Value tensor with shape `[patients, 12, 36]`
- Observation-mask tensor with shape `[patients, 12, 36]`
- Concatenated Medformer input with shape `[patients, 12, 72]`

Missing standardized values were set to zero, and the observation mask indicated whether a measurement was observed.

### Medformer External-Test Performance

| Model | AUROC | AUPRC | F1-score | Brier score |
|---|---:|---:|---:|---:|
| Medformer temporal model | 0.835 | 0.495 | 0.424 | 0.099 |

The Medformer model showed reasonable external-test temporal discrimination, but it did not exceed the strongest tabular XGBoost model.

Time-bin perturbation identified the **44-48 hour window** as the dominant temporal window, with smaller contributions from earlier and mid-window bins.

## Markov Trajectory Analysis

A rule-based Markov-chain severity representation was constructed as complementary trajectory evidence.

Each four-hour bin was assigned one of four severity states:

1. Stable
2. Moderate
3. High
4. Critical

The severity score used predefined abnormalities in neurological, hemodynamic, renal, respiratory, metabolic, electrolyte, and temperature variables. State cutoffs were estimated from internal-training severity-score quartiles and applied unchanged to validation and Set C.

Trajectory findings showed that deaths had:

- Lower stable-state persistence
- Greater movement toward higher-severity states
- Higher late severity
- Higher mean severity
- Higher final state

Survivors had:

- More stable time
- Stronger stable-to-stable transition behavior

Important interpretation note: the Markov severity representation is a **heuristic trajectory summary**, not a validated clinical severity score.

## Main Figures

| Figure | Description |
|---|---|
| Figure 1 | Cohort definition and external validation design |
| Figure 2 | External-test discrimination and calibration of tabular models |
| Figure 3 | Differential, CoxPH, and clinical support for model-associated features |
| Figure 4 | Explanation reliability across folds, models, and attribution methods |
| Figure 5 | AIF360-style subgroup audit |
| Figure 6 | Temporal Medformer prediction and Markov trajectory analysis |

## Supplementary Figures

| Supplementary Figure | Description |
|---|---|
| Figure S1 | Supplementary cohort characteristics |
| Figure S2 / S3 | Model-specific feature importance summaries |
| Figure S3 | Differential and CoxPH support details |
| Figure S4 | False-signal sensitivity and clinical-domain composition |
| Figure S5 | Subgroup diagnostics and threshold sensitivity |

Figure numbering may need to be finalized during manuscript preparation because some supplementary figure filenames currently contain overlapping `S3` labels.

## Suggested Repository Structure

```text
.
├── README.md
├── manuscript/
│   └── main_manuscript.tex
├── data/
│   ├── raw/
│   ├── processed/
│   └── cox/
├── notebooks/
│   ├── 09_combined_model_training_csv_export.ipynb
│   ├── 10_make_figure2_and_overlap_revised_v8_170mm.ipynb
│   ├── 01_compute_figure3_results_only_v6_with_cox_folder.ipynb
│   ├── 02_make_figure3_main_and_supplementary_170mm_v9_feature_column_rank.ipynb
│   ├── make_figure4_explanation_reliability_spurious_stress.ipynb
│   ├── make_figure4_main_170mm_publication_ready_v6.ipynb
│   ├── 01_temporal_modeling_medformer_markov_colab.ipynb
│   └── 02_temporal_figures_170mm_colab.ipynb
├── results/
│   ├── model_performance/
│   ├── feature_importance/
│   ├── explanation_reliability/
│   ├── fairness/
│   └── temporal/
├── figures/
│   ├── Figure_1/
│   ├── Figure_2/
│   ├── Figure_3/
│   ├── Figure_4/
│   ├── Figure_5/
│   └── Figure_6/
└── requirements.txt
```

This structure is suggested for clarity. Actual notebook names and folder paths may differ depending on the local analysis environment.

## Reproducibility Notes

To reproduce the analysis, follow the development-to-external-test design carefully.

Recommended workflow:

1. Prepare raw PhysioNet Challenge 2012 Sets A, B, and C.
2. Construct patient-level tabular features using Sets A+B as the development cohort and Set C as the held-out external test cohort.
3. Fit preprocessing parameters only on the development cohort or internal training splits.
4. Train logistic regression, XGBoost, and multilayer perceptron models.
5. Evaluate final model performance on Set C.
6. Generate Figure 2 performance and calibration panels.
7. Run development-only differential analysis and CoxPH support analysis.
8. Generate Figure 3 clinical support panels.
9. Run explanation stability, method agreement, and false-signal stress tests.
10. Generate Figure 4 explanation reliability panels.
11. Run the post-hoc AIF360-style subgroup audit on Set C predictions.
12. Generate Figure 5 subgroup audit panels.
13. Prepare first-48-hour temporal tensors.
14. Train and evaluate the Medformer temporal model.
15. Construct Markov trajectory features and transition summaries.
16. Generate Figure 6 temporal trajectory panels.

## Leakage Prevention Checklist

The following rules should be preserved when reproducing or extending the project:

- Do not use Set C for model development or hyperparameter tuning.
- Do not estimate preprocessing parameters using Set C.
- Do not select features using Set C.
- Do not tune decision thresholds using Set C.
- Do not select temporal checkpoints using Set C.
- Use development-only statistics for imputation, standardization, and temporal scaling.
- Treat Set C as a final external evaluation cohort.
- Interpret held-out effect-size replication as validation evidence, not as a feature-selection step.

## Interpretation Cautions

This project should be interpreted with the following limitations:

| Area | Caution |
|---|---|
| Explanation reliability | Stable explanations do not establish causality |
| CoxPH support | CoxPH results are association evidence and depend on survival-model assumptions |
| False-signal sensitivity | Spurious proxy injection is a stress test, not proof of real-world confounding |
| AIF360-style audit | Metrics were formula-equivalent; no mitigation algorithm was applied |
| Subgroup analysis | Results may depend on subgroup size, prevalence, and decision threshold |
| Medformer temporal model | Temporal analysis is complementary and did not outperform the strongest tabular model |
| Markov trajectory model | The severity score is heuristic and not a validated clinical score |
| Clinical deployment | The workflow is an audit framework and is not sufficient for deployment without prospective validation |

## Key Takeaway

In ICU mortality prediction, strong model performance alone is not enough to establish trustworthy clinical interpretation. This project shows that clinical ML models should also be evaluated for explanation stability, attribution-method agreement, false-signal vulnerability, clinical support, subgroup behavior, calibration, and temporal trajectory consistency.

The main contribution is an **explanation reliability workflow** for auditing high-stakes clinical machine learning models.

## Authors

- Junseo Park, The University of British Columbia
- Tia Sajeev, The University of British Columbia
- Steven Huang, The University of British Columbia
- Levi Dhanaraj, The University of British Columbia
- Laura Luo, The University of British Columbia

## Acknowledgements

The authors thank the STEM Fellowship 2026 Inter-University Health Data and AI Inquiry Program for supporting this research project.

## License

A license has not yet been specified. Add an appropriate license before public release if this repository will be shared publicly.

## Data Availability

The original ICU data are available through the PhysioNet/Computing in Cardiology Challenge 2012 dataset. This repository should not redistribute restricted or externally licensed data unless redistribution is explicitly permitted.
