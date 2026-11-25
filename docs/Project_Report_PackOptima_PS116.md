# PackOptima: Physics-Informed Three-Dimensional Bin Packing, Calibrated Transit Damage Risk Estimation, and Carbon-Aware Freight Optimization (PS-116)

**Author:** P Yathin Machaiah  
**Department:** Department of CSE - Cyber Security  
**University:** Dayananda Sagar University  
**Email:** p.yathinmachaiah18@gmail.com

---

## Abstract

Automated e-commerce fulfillment confronts competing geometric, mechanical, environmental, and financial constraints when consolidating multi-SKU orders. This technical report details PackOptima (PS-116), unifying compatibility-constrained partitioning across 10 SKU categories, OR-Tools CP-SAT three-dimensional bin packing over 14 BIS IS 2771 (Part 1): 2022 and cold-chain container tiers with ISTA 3A cushion allowances (`Shelf-FFD` fallback for `n > 35`), isotonic-calibrated `XGBoost` transit damage estimation trained over `12,000` shipments (`ROC-AUC = 0.8919`, `Brier = 0.0848`), Indian PIN-zone (Zones A–E) freight pricing, Scope 3 carbon accounting, SQLite relational persistence, and strictly isolated Gemini 2.5 / Qwen2.5 narrative rendering.

---

## Key Empirical Findings

| Metric / Model | Proposed / XGBoost | Comparison Baseline | Notes |
|---|---:|---:|---|
| Damage Classifier Test ROC-AUC | 0.8919 | RF 0.8892 / HGB 0.8827 | 3-fold isotonic calibration (`n_test = 2,400`) |
| Damage Classifier Brier Score | 0.0848 | RF 0.0857 / HGB 0.0879 | Lower indicates superior calibration |
| 200-Cart Mean Damage Risk | 0.112 | No Grouping 0.321 / Manual 0.437 | Wilcoxon p = 7.59e-21 |
| 200-Cart Mean Volumetric Utilization | 28.6% | Always Largest 1.2% | Reduces dimensional surcharge |
| Compatibility Grouping Ablation | 0.112 (ON) | 0.321 (OFF) | +185.6% risk shift |
| GenAI Narrative Isolation Delta | 0.00% | N/A | Zero numerical solver perturbation |
| Baseline Sensitivity Risk | 0.2870 | ISTA Cushion Elast. 0.678 | +/-25% perturbation sensitivity peak |
