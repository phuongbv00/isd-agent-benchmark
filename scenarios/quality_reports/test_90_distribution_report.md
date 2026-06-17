# Test 90 Distribution Report

Created at: 2026-06-17T16:10:37

## Summary
- Source split: `scenarios/test` (1202 scenarios)
- Target subset: `scenarios/test_90` (90 scenarios)
- Sampling seed: `42`
- Allocation: 10 domains × 3 difficulty levels × 3 scenarios
- Sampling method: deterministic stratified random sampling within each domain-difficulty cell

## Allocation Matrix

| Domain | Easy | Moderate | Hard | Total |
|---|---:|---:|---:|---:|
| AI | 3 | 3 | 3 | 9 |
| Business/HR/Admin Support | 3 | 3 | 3 | 9 |
| Education (Teaching & Learning) | 3 | 3 | 3 | 9 |
| Language | 3 | 3 | 3 | 9 |
| Mathematics | 3 | 3 | 3 | 9 |
| Medical/Nursing | 3 | 3 | 3 | 9 |
| Science | 3 | 3 | 3 | 9 |
| Service/Customer Support | 3 | 3 | 3 | 9 |
| Social Studies | 3 | 3 | 3 | 9 |
| Software Development/IT | 3 | 3 | 3 | 9 |
| **Total** | **30** | **30** | **30** | **90** |

## Source Availability Per Cell

| Domain | Easy | Moderate | Hard | Total |
|---|---:|---:|---:|---:|
| AI | 12 | 14 | 15 | 41 |
| Business/HR/Admin Support | 12 | 15 | 17 | 44 |
| Education (Teaching & Learning) | 14 | 17 | 18 | 49 |
| Language | 80 | 82 | 76 | 238 |
| Mathematics | 85 | 79 | 70 | 234 |
| Medical/Nursing | 12 | 15 | 15 | 42 |
| Science | 81 | 77 | 78 | 236 |
| Service/Customer Support | 13 | 14 | 15 | 42 |
| Social Studies | 77 | 77 | 80 | 234 |
| Software Development/IT | 11 | 15 | 16 | 42 |
| **Total** | **397** | **405** | **400** | **1202** |

## Variant Type Counts

| Variant type | Count |
|---|---:|
| balance_large_scale | 8 |
| balance_large_scale_v2 | 9 |
| balance_large_scale_v3 | 1 |
| balance_medium_scale | 1 |
| balance_self_directed | 1 |
| balance_self_filtered | 1 |
| balance_teen_filtered | 4 |
| balance_teen_k12 | 4 |
| context_variant | 49 |
| idld_aligned | 12 |

## Selected Files By Cell

### AI
- Easy: `scenario_bal_large2_0117.json`, `scenario_bal_large2_1197.json`, `scenario_idld_7144_v.json`
- Moderate: `scenario_idld_0018_v.json`, `scenario_idld_0389_v.json`, `scenario_idld_7602_v.json`
- Hard: `scenario_bal_teen_1158.json`, `scenario_idld_1204_v.json`, `scenario_idld_5331_v.json`

### Business/HR/Admin Support
- Easy: `scenario_bal_large2_1160.json`, `scenario_idld_4427_v.json`, `scenario_idld_7478_v.json`
- Moderate: `scenario_bal_large_0361.json`, `scenario_idld_1193_v.json`, `scenario_idld_5165_v.json`
- Hard: `scenario_bal_large2_0077.json`, `scenario_bal_large2_0317.json`, `scenario_idld_8544_v.json`

### Education (Teaching & Learning)
- Easy: `scenario_bal_large2_1949.json`, `scenario_idld_1509_v.json`, `scenario_idld_8697_v.json`
- Moderate: `scenario_bal_large2_0986.json`, `scenario_idld_2193_v.json`, `scenario_idld_3157_v.json`
- Hard: `scenario_idld_3469_v.json`, `scenario_idld_6302_v.json`, `scenario_idld_8537_v.json`

### Language
- Easy: `scenario_bal_teen_0846.json`, `scenario_idld_2116.json`, `scenario_idld_7790.json`
- Moderate: `scenario_bal_large2_0268.json`, `scenario_bal_teen_0035.json`, `scenario_idld_4498_v.json`
- Hard: `scenario_idld_0283_v.json`, `scenario_idld_2404_v.json`, `scenario_idld_3862.json`

### Mathematics
- Easy: `scenario_bal_large_0556.json`, `scenario_bal_self_0346.json`, `scenario_bal_teen_new_0279.json`
- Moderate: `scenario_bal_large_0231.json`, `scenario_bal_large_0425.json`, `scenario_idld_4934.json`
- Hard: `scenario_idld_4613.json`, `scenario_idld_5918_v.json`, `scenario_idld_6001_v.json`

### Medical/Nursing
- Easy: `scenario_bal_large2_1593.json`, `scenario_idld_3077_v.json`, `scenario_idld_3158_v.json`
- Moderate: `scenario_bal_large_0575.json`, `scenario_idld_2964_v.json`, `scenario_idld_6196_v.json`
- Hard: `scenario_idld_2076_v.json`, `scenario_idld_6067_v.json`, `scenario_idld_6835_v.json`

### Science
- Easy: `scenario_idld_0230.json`, `scenario_idld_5995.json`, `scenario_idld_8746.json`
- Moderate: `scenario_bal_large3_0160.json`, `scenario_bal_large_0353.json`, `scenario_bal_teen_new_0162.json`
- Hard: `scenario_bal_teen_new_0727.json`, `scenario_idld_2871.json`, `scenario_idld_3888_v.json`

### Service/Customer Support
- Easy: `scenario_bal_large_0383.json`, `scenario_bal_medium_0035.json`, `scenario_bal_teen_new_0071.json`
- Moderate: `scenario_idld_1553_v.json`, `scenario_idld_2744_v.json`, `scenario_idld_5750_v.json`
- Hard: `scenario_idld_0107_v.json`, `scenario_idld_3961_v.json`, `scenario_idld_7496_v.json`

### Social Studies
- Easy: `scenario_bal_self_new_0365.json`, `scenario_idld_1551_v.json`, `scenario_idld_1707.json`
- Moderate: `scenario_bal_large_0850.json`, `scenario_bal_teen_0590.json`, `scenario_idld_0677_v.json`
- Hard: `scenario_idld_0692.json`, `scenario_idld_2203_v.json`, `scenario_idld_7489.json`

### Software Development/IT
- Easy: `scenario_idld_0584_v.json`, `scenario_idld_3006_v.json`, `scenario_idld_3453_v.json`
- Moderate: `scenario_idld_4807_v.json`, `scenario_idld_5607_v.json`, `scenario_idld_8197_v.json`
- Hard: `scenario_idld_2086_v.json`, `scenario_idld_5536_v.json`, `scenario_idld_6846_v.json`
