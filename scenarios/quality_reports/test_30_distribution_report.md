# Test 30 Distribution Report

Created at: 2026-06-18T19:08:17

## Summary
- Full source split: `scenarios/test` (1202 scenarios)
- Immediate source subset: `scenarios/test_90` (90 scenarios)
- Target subset: `scenarios/test_30` (30 scenarios)
- Sampling seed: `42`
- Allocation: 10 domains × 3 difficulty levels × 1 scenario
- Sampling method: deterministic stratified random sampling within each domain-difficulty cell
- Nesting: every `test_30` scenario is also included in `test_90`

## Allocation Matrix

| Domain | Easy | Moderate | Hard | Total |
|---|---:|---:|---:|---:|
| AI | 1 | 1 | 1 | 3 |
| Business/HR/Admin Support | 1 | 1 | 1 | 3 |
| Education (Teaching & Learning) | 1 | 1 | 1 | 3 |
| Language | 1 | 1 | 1 | 3 |
| Mathematics | 1 | 1 | 1 | 3 |
| Medical/Nursing | 1 | 1 | 1 | 3 |
| Science | 1 | 1 | 1 | 3 |
| Service/Customer Support | 1 | 1 | 1 | 3 |
| Social Studies | 1 | 1 | 1 | 3 |
| Software Development/IT | 1 | 1 | 1 | 3 |
| **Total** | **10** | **10** | **10** | **30** |

## Source Availability Per Cell

| Domain | Full Test Easy | Full Test Moderate | Full Test Hard | Test 90 Easy | Test 90 Moderate | Test 90 Hard |
|---|---:|---:|---:|---:|---:|---:|
| AI | 12 | 14 | 15 | 3 | 3 | 3 |
| Business/HR/Admin Support | 12 | 15 | 17 | 3 | 3 | 3 |
| Education (Teaching & Learning) | 14 | 17 | 18 | 3 | 3 | 3 |
| Language | 80 | 82 | 76 | 3 | 3 | 3 |
| Mathematics | 85 | 79 | 70 | 3 | 3 | 3 |
| Medical/Nursing | 12 | 15 | 15 | 3 | 3 | 3 |
| Science | 81 | 77 | 78 | 3 | 3 | 3 |
| Service/Customer Support | 13 | 14 | 15 | 3 | 3 | 3 |
| Social Studies | 77 | 77 | 80 | 3 | 3 | 3 |
| Software Development/IT | 11 | 15 | 16 | 3 | 3 | 3 |

## Variant Type Counts

| Variant type | Count |
|---|---:|
| balance_large_scale | 3 |
| balance_large_scale_v2 | 3 |
| balance_large_scale_v3 | 1 |
| balance_teen_filtered | 3 |
| balance_teen_k12 | 2 |
| context_variant | 14 |
| idld_aligned | 4 |

## Selected Files By Cell

### AI
- Easy: `scenario_idld_7144_v.json`
- Moderate: `scenario_idld_0018_v.json`
- Hard: `scenario_bal_teen_1158.json`

### Business/HR/Admin Support
- Easy: `scenario_idld_7478_v.json`
- Moderate: `scenario_idld_1193_v.json`
- Hard: `scenario_bal_large2_0077.json`

### Education (Teaching & Learning)
- Easy: `scenario_bal_large2_1949.json`
- Moderate: `scenario_bal_large2_0986.json`
- Hard: `scenario_idld_8537_v.json`

### Language
- Easy: `scenario_bal_teen_0846.json`
- Moderate: `scenario_idld_4498_v.json`
- Hard: `scenario_idld_3862.json`

### Mathematics
- Easy: `scenario_bal_teen_new_0279.json`
- Moderate: `scenario_bal_large_0231.json`
- Hard: `scenario_idld_6001_v.json`

### Medical/Nursing
- Easy: `scenario_idld_3077_v.json`
- Moderate: `scenario_bal_large_0575.json`
- Hard: `scenario_idld_2076_v.json`

### Science
- Easy: `scenario_idld_0230.json`
- Moderate: `scenario_bal_large3_0160.json`
- Hard: `scenario_bal_teen_new_0727.json`

### Service/Customer Support
- Easy: `scenario_bal_teen_new_0071.json`
- Moderate: `scenario_idld_5750_v.json`
- Hard: `scenario_idld_0107_v.json`

### Social Studies
- Easy: `scenario_idld_1707.json`
- Moderate: `scenario_bal_large_0850.json`
- Hard: `scenario_idld_7489.json`

### Software Development/IT
- Easy: `scenario_idld_3453_v.json`
- Moderate: `scenario_idld_8197_v.json`
- Hard: `scenario_idld_6846_v.json`
