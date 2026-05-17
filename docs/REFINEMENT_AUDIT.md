# Refinement Audit

Date: `2026-05-09`

This audit is based on the current project build, saved artifacts, and latest passing tests.

## Overall Status

The project is already a working end-to-end prototype.

What is already in place:
- real-data scraping
- cleaning and curation
- house-only modeling path
- CNN training
- grouped bedroom support model
- residential property-type support model
- bilingual NLP
- multimodal fusion
- marketing automation
- Flask dashboard
- automated tests

The project is **not** in a “build from scratch” stage anymore.
It is in a **refinement and enhancement stage**.

## Strong Areas

### 1. End-to-end integration
The pipeline is connected from data collection to recommendation and campaign generation.

### 2. Data workflow
The project now uses a real-data workflow and no longer depends on simulated data for the main house path.

### 3. Cleaning and curation
The house-only curation decision is coherent with the assignment direction and makes the project easier to defend academically.

### 4. Condition prediction
The vision model performs strongly on `condition`.

Current test property accuracy:
- `condition`: `0.917`

### 5. Grouped bedroom improvement
The grouped bedroom support model is much better than the earlier exact-bedroom setup.

Current grouped bedroom test property accuracy:
- `0.500`

### 6. Property-type support model
The auxiliary residential property-type model gives a good academic answer to the Module 2 property-type requirement.

Current test property accuracy:
- `0.647`

### 7. NLP module
The bilingual NLP module is stable and explainable.

Current internal NLP signal:
- `query_success_rate = 1.0`

### 8. Fusion and marketing
The fusion layer is no longer a basic weighted sum only.
It now includes:
- evidence-aware weight rebalancing
- stored reasons for recommendations
- campaign subject lines
- preview text
- call to action
- estimated engagement

### 9. Stability
The full current test suite passes.

## Weak Areas

### 1. Main multi-task bedroom head
The original exact-bedroom head inside the main house vision model is still weak.

Current test property accuracy:
- `cnn_bedroom_class`: `0.208`

This is acceptable only because the grouped bedroom support model now carries the bedroom story better.

### 2. Environment prediction
The main house model is weak on `environment`.

Current test property accuracy:
- `environment`: `0.250`

This means environment should be treated as a softer supporting signal, not a major bragging point.

### 3. Text score magnitude inside recommendation
The recommendation system works, but text scores are still much lower than structured scores on average.

Current fusion averages:
- structured: `0.9602`
- text: `0.2387`
- vision: `0.8783`

This suggests the recommender is still driven more by structured matching than by deep textual semantics.

### 4. Inventory imbalance
The current recommendation inventory is still heavily dominated by `Maseru`.

Current district counts:
- `Maseru`: `67`
- `Berea`: `20`
- `Leribe`: `2`
- `Quthing`: `2`
- `Mohale's Hoek`: `2`

This makes the recommender less balanced geographically.

### 5. Some raw titles are still noisy
Although presentation titles are cleaned in the UI, some raw listing titles remain messy in the stored inventory.

Example:
- `Beautiful, Highly Finished House for Sale_Serious Potential Buyers ONLY`

This is mostly a polish issue now, not a pipeline failure.

### 6. Marketing language quality is better, but still template-driven
The campaign generator is clean and useful, but still obviously template-based rather than fully natural.
That is acceptable for the assignment, but it is still a refinement target.

## Current Metrics Snapshot

### Main House Vision Model
- bedroom property accuracy: `0.208`
- environment property accuracy: `0.250`
- style property accuracy: `0.542`
- condition property accuracy: `0.917`

### Grouped Bedroom Support Model
- test property accuracy: `0.500`

### Residential Property-Type Support Model
- test property accuracy: `0.647`

### Recommendation
- properties considered: `93`
- clients profiled: `6`
- matches generated: `18`
- campaigns generated: `6`
- mean top match score: `0.7373`

### Fusion
- mean fusion reliability: `0.9268`
- average weights:
  - structured: `0.4917`
  - text: `0.2975`
  - vision: `0.2108`

### Marketing
- mean estimated engagement: `0.8607`

## Refinement Priority

### Priority 1
Strengthen the remaining weak recommendation logic around text-vs-structured balance.

Reason:
The system works, but the text side still contributes less strongly than it should for a “multimodal” story.

### Priority 2
Clean remaining raw title noise and presentation wording.

Reason:
This is low-risk polish that improves the lecturer-facing experience immediately.

### Priority 3
Improve campaign realism slightly more.

Reason:
Module 4 is already strong enough to demonstrate, so only small polish additions are justified now.

## What Should Not Be Prioritized Now

These should **not** be the focus unless there is spare time:
- more scraping for its own sake
- rebuilding the dashboard from scratch
- adding large new features
- retraining many models without a focused reason
- trying to force the weak exact-bedroom head to become the main bedroom solution again

## Recommended Next Steps

1. Refine the recommendation presentation layer and inventory wording.
2. Reduce noisy raw title leakage where it still appears.
3. Optionally strengthen the text contribution inside the recommender if it can be done safely.
4. Freeze the project once those small refinements are complete.

## Blunt Summary

The project is already strong enough to demonstrate.

The biggest remaining weaknesses are:
- the weak exact-bedroom head in the main CNN
- weak environment prediction
- text contribution inside recommendation still being smaller than structured contribution
- a little remaining title/presentation noise

Nothing here suggests the project is broken.
It suggests the project is close to finished and should now be handled carefully.
