# Feedback Loop Output Schema

This document defines the baseline export schema for:
- `data/processed/feedback_loop_results.csv`
- `data/processed/feedback_loop_diagnostics.csv`

These outputs are produced by `scripts/run_feedback_loop.py`.

## `feedback_loop_results.csv`

One row per `corridor_id` x `year`.

Core columns:
- `year`: simulation year step (e.g., 0, 5, 10, ...).
- `corridor_id`: corridor identifier.
- `daily_riders`: modeled daily ridership after awareness ramp.
- `base_riders`: pre-ramp ridership estimate.
- `awareness`: awareness/adoption factor for year.
- `apm_mode_share`: APM mode share from mode-choice stage.
- `directional_fraction`: directional trip factor.
- `pop_catchment`: catchment population.
- `jobs_catchment`: catchment jobs.
- `bus_headway`: corridor-specific parallel bus headway (minutes).

Convergence columns (Week 1-2):
- `ridership_rel_delta`: relative change vs previous modeled step.
- `new_pop_rel_delta`: relative change in modeled population growth vs previous step.
- `new_jobs_rel_delta`: relative change in modeled jobs growth vs previous step.
- `is_converged`: boolean convergence flag for row.
- `convergence_state`: one of `baseline`, `converged`, `not_converged`.

Development columns:
- `new_units`: delivered residential units in the step.
- `new_comm_sqft`: delivered commercial square feet in the step.
- `new_res_sqft`: delivered residential square feet in the step.
- `new_pop`: population increment from step development.
- `new_jobs`: jobs increment from step development.
- `capacity_draw_sqft`: total parcel capacity consumed by this corridor in the step.
- `parcels_with_delivery`: count of parcels that delivered non-zero sqft for this corridor in the step.

Equity columns (if enabled in run):
- `riders_SE01`, `riders_SE02`, `riders_SE03`
- `riders_per_1k_SE01`, `riders_per_1k_SE02`, `riders_per_1k_SE03`
- `low_income_access_ratio`

## `feedback_loop_diagnostics.csv`

One row per modeled `year` (run-level diagnostics).

Convergence and stop-control columns:
- `year`
- `n_corridors`
- `n_converged`
- `pct_converged`
- `max_ridership_rel_delta`
- `max_new_pop_rel_delta`
- `max_new_jobs_rel_delta`
- `all_converged`
- `divergence_flag`
- `converged_streak`
- `divergent_streak`
- `stop_triggered`
- `stop_reason` (empty, `adaptive_converged_stop`, or `divergence_stop`)

Capacity ledger columns (Week 3):
- `year_capacity_draw_sqft`: total sqft delivered in the year across all corridors.
- `capacity_total_sqft`: cumulative theoretical capacity tracked in ledger.
- `capacity_remaining_sqft`: remaining undeveloped capacity in ledger.
- `capacity_consumed_sqft`: cumulative consumed capacity in ledger.
- `capacity_consumed_pct`: consumed share (`capacity_consumed_sqft / capacity_total_sqft`).
- `parcels_with_capacity`: parcels with tracked capacity entries.
- `parcels_exhausted_capacity`: tracked parcels with near-zero remaining capacity.

## Notes

- Week 3 capacity depletion prevents repeated re-use of the same parcel capacity over time.
- If adaptive controls are disabled, `stop_triggered` remains `False` and `stop_reason` remains empty.
