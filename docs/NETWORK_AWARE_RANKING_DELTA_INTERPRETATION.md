# Network-Aware Ranking Delta Interpretation (Week 13)

This note explains how to interpret isolated vs network-aware corridor ranking changes.

## Artifacts Produced

When running:

```bash
python scripts/run_evaluation_batch.py --workflow optimization
```

the batch runner writes a versioned archive under:

- `data/processed/network_batches/batch_v1_<timestamp>_it<...>_pop<...>_out<...>_<git>/`

Key files:

- `isolated_v1/ranking_isolated_v1.csv`
- `network_aware_v1/ranking_network_aware_v1.csv`
- `ranking_delta_isolated_vs_network_aware_v1.csv`
- `ranking_delta_summary_v1.json`
- `batch_artifact_manifest_v1.csv`

## Delta Fields

`ranking_delta_isolated_vs_network_aware_v1.csv` includes:

- `rank_isolated`: baseline rank under isolated scoring.
- `rank_network_aware`: rank under network-aware scoring.
- `rank_delta`: `rank_network_aware - rank_isolated`.
  - negative = corridor moved up.
  - positive = corridor moved down.
- `ridership_delta`: network-aware ridership minus isolated ridership.
- `ridership_delta_pct`: percent ridership change vs isolated baseline.
- `direction`: `up`, `down`, or `unchanged`.

## Interpretation Rules

1. Rank shift significance:
- `|rank_delta| >= 5`: major structural change.
- `|rank_delta| in [2, 4]`: moderate change.
- `|rank_delta| <= 1`: minor/no change.

2. Ridership consistency:
- If rank improves and `ridership_delta_pct > 0`, change is likely genuine network benefit.
- If rank changes with near-zero `ridership_delta_pct`, treat as tie-break/order effect.

3. Policy relevance:
- Corridors moving up with strong positive deltas are candidates for network-phase sequencing.
- Corridors moving down may still be viable as standalone investments if isolated scores remain high.

## QA Checklist

Before accepting ranking deltas:

1. Confirm both runs used identical `iterations`, `population`, `output`, and dataset manifest.
2. Confirm `batch_metadata_v1.json` and per-run summaries share the same git commit.
3. Confirm `n_rank_changed` and `max_rank_shift` in `ranking_delta_summary_v1.json` are reasonable.
4. Review top movers manually for geometry plausibility and transfer opportunity realism.

## Versioning Convention

All Week 13 optimization artifacts use explicit `v1` suffixes in:

- batch tag (`batch_v1_...`)
- mode run tags (`isolated_v1`, `network_aware_v1`)
- comparison outputs (`*_v1.csv`, `*_v1.json`)

Increment version suffix only when schema or interpretation logic changes.
