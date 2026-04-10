"""Analysis: APM ridership model disconnect from reality.

PROBLEM:
- C14 predicts ~16k monthly riders
- Comparable CityBus routes: Route 10 (~21k/month), Route 13 (~20k/month)
- But APM is NEW mode competing with existing transit + cars
- Model assumes all potential demand converts → massive overestimate

ROOT CAUSES:
1. No mode share modeling - assumes 100% capture of accessible population
2. No competition from existing modes (cars, buses)
3. No frequency/headway sensitivity (APM vs bus treated same)
4. Jobs-based gravity overweights commute trips (not all trips are work-related)

FIXES NEEDED:
1. Mode share calibration: realistically 5-20% of potential switches to new mode
2. Frequency factor: higher frequency = higher ridership (but diminishing returns)
3. Competing modes: discount where existing bus serves similar corridor
4. Trip purpose mix: weight residential pop higher (more trip types)
5. Destination accessibility: not just origin access, but jobs reachable matters

IMPLEMENTATION:
- Add mode_share parameter to ridership predictions (calibrate ~10-15%)
- Add frequency_factor based on headway (APM assumed 5-10 min vs bus 20-30 min)
- Adjust enhanced gravity to 60% pop / 40% jobs (was 60/40 opposite)
- Add destination accessibility scoring for stops

EXPECTED OUTCOME:
- C14 ridership: ~16k → ~2-4k monthly (more realistic for new mode)
- Still higher than average bus due to speed/frequency advantages
- More plausible NPV calculations (currently all negative due to overestimated capex vs revenue)
"""

# Mode share factors by transit type
MODE_SHARE_NEW_RAIL = 0.12  # New rail transit typically captures 10-15% of potential
MODE_SHARE_BRT = 0.08  # BRT ~8-12%
MODE_SHARE_APM = 0.10  # APM ~8-12% (similar to BRT, between bus and rail)

# Frequency factors (relative to base bus service at 20-min headway)
def frequency_multiplier(headway_minutes: float, base_headway: float = 20.0) -> float:
    """Compute ridership boost from frequency improvement.
    
    Based on transit elasticity research: ~0.3-0.5 elasticity to frequency.
    frequency_factor = (base_headway / new_headway) ^ 0.4
    
    Examples:
    - 10 min headway vs 20 min base: 1.32x ridership
    - 5 min headway vs 20 min base: 1.74x ridership
    """
    return (base_headway / max(headway_minutes, 1.0)) ** 0.4
