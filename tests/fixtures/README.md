Behavioral Scenario Fixtures

These JSON files are synthetic histories used by integration tests in
`tests/integration/test_behavioral_scenarios.py`. They simulate long-horizon
user behavior to validate feature dynamics and phase probability trends.

Fixtures

- `history_tectonic_shift.json`: Stable usage with a brief exploration spike,
  followed by stabilization in a new cluster.
- `history_relentless_explorer.json`: Persistent high-variance exploration.
- `history_autopilot_lock_in.json`: Low-variance, high model usage (daemon/ideal)
  leading to stable lock-in.
- `history_context_oscillation.json`: Alternating day/night contexts to stress
  variability and state detection.
