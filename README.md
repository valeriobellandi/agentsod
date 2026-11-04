# AgentSoD / SoD-RM

Prototype of the reference monitor described in

> V. Bellandi, *Towards Collusion-Resistant Separation of Duties for Multi-Agent LLM Systems*,
> AI-SEC 2027 (workshop at AINA 2027), submitted.

It implements the separation conditions (Sect. 4 of the paper), the AgentSoD policy
language and the decision procedure of Algorithm 1 with its hash-chained log (Sect. 5),
and the evaluation of Sect. 6. It is the deterministic enforcement core only: no LLM is
involved, and experiments with real agents are future work.

## Layout

| Path | Content |
|---|---|
| `sodrm/model.py` | Principals, substrates, attenuating delegations, events, effective principals, provenance labels |
| `sodrm/policy.py` | AgentSoD data structures and parser |
| `sodrm/monitor.py` | Decision point (Algorithm 1): levels 0–3, `require n separated ... before` clauses bound to artifact versions, single-use approvals, pairing history, one critical section per decision |
| `sodrm/ledger.py` | Hash-chained log, chain verification, recomputation of decisions from stored witnesses |
| `sodrm/reference.py` | Reference oracle: a from-scratch, non-incremental implementation of the definitions |
| `policies/energy_ops.sod` | Policy used in the tests and benchmarks (excerpt in Listing 1.1) |
| `tests/test_sodrm.py` | Targeted tests |
| `bench/run.py` | Synthetic workloads (monitor vs. oracle) and latency measurements |
| `bench/plot_lncs.py` | Fig. 2 of the paper |
| `experiments/gridops/` | First experiment with LLM agents (GridOps-SoD), see its README |
| `results/` | Results reported in the paper (`summary.json`, `latency_*.csv`, `fig2_latency.pdf`; `gridops/` for Table 2) |

## Reproducing the results

```bash
pip install pytest matplotlib
python -m pytest -q                         # targeted tests (+ pandapower for the experiment)
python bench/run.py                         # synthetic workloads and latency (~1 min)
python bench/plot_lncs.py results/fig2.pdf  # Fig. 2
```

Workloads are generated with fixed seeds (7 for the random traces, 17 for the two-party
approval traces, 11 and 13 for the latency runs), so `summary.json` is reproduced exactly;
latencies depend on the machine (the paper used one core of a 2.1 GHz Xeon, Python 3.13).

## Known simplifications

- Event times are assumed monotone (needed for window-based pruning).
- A pairing is recorded for each conflicting earlier event an accepted event was checked
  against; the bound κ applies to the pairings recorded before the current event.
- `require n` clauses check pairwise separation among approvals without pairing history.
- Atomicity uses one global lock.
- The monitor checks the version reference carried by the call; checking that the bytes
  passed to the tool hash to that reference is left to the proxy (not modelled).
- No crash recovery: exactly-once execution would need idempotent tool calls and
  persistent recovery state.
- Timings cover the decision core with an in-memory log, excluding proxy, persistence
  and contention.

## License

MIT, see `LICENSE`.
