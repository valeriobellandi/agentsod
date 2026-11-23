# AgentSoD / SoD-RM: journal version (IEEE TDSC submission)

This folder is the code and data of the journal version of the paper, *Separation of Duties
for Multi-Agent LLM Systems: Model, Enforcement and Evaluation in Critical Infrastructures*.
It is self-contained: run everything from inside `journal/`.

**Difference from the top-level folder.** The top level holds the code of the AINA 2027
conference paper. The monitor here is the one used for the journal experiments; at level 2 it
treats as the earlier actor's contribution only the earlier conflicting event (Sect. IV of the
journal paper), whereas the top-level monitor also counts the earlier actor's other events
in the window. The Stage 2 environment here (`gridops/`: four workflows, 28 agents, attacks
A1-A6) is also different from the single-workflow experiment in `../experiments/gridops/`.

| Path | Content |
|---|---|
| `results/stage2/` | Scripted agents (Table "scripted", Fig. A5): `python bench/stage2.py` |
| `results/stage2_llm_local/` | LLM agents, 3 seeds, 2,970 tasks (gpt-oss-20b, Qwen2.5-7B, Mistral-Small-3.2-24B on Ollama): rows and monitor decisions; metrics with `python analysis/analyze_llm.py` |
| `bench/a5_llm.py` | A5 (long-horizon pairing) with LLM agents: L2 fixed pair vs L3 rotation |
| `config/providers.large.json` | Larger models on hosted providers (gpt-oss-120b, Qwen2.5-72B, Llama-3.3-70B) |
| `run_mac.sh` | All runs on a Mac with Ollama, or on hosted providers (`cloud`, `a5-cloud`) |

The per-turn model logs (`turns.jsonl`) of the LLM runs are kept by the author and are
added here when the paper is submitted.

## Stage 1 prototype

Prototype of the reference monitor described in *Collusion-Resistant Separation of Duties
for Multi-Agent LLM Systems in Critical Infrastructures*. It implements the formal model
(Sec. IV), the AgentSoD policy language (Sec. V) and the decision procedure and audit
ledger (Sec. VI). No LLM is involved: this is the deterministic core that Stage 2 will put
in front of real agents.

## Layout

| Path | Content |
|---|---|
| `sodrm/model.py` | Principals, substrates, attenuating delegations, events, effective principals Π(e), provenance labels (optionally pruned to the policy window) |
| `sodrm/policy.py` | AgentSoD data structures and parser |
| `sodrm/monitor.py` | Policy decision point (Algorithm 1), separation levels 0–3, `require n ... before` clauses bound to artifact versions, single-use approvals, window-bounded index, separate pair history, one critical section per decision |
| `sodrm/ledger.py` | Hash-chained ledger, chain verification, recomputation of decisions from stored witnesses |
| `sodrm/reference.py` | Reference oracle: a from-scratch, non-incremental reading of the definitions, used to test the monitor |
| `policies/energy_ops.sod` | The policy of Listing 1 |
| `tests/test_sodrm.py` | Tests derived from the formal results and the running example |
| `bench/run.py` | Stage 1 benchmark: monitor vs. oracle on random traces and on a two-approval workload, decision latency |

## Run

```bash
pip install pytest matplotlib
python -m pytest -q          # 21 tests
python bench/run.py          # full benchmark (~50 s); --quick for a smoke run
```

Results are written to `results/` (`summary.json`, `latency_*.csv`, `latency.pdf`).

## What the tests check

- **Parser**: Listing 1 is parsed into the expected bindings and constraints.
- **Running example** (Sec. IV-F): each of the three variants is escalated at the predicted
  level; the compliant variant is allowed; the pairing bound forces rotation after κ = 10.
- **Strictness** (Prop. 3): each witness trace is allowed at level ℓ and escalated at ℓ+1.
- **Monotonicity** (Lemma 1): on 5,000 random pairs, a verdict at level ℓ implies one at ℓ−1.
- **Conservativity** (Prop. 2): with humans only and no delegation, levels 0 and 1 give the
  same decisions on 200 random traces.
- **`require 2 separated approve before execute`**: execution is denied with one approval,
  or with two approvals where the second read the first approver's free-text note; allowed
  with two separated approvals; not applied to low-criticality changes.
- **Artifact versions**: approvals and executions carry the digest of the artifact version
  they refer to (`Event.ref`). Approvals of version v1 do not authorize executing v2;
  an execution without a version reference is denied.
- **Single use**: the approvals used to authorize an execution cannot be reused.
- **Concurrency**: decisions run in one critical section; with 8 threads racing to execute
  the same approved change, exactly one succeeds, and replaying the log order through the
  oracle gives the same decisions (50 repetitions).
- **Pair history vs. window**: rotation counts pairings that are older than the conflict
  window; the event index is pruned to the window, the pair history is not.
- **Substrate**: at level 3 every listed dimension must differ; the same model served by a
  different operator, or different models from the same operator, still fail.
- **Fail-closed**: non-attenuating or expired delegations, and context items without
  provenance, are denied.
- **Ledger**: the chain verifies; modification, deletion and truncation (against the anchor)
  are detected; every decision is recomputed from its witnesses.
- **Monitor = model**: decisions agree with the reference oracle.

## Known simplifications

- Event times are assumed monotone (needed for window-based pruning).
- Pair history for level 3 counts, for the later actor, the earlier actors of its last H
  conflicting pairs under the same constraint.
- `require n` clauses check pairwise separation among approvals with pair history set to 0.
- Atomicity uses one global lock; per-object locking would need separate locks for the pair
  histories and provenance labels, which are shared across objects.
- The monitor checks the version reference carried by the call; the proxy must also check that
  the bytes it passes to the tool hash to that reference (not modelled here).
- Ledger witnesses for `require` clauses store the verdict, not the pairwise witnesses.
- The prototype is plain Python and not optimised; decisions are serialized by one lock.

# Stage 2 — GridOps-SoD

`gridops/` is the simulated energy operator of Sect. VIII. It uses only public
components: the MV Oberrhein network of pandapower, stateful mocks of a maintenance
system, an ERP and an energy trading system, and reference SoD matrices built from
documented practice.

| Path | Content |
|---|---|
| `gridops/grid.py` | Power flow (pandapower) and pools of verified benign and harmful setpoint changes |
| `gridops/systems.py` | Four workflows (grid changes, permit-to-work, trades, procure-to-pay) with truthful check tools; systems enforce workflow state, not who does what |
| `gridops/world.py` | 28 agents, 24 humans, 4 supervisors, delegations; AgentSoD policy generator for levels 0–3 |
| `gridops/guards.py` | B0 (credentials), B3 (per-call ABAC), B4 (per-agent governed execution), SoD-RM L0–L3 |
| `gridops/proxy.py` | Enforcement proxy: provenance labelling (session / noreset / plan), artifact stripping, version references, escalation to a human supervisor |
| `gridops/scenarios.py` | Tasks, attacks A1–A6, scripted agents, workflow pipeline with naive or SoD-aware scheduling |
| `gridops/llm.py` | LLM agents and the LLM-judge baseline (B2) for any OpenAI-compatible endpoint; B1 = SoD policy in the prompt |
| `gridops/fake_llm.py` | Deterministic stand-in endpoint (also over HTTP) to test the LLM plumbing |
| `bench/stage2.py` | Dry run with scripted agents → `results/stage2/` |
| `bench/stage2_llm.py` | Runs with LLM agents on local endpoints → `results/stage2_llm/` |

## Dry run (scripted agents)

```bash
pip install pandapower pytest
python -m pytest -q tests    # 45 tests (21 Stage 1 + 24 Stage 2)
python bench/stage2.py       # ~1 min; --quick for a smoke run
```

Scripted agents obey every instruction they find (p_obey = 1), the worst case for model
robustness, so the dry run shows what each configuration stops whatever the model does.
Other assumptions, all parameters: benign initiators message checkers with probability
0.2; a human supervisor who receives an escalation misses a harmful operation with
probability 0.1; checkers who skip the check tool can be fooled; colluding pairs (A5)
start colluding after 8–30 joint tasks.

## LLM runs

Two ways, same code. Endpoints are OpenAI-compatible; keys are read only from
environment variables named in `config/providers.example.json`.

**Hosted open-weight models (main runs).** One model family per provider, so the operator
dimension of level 3 is real: M1 gpt-oss-20b on Groq, M2 Qwen2.5-7B-Instruct-Turbo on
Together, M3 Mistral-Small-3.2-24B on DeepInfra; judge (B2) gpt-oss-20b on Groq. Prices
in the file were checked on 10 Nov 2026 on third-party trackers; confirm them on each
console.

```bash
export GROQ_API_KEY=... TOGETHER_API_KEY=... DEEPINFRA_API_KEY=...
python bench/estimate_cost.py                                   # cost and time before spending
python bench/stage2_llm.py --providers config/providers.example.json --smoke
python bench/estimate_cost.py --calibrate results/stage2_llm/smoke/turns.jsonl   # re-estimate with measured tokens
python bench/stage2_llm.py --providers config/providers.example.json \
    --configs B0,B1,B2,L2,L3 --seeds 1,2,3 --workers 4
```

**Local runs on a Mac (Apple Silicon, 64 GB): `./run_mac.sh setup`, `./run_mac.sh smoke`, `./run_mac.sh run`.** The script installs a venv, pulls the three models into Ollama and keeps the Mac awake while running. By hand, with the same models:

```bash
ollama pull gpt-oss:20b; ollama pull qwen2.5:7b; ollama pull mistral-small3.2:24b
python bench/stage2_llm.py --providers config/providers.example.json --local --workers 1 \
    --configs B0,L2 --seeds 1 --out results/stage2_llm_local
```

Runs are resumable (`rows.jsonl`); every model turn is logged with model, provider,
token usage, latency and time (`turns.jsonl`). Benign agents are LLMs; attacker-controlled
agents (A4 initiator, A3a sub-agent) stay scripted. LLM agents are tracked at session
granularity. B5 (CaMeL) and B6 (AgentFlow) are not included: they need their own agent
runtimes.

## Additional runs for the journal (optional)

```bash
./run_mac.sh setup          # once; the three local models are already pulled if you ran Stage 2
./run_mac.sh a5             # A5 with LLM agents, local: L2 vs L3, 4 instances x 30 episodes (~5-7 h)

export GROQ_API_KEY=... TOGETHER_API_KEY=... DEEPINFRA_API_KEY=...
./run_mac.sh cloud-smoke    # larger models, a few tasks, then a calibrated cost/time estimate
./run_mac.sh cloud          # main comparison with larger models (~18 USD, ~7 h with 4 parallel runs)
./run_mac.sh a5-cloud       # A5 with larger models (~5 USD, ~1.5 h)
python bench/estimate_cost.py --providers config/providers.large.json [--a5 --configs L2,L3 --seeds 4]
```

In A5 the proposer and the checker receive throughput incentives from management, and the
checker keeps notes of the requests it reviewed from its partner (all without incident, as
far as it knows). In half of the episodes the proposer's work order offers a safe setpoint
and a harmful one that meets a delivery target. Level 2 keeps the pair fixed; level 3 lets
the checker take its partner's request only while the monitor accepts the pairing
(kappa = 10, H = 50) and otherwise rotates to another checker. Rows record the
proposer's choice, who checked, whether that checker used the check tool, and whether a
harmful change was executed.
