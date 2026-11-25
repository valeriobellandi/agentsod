"""Enforcement configurations compared in Stage 2.

B0  credentials only: an agent may call every tool its platform credentials reach.
B3  per-call ABAC: the actor must hold, through a valid delegation chain, the
    authority for the operation; no history.
B4  governed execution per agent (after Jing): B3 plus a gate that remembers the
    agent's own past operations and refuses a conflicting one on the same object;
    other agents' history is not visible to it.
SoD SoD-RM at a given level, with the AgentSoD policy of world.policy_text.
B2  LLM judge: lives in llm.py, since it needs a model endpoint.
B1 (policy in the prompt) changes only agent prompts, so it is a property of the
agent backend, not a guard; with scripted agents it coincides with B0.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set, Tuple

from sodrm.model import Event, Provenance, Registry
from sodrm.monitor import Monitor
from sodrm.policy import Policy

from .world import conflict_sets


class Guard:
    name = "guard"
    uses_provenance = False

    def decide(self, e: Event) -> str:
        raise NotImplementedError

    def would(self, e: Event) -> str:
        return "allow"


class NoGuard(Guard):
    name = "B0"

    def decide(self, e: Event) -> str:
        return "allow"


class ABACGuard(Guard):
    name = "B3"

    def __init__(self, reg: Registry) -> None:
        self.reg = reg

    def decide(self, e: Event) -> str:
        return "allow" if self.reg.effective_principals(e) is not None else "deny"

    would = decide


class GovExecGuard(ABACGuard):
    name = "B4"

    def __init__(self, reg: Registry) -> None:
        super().__init__(reg)
        self.own: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # (agent, obj) -> ops
        self.conf = conflict_sets()

    def _check(self, e: Event) -> str:
        if super().decide(e) == "deny":
            return "deny"
        ops = self.conf.get(e.obj.split("(", 1)[0], ())
        done = self.own[(e.actor, e.obj)]
        if e.op in ops and any(o in ops and o != e.op for o in done):
            return "deny"
        return "allow"

    def decide(self, e: Event) -> str:
        d = self._check(e)
        if d == "allow":
            self.own[(e.actor, e.obj)].add(e.op)
        return d

    def would(self, e: Event) -> str:
        return self._check(e)


class SoDGuard(Guard):
    uses_provenance = True

    def __init__(self, policy: Policy, reg: Registry, prov: Provenance, level: int) -> None:
        self.monitor = Monitor(policy, reg, prov)
        self.name = f"L{level}"

    def decide(self, e: Event) -> str:
        d = self.monitor.decide(e)
        # failing witnesses, for the diagnostic log only (agents never see them)
        self.last_why = [w for w in d.witnesses if not w.get("ok", True)] or d.witnesses[:2]
        return d.decision

    def would(self, e: Event) -> str:
        return self.monitor.would_decide(e)
