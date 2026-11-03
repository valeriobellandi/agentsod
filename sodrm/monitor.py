"""SoD-RM policy decision point (Algorithm 1) with incremental state.

State kept by the monitor, and how long it is kept:
  - index of accepted events per (constraint, object): pruned to the constraint window W;
  - provenance labels: optionally pruned to the largest window (see Provenance);
  - pair histories for level 3: the last H pairings of each later actor, per constraint,
    kept independently of W (a pairing may be older than the window);
  - consumed approvals for `require n ... before` clauses (single use).
Each decision, including the updates above and the log append, runs in one critical
section, so the log order is a valid sequential order of all decisions.
"""
from __future__ import annotations

import threading

from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Deque, Dict, List, Optional, Tuple

from .ledger import Ledger
from .model import Event, Provenance, Registry
from .policy import SEVERITY, Constraint, Policy


def obj_type(obj: str) -> str:
    return obj.split("(", 1)[0]


@dataclass
class Decision:
    decision: str
    witnesses: List[dict] = field(default_factory=list)


def substrate_ok(reg: Registry, a_i: str, a_j: str, dims) -> Tuple[bool, dict, dict]:
    p_i, p_j = reg.principals[a_i], reg.principals[a_j]
    s_i, s_j = p_i.substrate(), p_j.substrate()
    if p_i.kind != "agent" or p_j.kind != "agent":
        return True, s_i, s_j  # humans are distinct substrates by construction
    ok = all(s_i[d] is not None and s_i[d] != s_j[d] for d in dims)
    return ok, s_i, s_j


def check_pair(c: Constraint, level: int, e_i: Event, e_j: Event, pi_i, pi_j,
               influenced, exempt, reg: Registry, pair_count: int) -> dict:
    """Evaluate e_i _|_level e_j in increasing cost order; return a witness."""
    w = {"constraint": c.name, "on_violation": c.on_violation, "level": level,
         "earlier": e_i.eid, "later": e_j.eid,
         "actor_i": e_i.actor, "actor_j": e_j.actor}
    # level 0
    if e_i.actor == e_j.actor:
        w.update(verdict=False, failed_at=0)
        return w
    if level >= 1:
        shared = sorted((set(pi_i) & set(pi_j)) - set(exempt))
        w.update(pi_i=sorted(pi_i), pi_j=sorted(pi_j), shared=shared)
        if shared:
            w.update(verdict=False, failed_at=1)
            return w
    if level >= 2:
        if influenced is None:
            w.update(influenced=None, verdict=False, failed_at=2)  # unknown provenance
            return w
        extra = sorted(set(influenced) - ({e_i.artifact} if e_i.artifact else set()))
        w.update(influenced=sorted(influenced), artifact=e_i.artifact, extra=extra)
        if extra:
            w.update(verdict=False, failed_at=2)
            return w
    if level >= 3:
        ok, s_i, s_j = substrate_ok(reg, e_i.actor, e_j.actor, c.dims)
        w.update(substrate_i=s_i, substrate_j=s_j, dims=list(c.dims),
                 kind_i=reg.principals[e_i.actor].kind, kind_j=reg.principals[e_j.actor].kind,
                 pair_count=pair_count, kappa=c.kappa, history=c.history)
        if not ok or pair_count > c.kappa:
            w.update(verdict=False, failed_at=3)
            return w
    w.update(verdict=True, failed_at=None)
    return w


class Monitor:
    def __init__(self, policy: Policy, registry: Registry, provenance: Optional[Provenance] = None,
                 ledger: Optional[Ledger] = None) -> None:
        self.policy = policy
        self.reg = registry
        self.prov = provenance or Provenance()
        self.ledger = ledger or Ledger()
        self.pi: Dict[str, frozenset] = {}
        # index: (constraint, object) -> accepted events relevant to that constraint
        self.index: Dict[Tuple[str, str], List[Event]] = defaultdict(list)
        # pair history: (constraint, later actor) -> deque of earlier actors (last H pairs)
        self.hist: Dict[Tuple[str, str], Deque[str]] = {}
        self.accepted: List[Event] = []
        # approvals already used to authorize a dependent operation: constraint -> eids
        self.consumed: Dict[str, set] = defaultdict(set)
        self._lock = threading.Lock()

    def _relevant(self, c: Constraint, op: str) -> bool:
        return op in c.conflict or op == c.require_op

    def _pairs(self, c: Constraint, a: str) -> Deque[str]:
        key = (c.name, a)
        if key not in self.hist:
            self.hist[key] = deque(maxlen=max(c.history, 1))
        return self.hist[key]

    def decide(self, e: Event) -> Decision:
        """Decide on e atomically: checks, state updates and log append are one step."""
        with self._lock:
            return self._decide(e)

    def _decide(self, e: Event) -> Decision:
        attrs = dict(e.attrs)
        ot = obj_type(e.obj)
        exempt = self.policy.exempt
        pi_j = self.reg.effective_principals(e)
        if pi_j is None:
            d = Decision("deny", [{"invalid_authority": True, "later": e.eid}])
            self.ledger.append(e, d.decision, d.witnesses, self.policy.digest())
            return d
        dec = "allow"
        wits: List[dict] = []
        pending_pairs: List[Tuple[Constraint, Event]] = []
        grants: List[Tuple[Constraint, List[str]]] = []
        for c in self.policy.constraints:
            if not c.applies(ot, attrs):
                continue
            key = (c.name, e.obj)
            if key in self.index:
                # events older than the window are never checked again (times are monotone)
                self.index[key] = [x for x in self.index[key] if e.t - x.t <= c.window]
            if c.conflict and e.op in c.conflict:
                for e_i in self.index[(c.name, e.obj)]:
                    if not c.conflicts(e_i.op, e.op) or e.t - e_i.t > c.window:
                        continue
                    infl = self.prov.influenced_by(e.context, e_i.eid) if c.level >= 2 else frozenset()
                    cnt = sum(1 for a in self._pairs(c, e.actor) if a == e_i.actor) if c.level >= 3 else 0
                    w = check_pair(c, c.level, e_i, e, self.pi[e_i.eid], pi_j, infl, exempt, self.reg, cnt)
                    wits.append(w)
                    pending_pairs.append((c, e_i))
                    if not w["verdict"] and SEVERITY[c.on_violation] > SEVERITY[dec]:
                        dec = c.on_violation
            if c.require_n and e.op == c.before_op:
                # only unused approvals of the exact artifact version the operation refers to
                apps = [a for a in self.index[(c.name, e.obj)]
                        if a.op == c.require_op and e.t - a.t <= c.window
                        and e.ref is not None and a.ref == e.ref
                        and a.eid not in self.consumed[c.name]]
                ok, w = self._find_independent(c, apps)
                w.update(constraint=c.name, on_violation=c.on_violation, later=e.eid, ref=e.ref)
                wits.append(w)
                if ok:
                    grants.append((c, w["approvals"]))
                elif SEVERITY[c.on_violation] > SEVERITY[dec]:
                    dec = c.on_violation
        self.ledger.append(e, dec, wits, self.policy.digest())
        if dec in ("allow", "audit_only"):
            self._accept(e, pi_j, pending_pairs)
            for c, used in grants:
                self.consumed[c.name].update(used)  # approvals are single use
        return Decision(dec, wits)

    def _find_independent(self, c: Constraint, apps: List[Event]) -> Tuple[bool, dict]:
        apps = sorted(apps, key=lambda a: a.t)
        for combo in combinations(apps, c.require_n):
            good = True
            for x, y in combinations(combo, 2):
                infl = self.prov.influenced_by(y.context, x.eid) if c.level >= 2 else frozenset()
                cnt = 0
                w = check_pair(c, c.level, x, y, self.pi[x.eid], self.pi[y.eid], infl,
                               self.policy.exempt, self.reg, cnt)
                if not w["verdict"]:
                    good = False
                    break
            if good:
                return True, {"require": c.require_n, "approvals": [a.eid for a in combo], "verdict": True}
        return False, {"require": c.require_n, "approvals": [a.eid for a in apps], "verdict": False}

    def _accept(self, e: Event, pi_j, pending_pairs) -> None:
        self.pi[e.eid] = pi_j
        self.prov.record_outputs(e)
        self.accepted.append(e)
        ot = obj_type(e.obj)
        for c in self.policy.constraints:
            if c.applies(ot, dict(e.attrs)) and self._relevant(c, e.op):
                self.index[(c.name, e.obj)].append(e)
        for c, e_i in pending_pairs:
            if c.level >= 3:
                self._pairs(c, e.actor).append(e_i.actor)
