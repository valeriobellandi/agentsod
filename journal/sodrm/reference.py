"""Reference oracle: a from-scratch, non-incremental reading of the definitions of Sect. 4.

It shares no state or helper with the monitor except the data classes and the
policy parser. For every incoming event it recomputes effective principals,
provenance labels and pair histories by replaying the accepted prefix.
It is deliberately slow; it exists to check that the monitor implements the model.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set

SEV = {"allow": 0, "audit_only": 1, "escalate": 2, "deny": 3}


def _covers(scope, op, obj) -> bool:
    return any(o == op and (b == obj or b == "*") for (o, b) in scope)


def ref_pi(reg, e) -> Optional[Set[str]]:
    if not e.chain:
        return {e.actor} if _covers(reg.own_scope.get(e.actor, ()), e.op, e.obj) else None
    ps = {e.actor}
    held = None
    prev = None
    for k, did in enumerate(e.chain):
        d = reg.delegations.get(did)
        if d is None:
            return None
        if k == 0:
            held, prev = reg.own_scope.get(d.grantor, frozenset()), d.grantor
        if d.grantor != prev or d.t_start > e.t or d.t_end < e.t:
            return None
        for (o, b) in d.scope:
            if not _covers(held, o, b):
                return None
        held, prev = d.scope, d.grantee
        ps |= {d.grantor, d.grantee}
    if prev != e.actor or not _covers(held, e.op, e.obj):
        return None
    return ps


def ref_labels(accepted, sources: Dict[str, FrozenSet[str]], verifiers=frozenset()) -> Dict[str, Set[str]]:
    lab: Dict[str, Set[str]] = {k: set(v) for k, v in sources.items()}
    arts: Set[str] = set()
    for e in accepted:
        inh: Set[str] = set()
        declass = e.op in verifiers and set(e.context) <= arts
        for d in ([] if declass else e.context):
            inh |= lab.get(d, set())
        if e.artifact:
            arts.add(e.artifact)
        out = inh | {e.eid}
        for d in set(e.outputs) | ({e.artifact} if e.artifact else set()):
            lab[d] = set(out)
    return lab


def _otype(obj: str) -> str:
    return obj.split("(", 1)[0]


def _independent(c, reg, exempt, x, y, pis, lab, count, contrib=None) -> bool:
    contrib = contrib if contrib is not None else {x.eid}
    if x.actor == y.actor:
        return False
    if c.level >= 1 and (pis[x.eid] & pis[y.eid]) - set(exempt):
        return False
    if c.level >= 2:
        for d in y.context:
            if d not in lab:
                return False
            if (lab[d] & contrib) and d != x.artifact:
                return False
    if c.level >= 3:
        px, py = reg.principals[x.actor], reg.principals[y.actor]
        if px.kind == "agent" and py.kind == "agent":
            for dim in c.dims:
                vx, vy = getattr(px, dim), getattr(py, dim)
                if vx is None or vx == vy:
                    return False
        if count > c.kappa:
            return False
    return True


def _conflict_pairs(c, accepted, upto=None):
    """All (x, y) accepted, same object, conflicting under c, within window, x before y."""
    seq = accepted if upto is None else accepted[:upto]
    out = []
    for j, y in enumerate(seq):
        if not (c.applies(_otype(y.obj), dict(y.attrs)) and y.op in c.conflict):
            continue
        for x in seq[:j]:
            if (x.obj == y.obj and c.applies(_otype(x.obj), dict(x.attrs)) and x.op in c.conflict
                    and x.op != y.op and y.t - x.t <= c.window):
                out.append((x, y))
    return out


def _contrib(accepted, x, t, window) -> Set[str]:
    """Events of x's actor within the window (the earlier actor's contributions, level 2)."""
    return {z.eid for z in accepted if z.actor == x.actor and 0 <= t - z.t <= window} | {x.eid}


def _first_combo(c, reg, policy, apps, pis, lab, accepted=()):
    apps = sorted(apps, key=lambda a: a.t)
    for combo in combinations(apps, c.require_n):
        if all(_independent(c, reg, policy.exempt, x, y, pis, lab, 0,
                            _contrib(accepted, x, y.t, c.window))
               for x, y in combinations(combo, 2)):
            return combo
    return None


def _consumed(c, reg, policy, accepted, pis, sources) -> Set[str]:
    """Replay accepted dependent operations to find which approvals they used up."""
    used: Set[str] = set()
    for k, y in enumerate(accepted):
        if y.op != c.before_op or not c.applies(_otype(y.obj), dict(y.attrs)):
            continue
        prefix = accepted[:k]
        lab_k = ref_labels(prefix, sources, policy.verifiers)
        apps = [x for x in prefix if x.obj == y.obj and x.op == c.require_op
                and c.applies(_otype(x.obj), dict(x.attrs)) and y.t - x.t <= c.window
                and y.ref is not None and x.ref == y.ref and x.eid not in used]
        combo = _first_combo(c, reg, policy, apps, pis, lab_k, prefix)
        if combo is not None:
            used |= {x.eid for x in combo}
    return used


def ref_decide(policy, reg, sources, accepted: List, e) -> str:
    pi_e = ref_pi(reg, e)
    if pi_e is None:
        return "deny"
    pis = {x.eid: ref_pi(reg, x) for x in accepted}
    pis[e.eid] = pi_e
    lab = ref_labels(accepted, sources, policy.verifiers)
    dec = "allow"
    for c in policy.constraints:
        if not c.applies(_otype(e.obj), dict(e.attrs)):
            continue
        if c.conflict and e.op in c.conflict:
            hist = [x.actor for (x, y) in _conflict_pairs(c, accepted) if y.actor == e.actor]
            last = hist[-c.history:] if c.history > 0 else hist[-1:]
            for x in accepted:
                if (x.obj == e.obj and c.applies(_otype(x.obj), dict(x.attrs))
                        and x.op in c.conflict and x.op != e.op and e.t - x.t <= c.window):
                    cnt = sum(1 for a in last if a == x.actor)
                    if not _independent(c, reg, policy.exempt, x, e, pis, lab, cnt,
                                        _contrib(accepted, x, e.t, c.window)):
                        dec = max(dec, c.on_violation, key=SEV.get)
        if c.require_n and e.op == c.before_op:
            consumed = _consumed(c, reg, policy, accepted, pis, sources)
            apps = [x for x in accepted if x.obj == e.obj and x.op == c.require_op
                    and c.applies(_otype(x.obj), dict(x.attrs)) and e.t - x.t <= c.window
                    and e.ref is not None and x.ref == e.ref and x.eid not in consumed]
            if _first_combo(c, reg, policy, apps, pis, lab, accepted) is None:
                dec = max(dec, c.on_violation, key=SEV.get)
    return dec


def ref_run(policy, reg, sources, events) -> List[str]:
    accepted: List = []
    out = []
    for e in events:
        d = ref_decide(policy, reg, sources, accepted, e)
        out.append(d)
        if d in ("allow", "audit_only"):
            accepted.append(e)
    return out
