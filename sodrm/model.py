"""Formal model of agentic SoD (Sect. 4 of the paper).

Principals, substrates, attenuating delegations, events, provenance labels.
Time is an integer (seconds).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

Scope = FrozenSet[Tuple[str, str]]  # set of (operation, object_id); "*" as wildcard object


@dataclass(frozen=True)
class Principal:
    pid: str
    kind: str  # "human" | "agent" | "org"
    model: Optional[str] = None
    operator: Optional[str] = None

    def substrate(self) -> Dict[str, Optional[str]]:
        return {"model": self.model, "operator": self.operator}


@dataclass(frozen=True)
class Delegation:
    did: str
    grantor: str
    grantee: str
    scope: Scope
    t_start: int
    t_end: int


@dataclass(frozen=True)
class Event:
    """e = (a, o, b, t, C) plus the delegation chain used and the artifact produced.

    chain: delegation ids from the root grantor down to the actor ([] if the
    actor acts on its own authority).
    context: ids of data items in the actor's decision context.
    artifact: id of the schema-validated artifact this event produces (if any).
    outputs: ids of all data items this event produces (artifact included).
    ref: digest of the artifact version this event acts upon (e.g. the proposal an
         approval or an execution refers to); None if not applicable.
    """
    eid: str
    actor: str
    op: str
    obj: str
    t: int
    context: FrozenSet[str] = frozenset()
    chain: Tuple[str, ...] = ()
    artifact: Optional[str] = None
    outputs: FrozenSet[str] = frozenset()
    attrs: Tuple[Tuple[str, object], ...] = ()  # object attributes, e.g. criticality
    ref: Optional[str] = None


def scope_covers(scope: Scope, op: str, obj: str) -> bool:
    return (op, obj) in scope or (op, "*") in scope


class Registry:
    """Identity, delegation and substrate registry (T3: written only by humans)."""

    def __init__(self) -> None:
        self.principals: Dict[str, Principal] = {}
        self.delegations: Dict[str, Delegation] = {}
        self.own_scope: Dict[str, Scope] = {}  # authority held directly (roles)

    def add_principal(self, p: Principal, own_scope: Iterable[Tuple[str, str]] = ()) -> None:
        self.principals[p.pid] = p
        self.own_scope[p.pid] = frozenset(own_scope)

    def add_delegation(self, d: Delegation) -> None:
        g = self.principals.get(d.grantor)
        if g is None or self.principals.get(d.grantee) is None:
            raise ValueError("unknown principal in delegation")
        if g.kind == "agent":
            # T3: agents may request but not grant; a grant must come from a human
            # (or be an attenuation of a human grant, see attenuate()).
            pass
        self.delegations[d.did] = d

    def effective_scope(self, pid: str, t: int, chain: Tuple[str, ...]) -> Optional[Scope]:
        """Scope available to pid through `chain`, or None if the chain is invalid.

        Attenuation: each hop's scope must be covered by the scope its grantor holds.
        """
        if not chain:
            return self.own_scope.get(pid, frozenset())
        first = self.delegations.get(chain[0])
        if first is None:
            return None
        held = self.own_scope.get(first.grantor, frozenset())
        prev_grantee = first.grantor
        for did in chain:
            d = self.delegations.get(did)
            if d is None or d.grantor != prev_grantee or not (d.t_start <= t <= d.t_end):
                return None
            if not all(scope_covers(held, o, b) for (o, b) in d.scope):
                return None  # not attenuating
            held = d.scope
            prev_grantee = d.grantee
        if prev_grantee != pid:
            return None
        return held

    def effective_principals(self, e: Event) -> Optional[FrozenSet[str]]:
        """Pi(e): actor plus every principal on the delegation path. None if invalid."""
        sc = self.effective_scope(e.actor, e.t, e.chain)
        if sc is None or not scope_covers(sc, e.op, e.obj):
            return None
        ps = {e.actor}
        for did in e.chain:
            d = self.delegations[did]
            ps.add(d.grantor)
            ps.add(d.grantee)
        return frozenset(ps)


class Provenance:
    """prov(d): ids of events whose outputs contributed to data item d."""

    def __init__(self, horizon: Optional[int] = None) -> None:
        """horizon: if set, labels keep only events at most `horizon` time units old.

        Sound for the monitor when horizon >= the largest window of the policy,
        since independence is only ever checked against events within the window.
        """
        self.labels: Dict[str, FrozenSet[str]] = {}
        self.horizon = horizon
        self.time: Dict[str, int] = {}

    def record_outputs(self, e: Event) -> None:
        # Outputs of e depend on e and, transitively, on everything in e's context.
        inherited = set()
        for d in e.context:
            inherited |= self.labels.get(d, frozenset())
        if self.horizon is not None:
            cutoff = e.t - self.horizon
            inherited = {x for x in inherited if self.time.get(x, cutoff) >= cutoff}
        self.time[e.eid] = e.t
        lab = frozenset(inherited | {e.eid})
        for d in e.outputs | ({e.artifact} if e.artifact else set()):
            self.labels[d] = lab

    def add_source(self, item: str, label: Iterable[str] = ()) -> None:
        """Untrusted external item (e.g. an e-mail); empty or given label."""
        self.labels[item] = frozenset(label)

    def influenced_by(self, context: FrozenSet[str], eid: str) -> Optional[FrozenSet[str]]:
        """Items of the context carrying provenance from eid; None if an item is unlabelled."""
        out = set()
        for d in context:
            lab = self.labels.get(d)
            if lab is None:
                return None  # fail-closed: unknown provenance (P4)
            if eid in lab:
                out.add(d)
        return frozenset(out)
