"""Tamper-evident audit ledger and offline recomputation of decisions (Sect. 5)."""
from __future__ import annotations

import hashlib
import json
from typing import List, Optional

GENESIS = "0" * 64
SEVERITY = {"allow": 0, "audit_only": 1, "escalate": 2, "deny": 3}


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=sorted)


def _event_dict(e) -> dict:
    return {"eid": e.eid, "actor": e.actor, "op": e.op, "obj": e.obj, "t": e.t,
            "chain": list(e.chain), "artifact": e.artifact, "ref": e.ref,
            # context items are stored as hashes, not contents (privacy, Sect. 5)
            "context_h": sorted(hashlib.sha256(d.encode()).hexdigest()[:16] for d in e.context)}


class Ledger:
    def __init__(self) -> None:
        self.records: List[dict] = []
        self.head = GENESIS

    def append(self, e, decision: str, witnesses: list, policy_digest: str) -> dict:
        rec = {"seq": len(self.records), "event": _event_dict(e), "decision": decision,
               "witnesses": witnesses, "policy": policy_digest, "prev": self.head}
        h = hashlib.sha256((self.head + _canon(rec)).encode()).hexdigest()
        rec["hash"] = h
        self.records.append(rec)
        self.head = h
        return rec

    def anchor(self) -> str:
        """Value to sign and anchor externally (e.g. RFC 3161 time-stamping)."""
        return self.head


def verify_chain(records: List[dict], anchored_head: Optional[str] = None) -> Optional[int]:
    """Return None if the chain is intact, else the index of the first bad record."""
    prev = GENESIS
    for i, r in enumerate(records):
        body = {k: v for k, v in r.items() if k != "hash"}
        if r.get("prev") != prev:
            return i
        h = hashlib.sha256((prev + _canon(body)).encode()).hexdigest()
        if h != r.get("hash"):
            return i
        prev = h
    if anchored_head is not None and prev != anchored_head:
        return len(records)
    return None


def _pair_verdict(w: dict) -> bool:
    if w["actor_i"] == w["actor_j"]:
        return False
    lv = w["level"]
    if lv >= 1 and w.get("shared"):
        return False
    if lv >= 2:
        if w.get("influenced") is None or w.get("extra"):
            return False
    if lv >= 3:
        si, sj = w["substrate_i"], w["substrate_j"]
        both_agents = w["kind_i"] == "agent" and w["kind_j"] == "agent"
        if both_agents and not all(si[d] is not None and si[d] != sj[d] for d in w["dims"]):
            return False
        if w["pair_count"] > w["kappa"]:
            return False
    return True


def recompute(record: dict) -> str:
    """Recompute a decision from the witnesses stored in a ledger record alone.

    The auditor does not need access to the PDP, the registries or the
    provenance store: the witnesses carry the values the decision depended on.
    Cross-checking witnesses against registry snapshots is a separate step.
    """
    dec = "allow"
    for w in record["witnesses"]:
        if w.get("invalid_authority"):
            return "deny"
        if "require" in w:
            ok = w["verdict"]
        else:
            ok = _pair_verdict(w)
        if not ok and SEVERITY[w["on_violation"]] > SEVERITY[dec]:
            dec = w["on_violation"]
    return dec
