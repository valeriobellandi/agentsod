"""Tests derived from the formal results of Sect. 4 and the design of Sect. 5."""
import copy
import random
from pathlib import Path

import pytest

from sodrm import Delegation, Event, Ledger, Monitor, Principal, Provenance, Registry, parse
from sodrm.ledger import recompute, verify_chain
from sodrm.monitor import check_pair
from sodrm.policy import Constraint
from sodrm.reference import ref_run

POLICY = parse((Path(__file__).parent.parent / "policies" / "energy_ops.sod").read_text())
DAY = 86400
SP = "SetpointChange(feeder7|c1)"
ALL_SP = [("propose", "*"), ("approve", "*"), ("execute", "*")]


def base_registry():
    r = Registry()
    r.add_principal(Principal("org:acme", "org"))
    for h in ("eng_rossi", "eng_bianchi", "supervisor"):
        r.add_principal(Principal(h, "human"), ALL_SP)
    r.add_principal(Principal("a1", "agent", "M1", "X"))
    r.add_principal(Principal("a2", "agent", "M2", "Y"))
    r.add_principal(Principal("a2same", "agent", "M1", "X"))
    r.add_principal(Principal("a3", "agent", "M3", "Z"))
    return r


def grant(r, did, grantor, grantee, ops):
    r.add_delegation(Delegation(did, grantor, grantee, frozenset((o, "*") for o in ops), 0, 10 * DAY))


def ev(eid, actor, op, t, chain, context=(), artifact=None, outputs=(), obj=SP, attrs=(), ref=None):
    return Event(eid, actor, op, obj, t, frozenset(context), tuple(chain), artifact,
                 frozenset(outputs), tuple(attrs), ref)


def only(policy_name):
    p = copy.copy(POLICY)
    p.constraints = [c for c in POLICY.constraints if c.name == policy_name]
    return p


def with_level(c: Constraint, level: int) -> Constraint:
    return Constraint(**{**c.__dict__, "level": level})


# ---------------------------------------------------------------- parser

def test_parser_reads_listing():
    assert POLICY.name == "energy_ops"
    assert POLICY.exempt == frozenset({"org:acme"})
    assert len(POLICY.bindings) == 6
    names = {c.name: c for c in POLICY.constraints}
    assert names["setpoint_4eyes"].level == 3 and names["setpoint_4eyes"].kappa == 10
    assert names["setpoint_4eyes"].history == 50 and names["setpoint_4eyes"].window == DAY
    assert names["critical_dual_approval"].require_n == 2
    assert names["critical_dual_approval"].before_op == "execute"
    assert names["critical_dual_approval"].level == 2
    assert names["p2p"].level == 1 and names["p2p"].window == 30 * DAY
    assert POLICY.bind("grid.approve_change", {"asset": "feeder7", "cid": "c1"}) == \
        ("approve", "SetpointChange", SP)


# ---------------------------------------------------------------- running example

def _example(variant):
    r = base_registry()
    grant(r, "d1", "eng_rossi", "a1", ["propose"])
    approver_grantor = "eng_rossi" if variant == "same_engineer" else "eng_bianchi"
    approver = "a2same" if variant == "same_substrate" else "a2"
    grant(r, "d2", approver_grantor, approver, ["approve"])
    prov = Provenance()
    m = Monitor(only("setpoint_4eyes"), r, prov)
    e1 = ev("e1", "a1", "propose", 0, ["d1"], artifact="art1", outputs=["rationale1"])
    assert m.decide(e1).decision == "allow"
    ctx = ["art1", "rationale1"] if variant == "rationale" else ["art1"]
    e2 = ev("e2", approver, "approve", 3600, ["d2"], context=ctx)
    return m.decide(e2)


@pytest.mark.parametrize("variant,failed_at", [
    ("same_engineer", 1), ("rationale", 2), ("same_substrate", 3)])
def test_running_example_violations(variant, failed_at):
    d = _example(variant)
    assert d.decision == "escalate"
    assert d.witnesses[0]["failed_at"] == failed_at


def test_running_example_compliant():
    assert _example("ok").decision == "allow"


def test_pairing_bound_forces_rotation():
    r = base_registry()
    grant(r, "d1", "eng_rossi", "a1", ["propose"])
    grant(r, "d2", "eng_bianchi", "a2", ["approve"])
    m = Monitor(only("setpoint_4eyes"), r, Provenance())
    decisions = []
    for k in range(13):
        obj = f"SetpointChange(feeder7|c{k})"
        m.decide(ev(f"p{k}", "a1", "propose", 10 * k, ["d1"], artifact=f"art{k}", obj=obj))
        decisions.append(m.decide(ev(f"a{k}", "a2", "approve", 10 * k + 1, ["d2"],
                                     context=[f"art{k}"], obj=obj)).decision)
    # kappa = 10: approvals 0..10 see <= 10 prior pairings, the 12th sees 11
    assert decisions[:11] == ["allow"] * 11
    assert decisions[11] == "escalate"


# ---------------------------------------------------------------- formal results

def test_strictness_witnesses():
    """Prop. Strictness: each trace satisfies level l and violates level l+1."""
    c = only("setpoint_4eyes").constraints[0]
    for variant, ell in [("same_engineer", 0), ("rationale", 1), ("same_substrate", 2)]:
        for level, expect in [(ell, "allow"), (ell + 1, "escalate")]:
            p = only("setpoint_4eyes")
            p.constraints = [with_level(c, level)]
            r = base_registry()
            grant(r, "d1", "eng_rossi", "a1", ["propose"])
            grant(r, "d2", "eng_rossi" if variant == "same_engineer" else "eng_bianchi",
                  "a2same" if variant == "same_substrate" else "a2", ["approve"])
            m = Monitor(p, r, Provenance())
            m.decide(ev("e1", "a1", "propose", 0, ["d1"], artifact="art1", outputs=["rat1"]))
            ctx = ["art1", "rat1"] if variant == "rationale" else ["art1"]
            actor = "a2same" if variant == "same_substrate" else "a2"
            got = m.decide(ev("e2", actor, "approve", 1, ["d2"], context=ctx)).decision
            assert got == expect, (variant, level, got)


def test_monotonicity_random():
    """Lemma Monotonicity on random pairs: verdict at level l implies verdict at l-1."""
    rng = random.Random(1)
    reg = base_registry()
    c = only("setpoint_4eyes").constraints[0]
    agents = ["a1", "a2", "a2same", "a3"]
    people = ["eng_rossi", "eng_bianchi", "supervisor", "org:acme"]
    for _ in range(5000):
        ai, aj = rng.choice(agents), rng.choice(agents)
        pi_i = frozenset({ai} | set(rng.sample(people, rng.randint(0, 2))))
        pi_j = frozenset({aj} | set(rng.sample(people, rng.randint(0, 2))))
        e_i = ev("x", ai, "propose", 0, [], artifact="art")
        e_j = ev("y", aj, "approve", 1, [])
        infl = frozenset(rng.sample(["art", "rat", "mail"], rng.randint(0, 2)))
        cnt = rng.randint(0, 15)
        v = [check_pair(c, lv, e_i, e_j, pi_i, pi_j, infl, {"org:acme"}, reg, cnt)["verdict"]
             for lv in range(4)]
        for lv in range(1, 4):
            assert not v[lv] or v[lv - 1]


def test_conservativity_humans_without_delegation():
    """Prop. Conservativity: with humans only and no delegation, levels 0 and 1 coincide."""
    rng = random.Random(2)
    base = POLICY.constraints[2]  # p2p
    for _ in range(200):
        r = Registry()
        humans = [f"h{i}" for i in range(4)]
        for h in humans:
            r.add_principal(Principal(h, "human"),
                            [("create_vendor", "*"), ("approve_inv", "*"), ("release", "*")])
        events = [ev(f"e{k}", rng.choice(humans), rng.choice(["create_vendor", "approve_inv", "release"]),
                     k, [], obj=f"VendorPayment({rng.randint(0, 3)})") for k in range(20)]
        outs = []
        for lv in (0, 1):
            p = copy.copy(POLICY)
            p.constraints = [with_level(base, lv)]
            m = Monitor(p, r, Provenance())
            outs.append([m.decide(e).decision for e in events])
        assert outs[0] == outs[1]


# ---------------------------------------------------------------- require n

def test_dual_approval_before_execute():
    r = base_registry()
    grant(r, "d1", "eng_rossi", "a1", ["propose", "execute"])
    grant(r, "d2", "eng_bianchi", "a2", ["approve"])
    grant(r, "d3", "supervisor", "a3", ["approve"])
    crit = (("criticality", "HIGH"),)
    m = Monitor(only("critical_dual_approval"), r, Provenance())
    m.decide(ev("p", "a1", "propose", 0, ["d1"], artifact="art", attrs=crit))
    m.decide(ev("ap1", "a2", "approve", 10, ["d2"], context=["art"], artifact="ok1",
                outputs=["note1"], attrs=crit, ref="art"))
    assert m.decide(ev("x1", "a1", "execute", 20, ["d1"], attrs=crit, ref="art")).decision == "deny"
    # second approver reads the first approver's free-text note: not separated at level 2
    m.decide(ev("ap2", "a3", "approve", 30, ["d3"], context=["art", "note1"], attrs=crit, ref="art"))
    assert m.decide(ev("x2", "a1", "execute", 40, ["d1"], attrs=crit, ref="art")).decision == "deny"
    m.decide(ev("ap3", "a3", "approve", 50, ["d3"], context=["art"], attrs=crit, ref="art"))
    assert m.decide(ev("x3", "a1", "execute", 60, ["d1"], attrs=crit, ref="art")).decision == "allow"
    # low criticality: the constraint does not apply
    low = (("criticality", "LOW"),)
    obj2 = "SetpointChange(feeder7|c2)"
    assert m.decide(ev("x4", "a1", "execute", 70, ["d1"], obj=obj2, attrs=low)).decision == "allow"


# ---------------------------------------------------------------- fail-closed and authority

def test_invalid_or_non_attenuating_delegation_is_denied():
    r = base_registry()
    grant(r, "d1", "eng_rossi", "a1", ["propose"])
    # a1 tries to pass on 'approve', which it does not hold: not attenuating
    r.add_delegation(Delegation("d9", "a1", "a3", frozenset({("approve", "*")}), 0, DAY))
    m = Monitor(only("setpoint_4eyes"), r, Provenance())
    assert m.decide(ev("e", "a3", "approve", 1, ["d1", "d9"])).decision == "deny"
    # expired delegation
    assert m.decide(ev("e2", "a1", "propose", 20 * DAY, ["d1"])).decision == "deny"


def test_delegation_laundering_by_agent():
    """A3: a1 re-delegates its own authority to a3, which approves a1's proposal."""
    r = base_registry()
    grant(r, "d1", "eng_rossi", "a1", ["propose", "approve"])
    r.add_delegation(Delegation("d2", "a1", "a3", frozenset({("approve", "*")}), 0, DAY))
    m = Monitor(only("setpoint_4eyes"), r, Provenance())
    m.decide(ev("e1", "a1", "propose", 0, ["d1"], artifact="art"))
    d = m.decide(ev("e2", "a3", "approve", 1, ["d1", "d2"], context=["art"]))
    assert d.decision == "escalate" and d.witnesses[0]["failed_at"] == 1
    assert "a1" in d.witnesses[0]["shared"]


def test_unlabelled_context_fails_closed():
    r = base_registry()
    grant(r, "d1", "eng_rossi", "a1", ["propose"])
    grant(r, "d2", "eng_bianchi", "a2", ["approve"])
    m = Monitor(only("setpoint_4eyes"), r, Provenance())
    m.decide(ev("e1", "a1", "propose", 0, ["d1"], artifact="art"))
    d = m.decide(ev("e2", "a2", "approve", 1, ["d2"], context=["art", "unknown_item"]))
    assert d.decision == "escalate" and d.witnesses[0]["failed_at"] == 2


def test_external_source_does_not_create_dependence():
    r = base_registry()
    grant(r, "d1", "eng_rossi", "a1", ["propose"])
    grant(r, "d2", "eng_bianchi", "a2", ["approve"])
    prov = Provenance()
    prov.add_source("email7")
    m = Monitor(only("setpoint_4eyes"), r, prov)
    m.decide(ev("e1", "a1", "propose", 0, ["d1"], context=["email7"], artifact="art"))
    # the approver read the same e-mail: allowed by the model (this is attack A6's residue)
    assert m.decide(ev("e2", "a2", "approve", 1, ["d2"], context=["art", "email7"])).decision == "allow"


# ---------------------------------------------------------------- ledger

def test_ledger_integrity_and_recompute():
    r = base_registry()
    grant(r, "d1", "eng_rossi", "a1", ["propose"])
    grant(r, "d2", "eng_rossi", "a2", ["approve"])
    grant(r, "d3", "eng_bianchi", "a3", ["approve"])
    m = Monitor(only("setpoint_4eyes"), r, Provenance())
    m.decide(ev("e1", "a1", "propose", 0, ["d1"], artifact="art", outputs=["rat"]))
    m.decide(ev("e2", "a2", "approve", 1, ["d2"], context=["art"]))
    m.decide(ev("e3", "a3", "approve", 2, ["d3"], context=["art", "rat"]))
    m.decide(ev("e4", "a3", "approve", 3, ["d3"], context=["art"]))
    recs = m.ledger.records
    assert verify_chain(recs, m.ledger.anchor()) is None
    assert [recompute(x) for x in recs] == [x["decision"] for x in recs]
    tampered = copy.deepcopy(recs)
    tampered[1]["decision"] = "allow"
    assert verify_chain(tampered) == 1
    assert verify_chain(recs[:2] + recs[3:]) == 2          # deletion detected
    assert verify_chain(recs[:3], m.ledger.anchor()) == 3   # truncation detected via anchor


# ---------------------------------------------------------------- monitor = model

def test_monitor_matches_reference_on_scenarios():
    r = base_registry()
    grant(r, "d1", "eng_rossi", "a1", ["propose", "approve", "execute"])
    grant(r, "d2", "eng_bianchi", "a2", ["approve"])
    grant(r, "d3", "supervisor", "a3", ["approve"])
    events = [
        ev("e1", "a1", "propose", 0, ["d1"], artifact="art", outputs=["rat"]),
        ev("e2", "a1", "approve", 1, ["d1"], context=["art"]),
        ev("e3", "a2", "approve", 2, ["d2"], context=["art", "rat"]),
        ev("e4", "a2", "approve", 3, ["d2"], context=["art"]),
        ev("e5", "a3", "approve", 4, ["d3"], context=["art"]),
    ]
    m = Monitor(POLICY, r, Provenance())
    assert [m.decide(e).decision for e in events] == ref_run(POLICY, r, {}, events)


# ---------------------------------------------------------------- artifact versions, single use, concurrency

def _dual_setup():
    r = base_registry()
    grant(r, "d1", "eng_rossi", "a1", ["propose", "execute"])
    grant(r, "d2", "eng_bianchi", "a2", ["approve"])
    grant(r, "d3", "supervisor", "a3", ["approve"])
    return r, (("criticality", "HIGH"),)


def test_execute_bound_to_approved_artifact_version():
    r, crit = _dual_setup()
    m = Monitor(only("critical_dual_approval"), r, Provenance())
    m.decide(ev("p1", "a1", "propose", 0, ["d1"], artifact="v1", attrs=crit))
    m.decide(ev("ap1", "a2", "approve", 1, ["d2"], context=["v1"], attrs=crit, ref="v1"))
    m.decide(ev("ap2", "a3", "approve", 2, ["d3"], context=["v1"], attrs=crit, ref="v1"))
    # the proposal is modified after approval: the old approvals do not cover v2
    m.decide(ev("p2", "a1", "propose", 3, ["d1"], artifact="v2", attrs=crit))
    assert m.decide(ev("x_v2", "a1", "execute", 4, ["d1"], attrs=crit, ref="v2")).decision == "deny"
    # an execution that does not say which version it executes is denied (fail-closed)
    assert m.decide(ev("x_none", "a1", "execute", 5, ["d1"], attrs=crit)).decision == "deny"
    # the approved version can be executed
    assert m.decide(ev("x_v1", "a1", "execute", 6, ["d1"], attrs=crit, ref="v1")).decision == "allow"


def test_approvals_are_single_use():
    r, crit = _dual_setup()
    m = Monitor(only("critical_dual_approval"), r, Provenance())
    m.decide(ev("p1", "a1", "propose", 0, ["d1"], artifact="v1", attrs=crit))
    m.decide(ev("ap1", "a2", "approve", 1, ["d2"], context=["v1"], attrs=crit, ref="v1"))
    m.decide(ev("ap2", "a3", "approve", 2, ["d3"], context=["v1"], attrs=crit, ref="v1"))
    assert m.decide(ev("x1", "a1", "execute", 3, ["d1"], attrs=crit, ref="v1")).decision == "allow"
    assert m.decide(ev("x2", "a1", "execute", 4, ["d1"], attrs=crit, ref="v1")).decision == "deny"
    m.decide(ev("ap3", "a2", "approve", 5, ["d2"], context=["v1"], attrs=crit, ref="v1"))
    m.decide(ev("ap4", "a3", "approve", 6, ["d3"], context=["v1"], attrs=crit, ref="v1"))
    assert m.decide(ev("x3", "a1", "execute", 7, ["d1"], attrs=crit, ref="v1")).decision == "allow"


def test_concurrent_executions_are_serialized():
    """Executions racing on the same approvals: exactly one is allowed, and the log
    order is a sequential order that the reference oracle reproduces."""
    import threading
    for _ in range(50):
        r, crit = _dual_setup()
        p = only("critical_dual_approval")
        m = Monitor(p, r, Provenance())
        pre = [ev("p1", "a1", "propose", 0, ["d1"], artifact="v1", attrs=crit),
               ev("ap1", "a2", "approve", 1, ["d2"], context=["v1"], attrs=crit, ref="v1"),
               ev("ap2", "a3", "approve", 2, ["d3"], context=["v1"], attrs=crit, ref="v1")]
        for e in pre:
            m.decide(e)
        racers = [ev(f"x{k}", "a1", "execute", 10, ["d1"], attrs=crit, ref="v1") for k in range(8)]
        out = {}
        barrier = threading.Barrier(len(racers))

        def go(e):
            barrier.wait()
            out[e.eid] = m.decide(e).decision
        ts = [threading.Thread(target=go, args=(e,)) for e in racers]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert sorted(out.values()) == ["allow"] + ["deny"] * 7
        by_id = {e.eid: e for e in pre + racers}
        replay = [by_id[rec["event"]["eid"]] for rec in m.ledger.records]
        assert ref_run(p, r, {}, replay) == [rec["decision"] for rec in m.ledger.records]


def test_pairing_history_outlives_the_window():
    """Pairings counted for rotation are kept even when the paired events are much
    older than the conflict window (two days apart, 24h window)."""
    r = base_registry()
    r.add_delegation(Delegation("d1", "eng_rossi", "a1", frozenset({("propose", "*")}), 0, 100 * DAY))
    r.add_delegation(Delegation("d2", "eng_bianchi", "a2", frozenset({("approve", "*")}), 0, 100 * DAY))
    p = only("setpoint_4eyes")
    m = Monitor(p, r, Provenance(horizon=DAY))
    decisions = []
    for k in range(13):
        obj = f"SetpointChange(feeder7|c{k})"
        t = k * 2 * DAY
        m.decide(ev(f"p{k}", "a1", "propose", t, ["d1"], artifact=f"art{k}", obj=obj))
        decisions.append(m.decide(ev(f"a{k}", "a2", "approve", t + 60, ["d2"],
                                     context=[f"art{k}"], obj=obj, ref=f"art{k}")).decision)
    assert decisions[:11] == ["allow"] * 11 and decisions[11] == "escalate"
    assert sum(1 for x in m._pairs(p.constraints[0], "a2") if x == "a1") >= 11


def test_substrate_must_differ_in_every_listed_dimension():
    """Same model served by a different operator: the pairs differ, but level 3 with
    dims (model, operator) still fails. Different model and operator: allowed."""
    for model, operator, expect in [("M1", "Y", "escalate"), ("M2", "X", "escalate"),
                                    ("M2", "Y", "allow")]:
        r = base_registry()
        r.add_principal(Principal("b", "agent", model, operator))
        grant(r, "d1", "eng_rossi", "a1", ["propose"])          # a1 runs on (M1, X)
        grant(r, "d2", "eng_bianchi", "b", ["approve"])
        m = Monitor(only("setpoint_4eyes"), r, Provenance())
        m.decide(ev("e1", "a1", "propose", 0, ["d1"], artifact="art"))
        d = m.decide(ev("e2", "b", "approve", 1, ["d2"], context=["art"], ref="art"))
        assert d.decision == expect, (model, operator, d.decision)
        if expect == "escalate":
            assert d.witnesses[0]["failed_at"] == 3


# ---------------------------------------------------------------- level 2: verifiers and earlier messages

def _l2_monitor(verifiers=()):
    r = base_registry()
    for h, extra in (("eng_rossi", "message"), ("eng_bianchi", "simulate"), ("eng_bianchi", "summarize")):
        r.own_scope[h] = r.own_scope[h] | {(extra, "*")}
    grant(r, "d1", "eng_rossi", "a1", ["propose", "message"])
    grant(r, "d2", "eng_bianchi", "a2", ["approve", "simulate", "summarize"])
    p = copy.copy(only("setpoint_4eyes"))
    p.constraints = [with_level(p.constraints[0], 2)]
    p.verifiers = frozenset(verifiers)
    prov = Provenance()
    return Monitor(p, r, prov), prov


def test_declared_verifier_on_artifact_is_readable_at_level_2():
    m, _ = _l2_monitor(verifiers={"simulate"})
    assert m.decide(ev("e1", "a1", "propose", 0, ["d1"], artifact="art1", outputs=["just1"])).decision == "allow"
    assert m.decide(ev("s1", "a2", "simulate", 10, ["d2"], context=["art1"], outputs=["sim1"])).decision == "allow"
    assert m.decide(ev("e2", "a2", "approve", 20, ["d2"], context=["art1", "sim1"])).decision == "allow"


def test_tool_re_presenting_the_justification_is_blocked_at_level_2():
    # an undeclared tool, or a declared verifier fed with the justification, keeps the provenance
    for op, verifiers in (("summarize", {"simulate"}), ("simulate", {"simulate"})):
        m, _ = _l2_monitor(verifiers=verifiers)
        m.decide(ev("e1", "a1", "propose", 0, ["d1"], artifact="art1", outputs=["just1"]))
        m.decide(ev("s1", "a2", op, 10, ["d2"], context=["art1", "just1"], outputs=["out1"]))
        assert m.decide(ev("e2", "a2", "approve", 20, ["d2"], context=["art1", "out1"])).decision != "allow"


def test_message_sent_before_the_proposal_is_blocked_at_level_2():
    m, prov = _l2_monitor()
    prov.add_source("ticket")
    msg_obj = "Chat(a1|a2)"
    assert m.decide(ev("m1", "a1", "message", 0, ["d1"], context=["ticket"], outputs=["msg1"],
                       obj=msg_obj)).decision == "allow"
    m.decide(ev("e1", "a1", "propose", 10, ["d1"], artifact="art1", outputs=["just1"]))
    assert m.decide(ev("e2", "a2", "approve", 20, ["d2"], context=["art1", "msg1"])).decision != "allow"
    # the oracle agrees on the same trace
    m2, prov2 = _l2_monitor()
    prov2.add_source("ticket")
    evs = [ev("m1", "a1", "message", 0, ["d1"], context=["ticket"], outputs=["msg1"], obj=msg_obj),
           ev("e1", "a1", "propose", 10, ["d1"], artifact="art1", outputs=["just1"]),
           ev("e2", "a2", "approve", 20, ["d2"], context=["art1", "msg1"])]
    assert ref_run(m2.policy, m2.reg, {"ticket": frozenset()}, evs) == [m2.decide(e).decision for e in evs]
