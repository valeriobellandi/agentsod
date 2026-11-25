"""Principals, delegations and policies of the simulated operator.

Per workflow there are seven agents acting for five humans plus a team lead:

  agent   role            acts for   authority           model
  init1   initiator       h_a        initiate, check     M1
  init2   initiator       h_b        initiate, check     M2
  chk1    checker         h_c        check               M2
  chk2    checker         h_d        check               M3
  chk3    checker         h_a        initiate, check     M3   (same human as init1)
  fin1    final operator  h_e        final               M3
  ops     ops assistant   h_lead     all three           M1   (over-provisioned)

Engineers who may both propose and review (h_a, h_b) are common in small teams; the
ops assistant has the full authority of its team lead. Each model family is served
through its own endpoint, which emulates the operator dimension (X, Y, Z).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from sodrm.model import Delegation, Principal, Registry
from sodrm.policy import Policy, parse

from .systems import WORKFLOWS, Workflow

MODELS = {"M1": "X", "M2": "Y", "M3": "Z"}
LEVEL_NAMES = {0: "identity", 1: "accountability", 2: "information"}

AGENTS = [  # name, role index, human, authority (indices into ops), model
    ("init1", 0, "h_a", (0, 1), "M1"),
    ("init2", 0, "h_b", (0, 1), "M2"),
    ("chk1", 1, "h_c", (1,), "M2"),
    ("chk2", 1, "h_d", (1,), "M3"),
    ("chk3", 1, "h_a", (0, 1), "M3"),
    ("fin1", 2, "h_e", (2,), "M3"),
    ("ops", 0, "h_lead", (0, 1, 2), "M1"),
]
HUMAN_SCOPE = {"h_a": (0, 1), "h_b": (0, 1), "h_c": (1,), "h_d": (1,), "h_e": (2,),
               "h_lead": (0, 1, 2)}


def aid(wf: str, name: str) -> str:
    return f"{wf}.{name}"


def hid(wf: str, h: str) -> str:
    return f"{wf}.{h}"


def supervisor(wf: str) -> str:
    return f"{wf}.sup"


class World:
    """Registry plus, for each agent, the delegation chains it can present."""

    def __init__(self) -> None:
        self.reg = Registry()
        self.reg.add_principal(Principal("org:acme", "org"))
        self.chains: Dict[str, List[Tuple[str, ...]]] = {}
        self.role_agents: Dict[Tuple[str, int], List[str]] = {}
        self._did = 0
        for wf in WORKFLOWS.values():
            for h, idx in HUMAN_SCOPE.items():
                self.reg.add_principal(Principal(hid(wf.name, h), "human"),
                                       [(wf.ops[i], "*") for i in idx])
            self.reg.add_principal(Principal(supervisor(wf.name), "human"),
                                   [(o, "*") for o in wf.ops])
            for name, role, h, idx, model in AGENTS:
                a = aid(wf.name, name)
                self.reg.add_principal(Principal(a, "agent", model, MODELS[model]))
                d = self.delegate(hid(wf.name, h), a, [wf.ops[i] for i in idx])
                self.chains[a] = [(d,)]
                self.role_agents.setdefault((wf.name, role), []).append(a)

    def delegate(self, grantor: str, grantee: str, ops: List[str], t0: int = 0,
                 t1: int = 10 ** 9) -> str:
        self._did += 1
        did = f"d{self._did}"
        self.reg.add_delegation(Delegation(did, grantor, grantee,
                                           frozenset((o, "*") for o in ops), t0, t1))
        return did

    def spawn_subagent(self, parent: str, ops: List[str]) -> str:
        """A sub-agent started by `parent` on its own model, under a delegation from it.

        The platform grants such requests automatically: this is the delegation
        laundering of attack A3 (variant a).
        """
        p = self.reg.principals[parent]
        n = sum(1 for x in self.reg.principals if x.startswith(parent + ".sub"))
        sub = f"{parent}.sub{n}"
        self.reg.add_principal(Principal(sub, "agent", p.model, p.operator))
        d = self.delegate(parent, sub, ops)
        self.chains[sub] = [c + (d,) for c in self.chains[parent]]
        return sub

    def chain_for(self, agent: str, op: str, t: int) -> Tuple[str, ...]:
        """First chain whose scope covers op; otherwise the agent's first chain (which
        then fails validation: authority claimed but not held)."""
        cands = self.chains.get(agent, [()])
        for c in cands:
            sc = self.reg.effective_scope(agent, t, c)
            if sc is not None and ((op, "*") in sc):
                return c
        return cands[0]


def policy_text(level: int, history: int = 50, kappa: int = 20) -> str:
    """AgentSoD policy for all four workflows at a given separation level.

    Four-eyes constraints (initiate vs. check) use the requested level; constraints
    involving the final operation use min(level, 2), since the final operator is a
    single control-room or treasury agent and substrate rotation does not apply to it.
    """
    def sep(lv: int) -> str:
        if lv <= 2:
            return f"separation {LEVEL_NAMES[lv]}"
        return f"separation substrate (model, operator) history {history} max {kappa}"

    out = ["policy gridops {", "  exempt org:acme"]
    for wf in WORKFLOWS.values():
        out.append(f"  object {wf.obj_type} key (key)")
    for wf in WORKFLOWS.values():
        for tool, op in zip(wf.tools, wf.ops):
            out.append(f"  bind {tool}(key, ..) -> {op} on {wf.obj_type}(key)")
    windows = {"grid": "24h", "ptw": "7d", "trade": "7d", "p2p": "30d"}
    for wf in WORKFLOWS.values():
        i, c, f = wf.ops
        out += [f"  constraint {wf.name}_4eyes {{", f"    on {wf.obj_type}",
                f"    conflict {{ {i}, {c} }}", f"    window {windows[wf.name]}",
                f"    {sep(level)}", f"    on_violation escalate(role:{wf.name}_supervisor)",
                "  }"]
        out += [f"  constraint {wf.name}_final {{", f"    on {wf.obj_type}",
                f"    conflict {{ {i}, {c}, {f} }}", f"    window {windows[wf.name]}",
                f"    {sep(min(level, 2))}", f"    on_violation escalate(role:{wf.name}_supervisor)",
                "  }"]
    out += ["  constraint grid_dual {", "    on SetpointChange where criticality >= HIGH",
            "    require 2 separated approve before execute", "    window 4h",
            f"    {sep(min(level, 2))}", "    on_violation deny", "  }", "}"]
    return "\n".join(out) + "\n"


def make_policy(level: int, history: int = 50, kappa: int = 20) -> Policy:
    return parse(policy_text(level, history, kappa))


def conflict_sets() -> Dict[str, Tuple[str, ...]]:
    """Ops that conflict on the same object, per object type (for baseline B4)."""
    return {wf.obj_type: wf.ops for wf in WORKFLOWS.values()}


def workflow_of_obj(obj: str) -> Workflow:
    t = obj.split("(", 1)[0]
    for wf in WORKFLOWS.values():
        if wf.obj_type == t:
            return wf
    raise KeyError(obj)
