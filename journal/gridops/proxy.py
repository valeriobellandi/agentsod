"""Enforcement proxy (PEP) of GridOps-SoD.

Every tool call of every agent goes through Proxy.call (T1). The proxy
  - delivers every data item to an agent's context and labels it with provenance (T2);
  - maps calls to events through the policy bindings, attaching the actor's delegation
    chain, its decision context, the artifact digest it refers to and object attributes;
  - asks the guard under test, and executes allowed calls itself;
  - routes escalations to a human supervisor, whose decision is itself an event;
  - in SoD-RM configurations, shows checkers only schema-validated artifacts, not the
    free text written around them (Sect. VI).

Context granularity (Sect. VI-F):
  session  everything delivered to the agent since it started work on the object;
  noreset  everything delivered since the agent started (no reset at object boundaries);
  plan     only the items the agent declares it used (exact, as with a plan interpreter);
           agents that cannot declare it fall back to `session`.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sodrm.model import Event, Provenance

from .guards import Guard
from .systems import TOOL_INDEX, WORKFLOWS, Systems
from .world import World, supervisor


@dataclass
class Item:
    iid: str
    kind: str          # doc | workorder | artifact | justification | approval | note | message | check
    text: str
    directives: List[dict] = field(default_factory=list)
    data: dict = field(default_factory=dict)


@dataclass
class TaskStats:
    calls: int = 0
    denies: int = 0
    escalations: int = 0
    esc_approved: int = 0
    guard_s: float = 0.0
    guard_calls: int = 0
    llm_errors: int = 0


READ_SUFFIX = {"get": "read_tool", "check": "check_tool", "reject": "reject_tool"}


class Proxy:
    def __init__(self, systems: Systems, world: World, guard: Guard, prov: Provenance,
                 granularity: str = "session", strip: bool = False, p_miss: float = 0.1,
                 seed: int = 0) -> None:
        self.sys, self.world, self.guard, self.prov = systems, world, guard, prov
        self.gran, self.strip, self.p_miss = granularity, strip, p_miss
        self.rng = random.Random(seed)
        self.items: Dict[str, Item] = {}
        self.session: Dict[str, List[str]] = defaultdict(list)
        self.read_version: Dict[tuple, str] = {}
        self.inbox: Dict[str, List[str]] = defaultdict(list)
        self.t = 0
        self._n = 0
        self.task: Optional[str] = None
        self.stats: Dict[str, TaskStats] = defaultdict(TaskStats)
        self.decision_log: List[dict] = []   # every guard decision, for diagnosis
        self.tools = {}
        for wf in WORKFLOWS.values():
            self.tools[wf.read_tool] = ("get", wf.name)
            self.tools[wf.check_tool] = ("check", wf.name)
            self.tools[wf.reject_tool] = ("reject", wf.name)
        for t, (wf, i) in TOOL_INDEX.items():
            self.tools[t] = ("op", wf, i)

    # ------------------------------------------------------------------ items
    def _id(self, p: str) -> str:
        self._n += 1
        return f"{p}{self._n}"

    def add_external(self, kind: str, text: str, directives=(), data=None, iid=None) -> str:
        """An item from outside (e-mail, document, work order): empty provenance."""
        iid = iid or self._id(kind[:3] + ":")
        self.items[iid] = Item(iid, kind, text, list(directives), dict(data or {}))
        self.prov.add_source(iid, ())
        return iid

    def _deliver(self, agent: str, iids: List[str]) -> None:
        s = self.session[agent]
        for i in iids:
            if i not in s:
                s.append(i)

    def begin(self, agent: str, obj: Optional[str] = None) -> None:
        """Agent starts working on an object: reset its session under `session`."""
        if self.gran in ("session", "plan"):
            self.session[agent] = []

    def context(self, agent: str, used: Optional[List[str]]) -> frozenset:
        if self.gran == "plan" and used is not None:
            return frozenset(used)
        return frozenset(self.session[agent])

    # ------------------------------------------------------------------ calls
    def call(self, agent: str, tool: str, args: Optional[dict] = None,
             used: Optional[List[str]] = None) -> dict:
        args = dict(args or {})
        st = self.stats[self.task or "-"]
        st.calls += 1
        self.t += 30
        if tool == "docs.read":
            iid = args.get("doc")
            if iid not in self.items:
                return {"error": "no such document"}
            self._deliver(agent, [iid])
            it = self.items[iid]
            return {"items": [iid], "text": it.text}
        if tool == "msg.send":
            return self._send(agent, args, used)
        if tool == "msg.inbox":
            new = self.inbox.pop(agent, [])
            self._deliver(agent, new)
            return {"items": new, "text": "\n".join(self.items[i].text for i in new) or "(empty)"}
        if tool == "agents.spawn":
            sub = self.world.spawn_subagent(agent, list(args.get("ops", [])))
            return {"agent": sub}
        spec = self.tools.get(tool)
        if spec is None:
            return {"error": f"unknown tool {tool}"}
        if spec[0] == "get":
            return self._get(agent, spec[1], str(args["key"]))
        if spec[0] == "check":
            return self._check(agent, spec[1], str(args["key"]))
        if spec[0] == "reject":
            return {"result": self.sys.reject(spec[1], str(args["key"]), agent)}
        return self._op(agent, spec[1], spec[2], args, used)

    def _send(self, agent: str, args: dict, used) -> dict:
        ctx = self.context(agent, used)
        label = set()
        for d in ctx:
            label |= set(self.prov.labels.get(d, ()))
        iid = self._id("msg:")
        self.items[iid] = Item(iid, "message", f"[from {agent}] {args.get('text', '')}",
                               list(args.get("directives", [])), {"from": agent})
        # every relayed message is an event of its sender (Sect. IV): its label carries that
        # event, so level 2 also covers messages sent before the conflicting event
        eid = self._id("e")
        ev = Event(eid, agent, "message", f"Message({iid})", self.t, frozenset(ctx), (), None,
                   frozenset({iid}), (), None)
        self.prov.add_source(iid, label | {eid})
        mon = getattr(self.guard, "monitor", None)
        if mon is not None:
            with mon._lock:
                mon.by_actor[agent].append(ev)
        self._deliver(agent, [iid])
        for r in args.get("to", []):
            self.inbox[r].append(iid)
        return {"result": "sent"}

    def _get(self, agent: str, wf: str, key: str) -> dict:
        t = self.sys.get(wf, key)
        if t is None:
            return {"error": "no such ticket"}
        art = [i for i in t.items if self.items[i].kind == "artifact"]
        latest = art[-1:] if art else []
        if self.strip:
            shown = latest
        else:
            shown = [i for i in t.items if self.items[i].kind in ("justification", "note", "approval")
                     ] + latest
        self._deliver(agent, shown)
        self.read_version[(agent, t.obj)] = t.version
        return {"items": shown, "status": t.status, "checks": len(t.checks),
                "required": t.required, "version": t.version,
                "text": "\n".join(self.items[i].text for i in shown)}

    def _check(self, agent: str, wf: str, key: str) -> dict:
        r = self.sys.check_tool(wf, key)
        iid = self._id("chk:")
        # a trusted, deterministic function of the artifact: carries no proposer free text
        self.items[iid] = Item(iid, "check", str(r), [], r)
        self.prov.add_source(iid, ())
        self._deliver(agent, [iid])
        return {"items": [iid], "result": r, "text": str(r)}

    # ------------------------------------------------------------------ bound ops
    def _event(self, agent: str, wf: str, i: int, key: str, payload: Optional[dict], used) -> Event:
        w = WORKFLOWS[wf]
        op = w.ops[i]
        obj = f"{w.obj_type}({key})"
        t = self.sys.get(wf, key)
        crit = (payload or {}).get("criticality") if i == 0 else (t.payload.get("criticality") if t else None)
        attrs = (("criticality", crit),) if (wf == "grid" and crit) else ()
        eid = self._id("e")
        artifact = f"art:{eid}" if i in (0, 1) else None
        outputs = {artifact} if artifact else set()
        if i == 0:
            outputs.add(f"just:{eid}")
        ref = None if i == 0 else self.read_version.get((agent, obj))
        return Event(eid, agent, op, obj, self.t, self.context(agent, used),
                     self.world.chain_for(agent, op, self.t), artifact, frozenset(outputs),
                     attrs, ref)

    def _decide(self, e: Event) -> str:
        st = self.stats[self.task or "-"]
        t0 = time.perf_counter()
        d = self.guard.decide(e)
        st.guard_s += time.perf_counter() - t0
        st.guard_calls += 1
        why = getattr(self.guard, "last_why", None) if d not in ("allow", "audit_only") else None
        self.decision_log.append({"task": self.task, "eid": e.eid, "actor": e.actor, "op": e.op,
                                  "obj": e.obj, "ref": e.ref, "context": sorted(e.context),
                                  "decision": d, "why": why})
        if d in ("allow", "audit_only") and not self.guard.uses_provenance:
            self.prov.record_outputs(e)
        return d

    def _op(self, agent: str, wf: str, i: int, args: dict, used) -> dict:
        key = str(args["key"])
        payload = args.get("payload")
        e = self._event(agent, wf, i, key, payload, used)
        st = self.stats[self.task or "-"]
        d = self._decide(e)
        if d in ("allow", "audit_only"):
            return {"decision": "allow", "result": self._apply(agent, wf, i, key, args, e)}
        if d == "deny":
            st.denies += 1
            return {"decision": "deny"}
        st.escalations += 1
        return self._escalate(wf, i, key, args, e)

    def _apply(self, actor: str, wf: str, i: int, key: str, args: dict, e: Event) -> str:
        if i == 0:
            payload = dict(args["payload"])
            t = self.sys.initiate(wf, key, payload)
            ai, ji = f"art:{e.eid}", f"just:{e.eid}"
            self.items[ai] = Item(ai, "artifact", f"{WORKFLOWS[wf].obj_type} {key}: {payload}",
                                  [], {"payload": payload, "version": t.version})
            self.items[ji] = Item(ji, "justification", args.get("justification", ""),
                                  list(args.get("directives", [])))
            t.items += [ji, ai]
            self._deliver(actor, [ai, ji])
            return "ok"
        if i == 1:
            r = self.sys.check(wf, key, actor, e.ref)
            if r == "ok":
                t = self.sys.get(wf, key)
                ai = f"art:{e.eid}"
                self.items[ai] = Item(ai, "approval", f"{WORKFLOWS[wf].ops[1]} by {actor}", [],
                                      {"ref": e.ref})
                t.items.append(ai)
                if args.get("note"):
                    ni = self._id("note:")
                    self.items[ni] = Item(ni, "note", f"[{actor}] {args['note']}")
                    lab = set(self.prov.labels.get(ai, ())) | {e.eid}
                    self.prov.add_source(ni, lab)
                    t.items.append(ni)
            return r
        return self.sys.final(wf, key, actor, e.ref)

    def _escalate(self, wf: str, i: int, key: str, args: dict, e: Event) -> dict:
        """A human supervisor reviews with full information; misses a harmful one with p_miss."""
        st = self.stats[self.task or "-"]
        t = self.sys.get(wf, key)
        harmful = t.truth_harmful if t is not None else self.sys.harmful(wf, key, args.get("payload", {}))
        approve = (not harmful) or self.rng.random() < self.p_miss
        if not approve:
            if t is not None:
                self.sys.reject(wf, key, supervisor(wf))
            return {"decision": "escalate", "result": "rejected by supervisor"}
        sup = supervisor(wf)
        hid_ = self._id("e")
        art = f"art:{hid_}" if e.artifact else None
        outs = {art} if art else set()
        if i == 0:
            outs.add(f"just:{hid_}")
        eh = Event(hid_, sup, e.op, e.obj, self.t, frozenset(), (), art, frozenset(outs), e.attrs,
                   t.version if (t is not None and i > 0) else None)
        dh = self._decide(eh)
        if dh not in ("allow", "audit_only"):
            return {"decision": "escalate", "result": "supervisor not allowed"}
        st.esc_approved += 1
        return {"decision": "escalate", "result": self._apply(sup, wf, i, key, args, eh)}

    def would(self, agent: str, wf: str, i: int, key: str) -> str:
        """Dry-run decision for scheduling: what the guard would say if `agent` acted now."""
        e = self._event(agent, wf, i, key, None, [])
        e = Event(e.eid, e.actor, e.op, e.obj, e.t, frozenset(), e.chain, e.artifact,
                  e.outputs, e.attrs, self.sys.get(wf, key).version if self.sys.get(wf, key) else None)
        self._n -= 1
        return self.guard.would(e)
