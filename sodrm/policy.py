"""AgentSoD policies: data structures and a parser for the language of Sect. 5."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

LEVELS = {"identity": 0, "accountability": 1, "informational": 2, "information": 2, "substrate": 3}
ORDINALS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
SEVERITY = {"allow": 0, "audit_only": 1, "escalate": 2, "deny": 3}


@dataclass(frozen=True)
class Binding:
    tool: str
    tool_args: Tuple[str, ...]
    op: str
    obj_type: str
    key_args: Tuple[str, ...]


@dataclass(frozen=True)
class Constraint:
    name: str
    obj_type: str
    level: int
    window: int
    on_violation: str
    conflict: FrozenSet[str] = frozenset()      # conflict {o1..om}: pairwise
    require_n: int = 0                           # require n independent <op> before <dep>
    require_op: Optional[str] = None
    before_op: Optional[str] = None
    where: Optional[Tuple[str, str, str]] = None  # (attr, cmp, value)
    dims: Tuple[str, ...] = ()
    history: int = 0
    kappa: int = 0
    artifact: Optional[str] = None
    escalate_to: Optional[str] = None

    def conflicts(self, o1: str, o2: str) -> bool:
        return bool(self.conflict) and o1 in self.conflict and o2 in self.conflict and o1 != o2

    def applies(self, obj_type: str, attrs: Dict[str, object]) -> bool:
        if obj_type != self.obj_type:
            return False
        if self.where is None:
            return True
        attr, cmp, val = self.where
        v = attrs.get(attr)
        if v is None:
            return True  # fail-closed: unknown attribute -> constraint applies
        a = ORDINALS.get(str(v), v)
        b = ORDINALS.get(val, val)
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            pass
        return {">=": a >= b, ">": a > b, "<=": a <= b, "<": a < b, "==": a == b}[cmp]


@dataclass
class Policy:
    name: str
    exempt: FrozenSet[str]
    objects: Dict[str, Tuple[str, ...]]
    bindings: List[Binding]
    constraints: List[Constraint]
    source: str = ""

    def digest(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()

    def bind(self, tool: str, args: Dict[str, object]) -> Optional[Tuple[str, str, str]]:
        """Map a concrete tool call to (op, obj_type, obj_id), or None if unbound."""
        for b in self.bindings:
            if b.tool == tool:
                key = "|".join(str(args[k]) for k in b.key_args)
                return b.op, b.obj_type, f"{b.obj_type}({key})"
        return None


_DUR = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _dur(tok: str) -> int:
    m = re.fullmatch(r"(\d+)([smhd])", tok)
    if not m:
        raise SyntaxError(f"bad duration {tok!r}")
    return int(m.group(1)) * _DUR[m.group(2)]


def _strip_comments(src: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in src.splitlines())


def parse(src: str) -> Policy:
    text = _strip_comments(src)
    m = re.match(r"\s*policy\s+(\w+)\s*\{(.*)\}\s*$", text, re.S)
    if not m:
        raise SyntaxError("expected: policy <id> { ... }")
    name, body = m.group(1), m.group(2)
    exempt: set = set()
    objects: Dict[str, Tuple[str, ...]] = {}
    bindings: List[Binding] = []
    constraints: List[Constraint] = []

    # constraints first (they contain braces)
    for cm in re.finditer(r"constraint\s+(\w+)\s*\{(.*?)\n\s*\}", body, re.S):
        constraints.append(_parse_constraint(cm.group(1), cm.group(2)))
    rest = re.sub(r"constraint\s+\w+\s*\{.*?\n\s*\}", "", body, flags=re.S)

    for em in re.finditer(r"exempt\s+([\w:,\s]+?)\n", rest + "\n"):
        exempt |= {x.strip() for x in re.split(r"[\s,]+", em.group(1)) if x.strip()}
    for om in re.finditer(r"object\s+(\w+)\s+key\s*\(([^)]*)\)", rest):
        objects[om.group(1)] = tuple(a.strip() for a in om.group(2).split(","))
    for bm in re.finditer(
        r"bind\s+([\w.]+)\s*\(([^)]*)\)\s*->\s*(\w+)\s+on\s+(\w+)\s*\(([^)]*)\)", rest, re.S
    ):
        targs = tuple(a.strip() for a in bm.group(2).split(",") if a.strip() and a.strip() != "..")
        kargs = tuple(a.strip() for a in bm.group(5).split(","))
        if bm.group(4) not in objects:
            raise SyntaxError(f"binding to undeclared object {bm.group(4)}")
        bindings.append(Binding(bm.group(1), targs, bm.group(3), bm.group(4), kargs))
    return Policy(name, frozenset(exempt), objects, bindings, constraints, src)


def _parse_constraint(name: str, body: str) -> Constraint:
    s = " ".join(body.split())
    kw: Dict[str, object] = {"name": name}
    m = re.search(r"\bon\s+(\w+)(?:\s+where\s+(\w+)\s*(>=|<=|==|>|<)\s*(\w+))?", s)
    if not m:
        raise SyntaxError(f"{name}: missing 'on'")
    kw["obj_type"] = m.group(1)
    if m.group(2):
        kw["where"] = (m.group(2), m.group(3), m.group(4))
    m = re.search(r"\bconflict\s*\{([^}]*)\}", s)
    if m:
        kw["conflict"] = frozenset(x.strip() for x in m.group(1).split(",") if x.strip())
    m = re.search(r"\brequire\s+(\d+)\s+(?:independent|separated)\s+(\w+)(?:\s+before\s+(\w+))?", s)
    if m:
        kw["require_n"], kw["require_op"], kw["before_op"] = int(m.group(1)), m.group(2), m.group(3)
        if not m.group(3):
            raise SyntaxError(f"{name}: 'require' needs 'before <op>'")
    if "conflict" not in kw and "require_n" not in kw:
        raise SyntaxError(f"{name}: needs 'conflict' or 'require'")
    m = re.search(r"\bwindow\s+(\w+)", s)
    if not m:
        raise SyntaxError(f"{name}: missing window")
    kw["window"] = _dur(m.group(1))
    m = re.search(
        r"\b(?:independence|separation)\s+(identity|accountability|informational|information|substrate)"
        r"(?:\s*\(([^)]*)\)\s*history\s+(\d+)\s+max\s+(\d+))?", s)
    if not m:
        raise SyntaxError(f"{name}: missing separation level")
    kw["level"] = LEVELS[m.group(1)]
    if kw["level"] == 3:
        if not m.group(2):
            raise SyntaxError(f"{name}: substrate needs (dims) history H max k")
        kw["dims"] = tuple(x.strip() for x in m.group(2).split(","))
        kw["history"], kw["kappa"] = int(m.group(3)), int(m.group(4))
    m = re.search(r"\bartifact\s+(\w+)", s)
    if m:
        kw["artifact"] = m.group(1)
    m = re.search(r"\bon_violation\s+(deny|audit_only|escalate)(?:\s*\(([^)]*)\))?", s)
    if not m:
        raise SyntaxError(f"{name}: missing on_violation")
    kw["on_violation"] = m.group(1)
    if m.group(2):
        kw["escalate_to"] = m.group(2).strip()
    return Constraint(**kw)  # type: ignore[arg-type]
