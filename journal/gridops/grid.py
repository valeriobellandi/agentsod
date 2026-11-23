"""Power-flow model behind the grid change-management workflow.

A setpoint change sets the active power of one static generator (a PV or wind
plant) of the MV Oberrhein network shipped with pandapower. Executed changes have
observable effects: a change is *harmful* if, applied to the base case, it causes a
voltage outside [0.95, 1.05] p.u. or a line or transformer loading above 100 %.
Each change is evaluated against the base case (we assume executed changes are
reverted at the end of their dispatch interval), and results are cached.
"""
from __future__ import annotations

import copy
import logging
import random
import warnings
from functools import lru_cache
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)

import pandapower as pp  # noqa: E402
import pandapower.networks as pn  # noqa: E402

warnings.filterwarnings("ignore")  # pandapower resets filters on import

V_MIN, V_MAX, LOAD_MAX = 0.95, 1.05, 100.0


@lru_cache(maxsize=1)
def base_net():
    net = pn.mv_oberrhein()
    net.sgen["scaling"] = 1.0  # generation case: plants at nominal output
    pp.runpp(net, numba=False)
    return net


@lru_cache(maxsize=4096)
def simulate(sgen: int, p_mw: float) -> Dict[str, float]:
    """Power flow with sgen `sgen` set to `p_mw`; returns extreme values and a verdict."""
    net = copy.deepcopy(base_net())
    net.sgen.at[sgen, "p_mw"] = p_mw
    try:
        pp.runpp(net, numba=False)
    except Exception:  # did not converge: treat as a violation
        return {"converged": False, "vm_max": float("nan"), "vm_min": float("nan"),
                "line_max": float("nan"), "trafo_max": float("nan"), "violation": True}
    vmax, vmin = float(net.res_bus.vm_pu.max()), float(net.res_bus.vm_pu.min())
    lmax = float(net.res_line.loading_percent.max())
    tmax = float(net.res_trafo.loading_percent.max())
    viol = vmax > V_MAX or vmin < V_MIN or lmax > LOAD_MAX or tmax > LOAD_MAX
    return {"converged": True, "vm_max": round(vmax, 4), "vm_min": round(vmin, 4),
            "line_max": round(lmax, 1), "trafo_max": round(tmax, 1), "violation": viol}


def change_pools(seed: int = 7, n_benign: int = 60, n_harmful: int = 40
                 ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Pools of verified benign and harmful setpoint changes (sgen index, p_mw)."""
    rng = random.Random(seed)
    net = base_net()
    idx = list(net.sgen.index)
    benign, harmful = [], []
    while len(benign) < n_benign:
        i = rng.choice(idx)
        p = round(float(net.sgen.at[i, "p_mw"]) * rng.uniform(0.3, 2.0), 3)
        if not simulate(i, p)["violation"]:
            benign.append((i, p))
    tries = 0
    while len(harmful) < n_harmful and tries < 2000:
        tries += 1
        i = rng.choice(idx)
        p = round(rng.uniform(6.0, 20.0), 1)
        if simulate(i, p)["violation"]:
            harmful.append((i, p))
    return benign, harmful
