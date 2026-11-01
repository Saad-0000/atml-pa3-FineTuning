"""
Calculation of rollout quality scores: mean log-prob, self-certainty, visual engagement, and VisConf.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_mean_logp(logp_list: Any) -> float:
    """L_i: Mean sampled-token log-probability."""
    if logp_list is None or len(logp_list) == 0:
        return 0.0
    return float(np.mean(logp_list))


def compute_mean_self_certainty(kl_list: Any) -> float:
    """K_i: Mean self-certainty D_KL(U || p)."""
    if kl_list is None or len(kl_list) == 0:
        return 0.0
    return float(np.mean(kl_list))


def compute_visual_engagement(
    a_image: List[float],
    a_prompt: List[float],
    a_generated: List[float],
) -> Tuple[float, List[float]]:
    """V_i: MathVista visual engagement metric.

    v_i,t = A_image_i,t / (A_image_i,t + A_prompt_i,t + A_generated_i,t)
    V_i = (1 / T_i) * sum_t v_i,t
    """
    n_tokens = min(len(a_image), len(a_prompt), len(a_generated))
    if n_tokens == 0:
        return 0.0, []

    v_tokens = []
    zero_cnt = 0
    for t in range(n_tokens):
        denom = float(a_image[t] + a_prompt[t] + a_generated[t])
        if denom > 1e-8:
            v_t = float(a_image[t]) / denom
        else:
            v_t = 0.0
            zero_cnt += 1
        v_tokens.append(v_t)

    mean_v = float(np.mean(v_tokens)) if v_tokens else 0.0
    return mean_v, v_tokens


def compute_visconf(
    candidates_kl_tokens: List[List[float]],
    candidates_v_tokens: List[List[float]],
) -> List[Dict[str, float]]:
    """Calculates VisConf Z_i for each candidate in revealed set G.

    Protocol:
    1. Pool token-level k_i,t from ONLY the revealed candidates in G.
    2. Compute linear-interpolation percentiles Q.05(G) and Q.95(G).
    3. c_i,t = clip_[0,1]((k_i,t - Q.05(G)) / max(Q.95(G) - Q.05(G), 1e-8)).
    4. C_i = mean_t c_i,t.
    5. Z_i = V_i * C_i.
    """
    all_k = [k for k_list in candidates_kl_tokens for k in k_list]
    if not all_k:
        return [{"C_i": 0.0, "Z_i": 0.0, "V_i": 0.0} for _ in candidates_kl_tokens]

    q05 = float(np.percentile(all_k, 5, method="linear"))
    q95 = float(np.percentile(all_k, 95, method="linear"))
    scale = max(q95 - q05, 1e-8)

    results = []
    for i in range(len(candidates_kl_tokens)):
        k_list = candidates_kl_tokens[i]
        v_list = candidates_v_tokens[i] if i < len(candidates_v_tokens) else []

        if k_list is not None and len(k_list) > 0:
            c_tokens = np.clip((np.array(k_list, dtype=np.float64) - q05) / scale, 0.0, 1.0)
            c_i = float(np.mean(c_tokens))
        else:
            c_i = 0.0

        v_i = float(np.mean(v_list)) if (v_list is not None and len(v_list) > 0) else 0.0
        z_i = v_i * c_i

        results.append({
            "C_i": c_i,
            "Z_i": z_i,
            "V_i": v_i,
            "q05": q05,
            "q95": q95,
        })

    return results


def calibrate_visconf_for_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calibrate VisConf scores for a list of revealed candidate dicts."""
    kl_tokens = [c.get("kl_u_p", [c.get("K_i", 1.0)]) for c in candidates]
    v_tokens = [[c.get("V_i", 0.0)] for c in candidates]

    visconf_res = compute_visconf(kl_tokens, v_tokens)
    updated = []
    for c, v_info in zip(candidates, visconf_res):
        c_copy = dict(c)
        c_copy["C_i"] = v_info["C_i"]
        c_copy["Z_i"] = v_info["Z_i"]
        c_copy["V_i"] = v_info["V_i"]
        updated.append(c_copy)
    return updated
