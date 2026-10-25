"""Stems and the greedy conflict/pseudoknot-free structure tree.

Shared by step3 (whole RNA), step4 (structure feature) and
step5 (alternative structures).  All coordinates handed to these functions are
*local* indices of the region that will be folded:

* one fragment  -> position - region_start
* two fragments -> left:  position - left_start
                   right: position - right_start + left_length

so that a unit is always a list of (i, j) with 0 <= i < j < length.
"""
from __future__ import annotations

import numpy as np


def chain_units(pixels):
    """ Chain candidate pixels along anti-diagonals: (i, j) -> (i+1, j-1) """
    pixels = list(pixels)
    pos = set(pixels)
    unit_loc, pixel_unit, checked = {}, {}, set()
    k = 0
    for rc in pixels:
        nb = (rc[0] + 1, rc[1] - 1)
        if nb not in pos:
            continue
        if rc not in checked:
            k += 1
            uid = f"s{k}"
            unit_loc[uid] = [rc, nb]
        else:
            uid = pixel_unit[rc]
            unit_loc[uid].append(nb)
        pixel_unit[rc] = uid
        pixel_unit[nb] = uid
        checked.add(rc)
        checked.add(nb)
    return unit_loc


def _span(locs):
    left = [a for a, _ in locs]
    right = [b for _, b in locs]
    return min(left), max(left), min(right), max(right)


def is_pseudoknot(path, unit_loc, uid):
    sls, sle, srs, sre = _span(unit_loc[uid])
    for p in path:
        if p == "s0":
            continue
        ls, le, rs, re_ = _span(unit_loc[p])
        if (le < sls < rs < srs > re_) or (srs > le and sre < rs and sle < ls):
            return True
    return False


def compatible(cands, unit_loc, path, length):
    used = np.zeros(length, dtype=bool)
    for p in path:
        if p == "s0":
            continue
        for a, b in unit_loc[p]:
            used[a] = used[b] = True
    out = []
    for u in cands:
        if is_pseudoknot(path, unit_loc, u):
            continue
        if all(not used[a] and not used[b] for a, b in unit_loc[u]):
            out.append(u)
    return out


def build_structure_paths(unit_scores, unit_loc, length):
    """Greedy tree over score tiers; returns deterministic candidate paths."""
    ranked = sorted(unit_scores, key=lambda x: x[1], reverse=True)
    if not ranked:
        return []
    value = dict(ranked)
    top = ranked[0][1]
    frontier = [["s0", u] for u in compatible([u for u, v in ranked if v == top], unit_loc, ["s0"], length)]
    paths = []
    while frontier:
        nxt = []
        for path in frontier:
            last_v = value[path[-1]]
            rest = [(u, v) for u, v in ranked if v < last_v]
            chosen = []
            while rest:
                tier = [u for u, v in rest if v == rest[0][1]]
                chosen = compatible(tier, unit_loc, path, length)
                if chosen:
                    break
                rest = [(u, v) for u, v in rest if v != rest[0][1]]
            if not rest:
                paths.append(path)
                continue
            nxt.extend(path + [u] for u in chosen)
        frontier = nxt
    return sorted({tuple(p) for p in paths}, key=lambda p: (len(p), p))


def path_pairs(path, unit_loc):
    """All (i, j) pairs of a candidate structure, sorted."""
    pairs = []
    for uid in path:
        if uid == "s0":
            continue
        pairs.extend(unit_loc[uid])
    return sorted(set(pairs))
