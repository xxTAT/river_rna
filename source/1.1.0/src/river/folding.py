"""Constraint files, folding output parsers and structure writers.

RNAstructure (`Fold`)  : ``.const`` file with a Pairs and a Forbids section.
ViennaRNA (`RNAcofold`): two fragments in one FASTA plus a constraint line
                         ``<<<((<<<&>>))>>>`` (`--enforceConstraint`).
"""
from __future__ import annotations

import logging



from .matrix_utils import BEDPE_HEADER

log = logging.getLogger("river")


# ------------------------------------------------------------ RNAstructure
def write_constraint(path, pairs, forbids=()):
    """`.const` for Fold; `pairs`/`forbids` are 1-based local (i, j)."""
    with open(path, "w") as f:
        f.write("DS:\n-1\nSS:\n-1\nMod:\n-1\nPairs:\n")
        for a, b in pairs:
            f.write(f"{a} {b}\n")
        f.write("-1 -1\nFMN:\n-1\nForbids:\n")
        for a, b in forbids:
            f.write(f"{a} {b}\n")
        f.write("-1 -1\n")


def forbidden_pairs(region_start, rect):
    """All 1-based local pairs of a prohibited rectangle"""
    s1, e1, s2, e2 = (int(x) for x in rect)
    out = [(l + 1 - region_start, r + 1 - region_start) for l in range(s1, e1) for r in range(s2, e2)]
    if len(out) > 2_000_000:
        log.warning("    %d forbidden pairs - RNAstructure may be slow", len(out))
    return out


def parse_ct(ct_file, offset):
    """All structures of a CT file as absolute 0-based/1-based (i, j) pairs.
    `offset` is added to the CT index, i.e. offset = region_start gives 1-based positions of the full RNA.
    """
    structs, energy, cur = {}, {}, None
    with open(ct_file) as f:
        for line in f:
            sub = line.split()
            if not sub:
                continue
            is_data = (len(sub) >= 6 and "ENERGY" not in sub and sub[0].isdigit() and sub[4].lstrip("-").isdigit())
            if not is_data:
                cur = f"s{len(structs) + 1}"
                structs[cur] = []
                try:
                    energy[cur] = float(sub[3]) if "ENERGY" in line and len(sub) >= 4 else 10000.0
                except ValueError:
                    energy[cur] = 10000.0
                continue
            if cur is None or int(sub[4]) == 0:
                continue
            a, b = offset + int(sub[0]), offset + int(sub[4])
            if a < b:
                structs[cur].append((a, b))
    return structs, energy


def best_structure(structs, energy):
    if not structs:
        return []
    best = min(energy, key=energy.get) if energy else next(iter(structs))
    return sorted(structs[best])


def to_dot_bracket(pairs, length):
    brackets = ["()", "[]", "{}", "<>"]
    levels = [[] for _ in brackets]
    s = ["."] * length
    for a, b in sorted(pairs):
        for lv, lst in enumerate(levels):
            if all(not (x < a < y < b or a < x < b < y) for x, y in lst):
                lst.append((a, b))
                s[a - 1], s[b - 1] = brackets[lv]
                break
    return "".join(s)


# ---------------------------------------------------------------- ViennaRNA
def cofold_constraint(pairs, left_len, right_len, left_pad=0, right_pad=0):
    """
    `pairs` are (left_local, right_local) indices of the *unpadded* fragments;
    the fragments written to the FASTA carry `left_pad` extra nucleotides at
    the 3' end of the left and `right_pad` at the 5' end of the right fragment.
    """
    left = ["<"] * (left_len + left_pad)
    right = [">"] * (right_len + right_pad)
    for a, b in pairs:
        left[a] = "("
        right[b + right_pad] = ")"
    return "".join(left) + "&" + "".join(right)


def write_cofold_fasta(path, name, left_seq, right_seq, constraint):
    with open(path, "w") as f:
        f.write(f">{name}\n{left_seq}&{right_seq}\n{constraint}\n")


def read_cofold_fasta(path):
    name = left = right = cons = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:]
            elif set(line.upper()) <= set("ACGUTN&"):
                left, right = line.split("&")
            else:
                cons = line
    return name, left, right, cons


def relax_constraint(left_dot, right_dot):
    """
    release the innermost and all isolated pairs.
    Used when `RNAcofold --enforceConstraint` cannot satisfy the constraint and returns nothing.
    """
    left = list(left_dot)
    right = list(right_dot)
    lp = [i for i, s in enumerate(left) if s == "("]
    rp = [i for i, s in enumerate(right) if s == ")"]
    if lp:
        left[max(lp)] = "<"
    if rp:
        right[min(rp)] = ">"
    lp = [i for i, s in enumerate(left) if s == "("]
    rp = [i for i, s in enumerate(right) if s == ")"]
    lset, rset = set(lp), set(rp)
    for a, b in zip(lp, rp[::-1]):
        if (a - 1 not in lset and a + 1 not in lset
                and b - 1 not in rset and b + 1 not in rset):
            left[a] = "<"
            right[b] = ">"
    return "".join(left), "".join(right)


def parse_cofold(text):
    """Return (left_dot, right_dot) of an `RNAcofold` run."""
    seq = dot = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        body = line.split()[0]
        if set(body.upper()) <= set("ACGUTN&"):
            seq = body
        elif set(body) <= set(".()&<>[]{}") and "&" in body and seq is not None:
            dot = body
            break
    if seq is None or dot is None:
        return None
    return tuple(dot.split("&")[:2])


def cofold_energy(text):
    """Free energy (kcal/mol) printed by RNAcofold"""
    import re

    for line in text.splitlines():
        body = line.strip()
        if not body or body.startswith(">") or "&" not in body:
            continue
        m = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*\)\s*$", body)
        if m and set(body.split()[0]) <= set(".()&<>[]{}"):
            return float(m.group(1))
    return None


def pairs_from_dots(left_dot, right_dot, left_pad=0, right_pad=0):
    """Inter-molecular pairs of a cofold result, the padding stripped."""
    left_dot = left_dot[:-left_pad] if left_pad else left_dot
    right_dot = right_dot[right_pad:] if right_pad else right_dot
    lp = [i for i, s in enumerate(left_dot) if s == "("]
    rp = [i for i, s in enumerate(right_dot) if s == ")"]
    if len(lp) != len(rp):
        log.warning("    cofold: %d '(' vs %d ')' - extra pairs ignored", len(lp), len(rp))
    return [(a, b) for a, b in zip(lp, rp[::-1])]


# ------------------------------------------------------------------ outputs
def write_structure_outputs(prefix, seq, segments, pairs, chrom, support=None, restrict=None):
    """Write `<prefix>.ct`, `<prefix>.bedpe` and `<prefix>_annotation.txt`.

    segments : [(start, end), ...] absolute 0-based half-open blocks that make
               up the folded sequence (one block = diagonal/whole region,
               two blocks = the two arms of a feature)
    pairs    : absolute 0-based (a, b) pairs
    support  : callable (a, b) -> interaction value, written to the annotation file
    restrict : optional (s1, e1, s2, e2); pairs inside this rectangle are
               written to `<prefix>_cluster.bedpe` as well

    Returns the base pairs that were written, as absolute 0-based (a, b).
    """
    pos = [p for s, e in segments for p in range(s, e)]
    local = {p: i for i, p in enumerate(pos)}
    pair_of = {}
    for a, b in pairs:
        if a in local and b in local:
            pair_of[a] = b
            pair_of[b] = a
    n = len(pos)
    # 1-based CT indices of the first / last base of every segment; the backbone
    # link is cut there (prev = 0 / next = 0) so that VARNA draws the segments
    # as separate strands instead of joining them at the junction
    firsts, lasts, cut = set(), set(), 0
    for s, e in segments:
        if e > s:
            firsts.add(cut + 1)
            lasts.add(cut + e - s)
            cut += e - s
    with open(f"{prefix}.ct", "w") as f:
        f.write(f"{n}\n")
        for i, p in enumerate(pos, 1):
            j = pair_of.get(p)
            prev_i = 0 if i in firsts else i - 1
            next_i = 0 if i in lasts else i + 1
            f.write(f"{i} {seq[p]} {prev_i} {next_i} "
                    f"{local[j] + 1 if j is not None else 0} {i}\n")
    kept = sorted({(min(a, b), max(a, b)) for a, b in pair_of.items()})
    with open(f"{prefix}.bedpe", "w") as f:
        f.write(BEDPE_HEADER)
        for a, b in kept:
            f.write(f"{chrom}\t{a}\t{a + 1}\t{chrom}\t{b}\t{b + 1}\n")
    if restrict is not None:
        s1, e1, s2, e2 = (int(x) for x in restrict)
        with open(f"{prefix}_cluster.bedpe", "w") as f:
            f.write(BEDPE_HEADER)
            for a, b in kept:
                if s1 <= a < e1 and s2 <= b < e2:
                    f.write(f"{chrom}\t{a}\t{a + 1}\t{chrom}\t{b}\t{b + 1}\n")
    if support is not None:
        with open(f"{prefix}_annotation.txt", "w") as f:
            for i, p in enumerate(pos, 1):
                j = pair_of.get(p)
                f.write(f"{i}\t{0 if j is None else support(min(p, j), max(p, j))}\n")
    dbn = to_dot_bracket([(local[a] + 1, local[b] + 1) for a, b in kept], n)
    cut, blocks = 0, []
    for s, e in segments:
        blocks.append(dbn[cut:cut + e - s])
        cut += e - s
    with open(f"{prefix}.dbn", "w") as f:
        f.write(">" + "_".join(f"{s}_{e}" for s, e in segments) + "\n")
        f.write("&".join(seq[s:e] for s, e in segments) + "\n")
        f.write("&".join(blocks) + "\n")
    return kept
