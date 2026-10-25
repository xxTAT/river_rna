"""Structural-domain segmentation used for medium/long RNAs

Thresholds: max_len (600), edge_len (100), gap_len (200) are parameters.
"""
from __future__ import annotations

from pathlib import Path

from .matrix_utils import BEDPE_HEADER


def split_domain(structure_bedpe, length, chrom):
    out_file = str(structure_bedpe).replace(".bedpe", "") + "_domain.bedpe"
    pairs, pixels, pixels_list, overlap, domain_loc = {}, {}, [], set(), []
    n = 0
    with open(structure_bedpe) as sb:
        for line in sb:
            if "chrom1" in line or not line.strip():
                continue
            n += 1
            pid = f"pixel{n}"
            pixels_list.append(pid)
            sub = line.split()
            pixels[pid] = (int(sub[1]), int(sub[4]))
            pairs[int(sub[1])] = int(sub[4])
            pairs[int(sub[4])] = int(sub[1])
    for pid in pixels_list:
        left, right = pixels[pid]
        left_len = left + 1
        right_len = length - 1 - right + 1
        if left_len <= right_len:
            for i in range(left - 1, -1, -1):
                if int(pairs.get(i, 0)) > right:
                    overlap.add(pid)
                    break
        else:
            for i in range(right + 1, length):
                if int(pairs.get(i, length)) < left:
                    overlap.add(pid)
                    break
    with open(out_file, "w") as of:
        of.write(BEDPE_HEADER)
        for pid in pixels_list:
            if pid in overlap:
                continue
            left, right = pixels[pid]
            of.write(f"{chrom}\t{left}\t{right + 1}\t{chrom}\t{left}\t{right + 1}\n")
            domain_loc.append((left, right + 1))
    return domain_loc


def find_blank(domain_locs, length):
    ds = sorted(domain_locs, key=lambda x: x[0])
    blank = []
    for index, domain in enumerate(ds):
        if index == 0:
            if domain[0] > 0:
                blank.append((0, domain[0]))
            continue
        if index == len(ds) - 1:
            if domain[1] < length - 1:
                blank.append((domain[1], length - 1))
            continue
        last = ds[index - 1]
        if domain[0] - last[1] > 0:
            blank.append((last[1], domain[0]))
    return blank


def _walk(tmp_list, start, limit, final, cond_end):
    end = start
    n, total = 0, len(tmp_list)
    while n < total:
        this = tmp_list[n]
        end = this[1]
        if end - start + 1 > limit:
            final.append((start, this[0]))
            start = this[0]
            end = start
        n += 1
        if n == total:
            cond_end(start, end)
    return start


def connect_domain_and_fill_blank(domain_locs, bedpe_file_name, length, chrom,
                                  max_len=600, edge_len=100, gap_len=200):
    ds = sorted(domain_locs, key=lambda x: x[0])
    large = [d for d in ds if d[1] - d[0] + 1 > max_len]
    final = []
    for index, domain in enumerate(large):
        if index == 0:
            if domain[0] < max_len:
                if domain[0] < edge_len:
                    final.append((0, domain[1]))
                    continue
                final.append((0, domain[0]))
                final.append((domain[0], domain[1]))
            else:
                ti = ds.index(domain)
                if ti == 0:
                    final.append((0, domain[0]))
                    final.append((domain[0], domain[1]))
                    continue
                tmp = ds[:ti]

                def _end(s, e, d=domain):
                    if e < d[1] - 1:
                        final.append((s, d[0]))
                _walk(tmp, 0, max_len, final, _end)
                final.append((domain[0], domain[1]))
        elif index == len(large) - 1:
            if length - 1 - domain[1] + 1 < max_len:
                if length - 1 - domain[1] + 1 < edge_len:
                    final.append((domain[0], length))
                    continue
                final.append((domain[1], length))
                final.append((domain[0], domain[1]))
            else:
                ti = ds.index(domain)
                if ti == len(ds) - 1:
                    final.append((domain[1], length))
                    final.append((domain[0], domain[1]))
                    continue
                tmp = ds[ti + 1:]

                def _end(s, e):
                    if e < length:
                        final.append((s, length))
                _walk(tmp, domain[1], max_len, final, _end)
                final.append((domain[0], domain[1]))
        elif index == len(large) - 2:
            last_list = [domain, large[index - 1]]
            this_list = [large[index + 1], domain]
            for i in range(2):
                last, this = last_list[i], this_list[i]
                if this[0] - last[1] + 1 < max_len:
                    if this[0] - last[1] + 1 < gap_len:
                        final.append((last[0], this[0]))
                        continue
                    final.append((last[1], this[0]))
                    final.append((last[0], last[1]))
                else:
                    li, ti = ds.index(last), ds.index(this)
                    if ti - li == 1:
                        final.append((last[1], this[0]))
                        final.append((last[0], last[1]))
                        continue
                    tmp = ds[li + 1:ti]

                    def _end(s, e, t=this):
                        if e <= t[0]:
                            final.append((s, t[0]))
                    _walk(tmp, last[1], max_len, final, _end)
                    final.append((last[0], last[1]))
        else:
            last, this = large[index - 1], domain
            if this[0] - last[1] + 1 < max_len:
                if this[0] - last[1] + 1 < edge_len:
                    final.append((last[0], this[0]))
                    continue
                final.append((last[1], this[0]))
                final.append((last[0], last[1]))
            else:
                li, ti = ds.index(last), ds.index(this)
                if ti - li == 1:
                    final.append((last[1], this[0]))
                    final.append((last[0], last[1]))
                    continue
                tmp = ds[li + 1:ti]

                def _end(s, e, t=this):
                    if e <= t[0]:
                        final.append((s, t[0]))
                _walk(tmp, last[1], max_len, final, _end)
                if index == 1:
                    continue
                final.append((last[0], last[1]))

    for b in find_blank(final, length):
        if b[1] - b[0] + 1 < edge_len:
            lr = [d for d in final if d[1] == b[0] or d[1] == b[1]]
            if len(lr) == 2:
                l_len = lr[0][1] - lr[0][0] + 1
                r_len = lr[1][1] - lr[1][0] + 1
                if l_len > r_len:
                    final.remove(lr[1])
                    final.append((b[0], lr[1][1]))
                else:
                    final.remove(lr[0])
                    final.append((lr[0][0], b[1]))
            else:
                mn = min(b[0], lr[0][0])
                mx = max(lr[0][1], b[1])
                final.remove(lr[0])
                final.append((mn, mx))
        elif b[1] - b[0] + 1 < max_len:
            final.append(b)
        else:
            small = [d for d in ds if d[0] >= b[0] and d[1] <= b[1]]
            if not small:
                final.append(b)
            else:
                def _end(s, e, bb=b):
                    if e < bb[1]:
                        final.append((s, bb[1]))
                _walk(small, b[0], max_len, final, _end)

    out = str(bedpe_file_name).replace(".bedpe", "") + "_final_domain_locs.bedpe"
    with open(out, "w") as of:
        of.write(BEDPE_HEADER)
        for d in final:
            of.write(f"{chrom}\t{d[0]}\t{d[1]}\t{chrom}\t{d[0]}\t{d[1]}\n")
    return final


def fallback_domains(length, max_len):
    """Used only when domain construction yields nothing / fails."""
    return [(s, min(s + max_len, length)) for s in range(0, length, max_len)]
