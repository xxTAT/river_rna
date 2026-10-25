"""Step 2: contact-map dumping and Canny-like structure feature detection."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .matrix_utils import (BEDPE_HEADER, double_threshold, ensure_dir, gaussian_blur,
                           gaussian_kernel, get_oe, load_dump_matrix, mask_by_oe, nms,
                           read_bedpe_starts, write_pixel_bedpe, zero_diagonal_cross)

log = logging.getLogger("river")


def dump_matrices(tool, hic, chrom, resolutions, norm_order, outdir, resume=False):
    base = Path(hic).name.replace(".hic", "")
    files = {}
    for res in resolutions:
        for norm in norm_order:
            out = Path(outdir) / f"{base}_{norm}_{res}.matrix"
            if resume and out.exists() and out.stat().st_size > 0:
                files[res] = out
                break
            if tool.dump_observed(hic, chrom, res, norm, out):
                files[res] = out
                log.info("  dumped %s bp with %s normalisation", res, norm)
                break
            if out.exists():
                out.unlink()
        else:
            log.warning("  resolution %s bp is empty for all normalisations", res)
    return files


def canny_detect(matrix_file, res, params, chrom, zero_cross, mode, oe_mask):
    raw = load_dump_matrix(matrix_file, res)
    if zero_cross:
        zero_diagonal_cross(raw)
    kernel = gaussian_kernel(params.gaussian_sigma, params.gaussian_ksize)
    stem = str(matrix_file).replace(".matrix", "")
    if oe_mask:
        oe = get_oe(raw)
        mask_by_oe(raw, oe)
        out = Path(f"{stem}_{res}_OEmask.bedpe")
    else:
        out = Path(f"{stem}_{res}.bedpe")
    filt = gaussian_blur(raw, kernel)
    if mode == "mid":
        filt = nms(filt)
    dt, tl, th = double_threshold(filt, params.canny_tlf, params.canny_thf)
    log.info("  canny %s bp (%s): TL=%.4g TH=%.4g", res, mode, tl, th)
    rows, cols = np.nonzero(dt == 1)
    write_pixel_bedpe(out, rows, cols, res, chrom)
    return out


def low_res_to_mid_res(bedpe, mid_res, chrom):
    """Map low-resolution edge pixels onto the mid-resolution grid"""
    out = Path(str(bedpe).replace(".bedpe", "") + f"_remake{mid_res}.bedpe")
    s1, e1, s2, e2 = read_bedpe_starts(bedpe)
    px = set()
    for a, b, c, d in zip(s1.tolist(), e1.tolist(), s2.tolist(), e2.tolist()):
        if a % 10 == 5:
            a -= 5
        if b % 10 == 5:
            b += 5
        if c % 10 == 5:
            c -= 5
        if d % 10 == 5:
            d -= 5
        px.add((a, a + mid_res, c, c + mid_res))
        px.add((a, a + mid_res, d - mid_res, d))
        px.add((b - mid_res, b, c, c + mid_res))
        px.add((b - mid_res, b, d - mid_res, d))
    with open(out, "w") as f:
        f.write(BEDPE_HEADER)
        for p in sorted(px):
            f.write(f"{chrom}\t{p[0]}\t{p[1]}\t{chrom}\t{p[2]}\t{p[3]}\n")
    return out


def diagonal_fill(bedpe, mid_res, length, chrom):
    n = int(length / mid_res) + 1
    t = np.zeros((n, n))
    s1, _, s2, _ = read_bedpe_starts(bedpe)
    r, c = s1 // mid_res, s2 // mid_res
    ok = (r < n) & (c < n)
    t[r[ok], c[ok]] = 1
    for i in range(n):
        if i - 1 >= 0:
            if i + 1 <= n - 1 and t[i - 1, i + 1] == 1 and t[i + 1, i - 1] == 1:
                t[i, i] = 1
            if i + 2 <= n - 1 and t[i - 1, i + 2] == 1 and t[i + 2, i - 1] == 1:
                t[i, i + 1] = t[i + 1, i] = 1
        if i - 2 >= 0 and i + 2 <= n - 1:
            if t[i - 2, i + 2] == 1 and t[i + 2, i - 2] == 1:
                t[i - 1, i + 1] = t[i + 1, i - 1] = t[i, i] = 1
    out = Path(str(bedpe).replace(".bedpe", "") + "_fill_diagonal.bedpe")
    rows, cols = np.nonzero(t == 1)
    write_pixel_bedpe(out, rows, cols, mid_res, chrom)
    return out


def merge_bedpe(a, b, chrom):
    out = Path(str(a).replace(".bedpe", "") + "_merge_" + Path(b).name)
    px = set()
    for f in (a, b):
        s1, e1, s2, e2 = read_bedpe_starts(f)
        px.update(zip(s1.tolist(), e1.tolist(), s2.tolist(), e2.tolist()))
    with open(out, "w") as fh:
        fh.write(BEDPE_HEADER)
        for p in sorted(px):
            fh.write(f"{chrom}\t{p[0]}\t{p[1]}\t{chrom}\t{p[2]}\t{p[3]}\n")
    return out


def rri_gaussian_filter(canny_bedpe, rri_file, mid_res, length, params, chrom):
    n = int(length / mid_res) + 1
    canny = np.zeros((n, n))
    s1, _, s2, _ = read_bedpe_starts(canny_bedpe)
    r, c = s1 // mid_res, s2 // mid_res
    ok = (r < n) & (c < n)
    canny[r[ok], c[ok]] = 1
    canny[c[ok], r[ok]] = 1
    d = np.loadtxt(rri_file, usecols=(0, 1, 2), ndmin=2)
    d = d[~np.isnan(d[:, 2])]
    r = (d[:, 0] // mid_res).astype(int)
    c = (d[:, 1] // mid_res).astype(int)
    ok = (r < n) & (c < n)
    rri = np.zeros((n, n))
    rri[r[ok], c[ok]] = d[ok, 2]
    rri[c[ok], r[ok]] = d[ok, 2]
    g = gaussian_blur(rri, gaussian_kernel(params.gaussian_sigma, params.gaussian_ksize))
    cutoff = np.max(g) * params.rri_ctf
    log.info("  rri gaussian cutoff = %.4g", cutoff)
    keep = (canny == 1) & (g > cutoff) & ~(rri <= 0)
    out = Path(str(canny_bedpe).replace(".bedpe", "") + "_filter_gaussian.bedpe")
    rows, cols = np.nonzero(keep)
    write_pixel_bedpe(out, rows, cols, mid_res, chrom)
    return out


def cluster_pixels(tool, filtered_bedpe, mid_res, chrom):
    """Auxiliary output: clusters of adjacent feature pixels (bedtools)."""
    pix = Path(str(filtered_bedpe).replace(".bedpe", "_pixels.bedpe"))
    n = 0
    with open(filtered_bedpe) as f, open(pix, "w") as o:
        for line in f:
            if "chrom1" in line:
                continue
            sub = line.split()
            if int(sub[1]) > int(sub[4]):
                continue
            n += 1
            o.write(line.rstrip("\n") + f"\tpixel{n}\n")
    p2p = Path(str(pix).replace("_pixels.bedpe", "_pixel2pixel.txt"))
    tool.pairtopair(pix, pix, mid_res, p2p)
    pairs, pixels = {}, {}
    with open(p2p) as f:
        for line in f:
            sub = line.split()
            if len(sub) < 14:
                continue
            pixels[sub[6]] = sub[0:6]
            pixels[sub[13]] = sub[7:13]
            pairs.setdefault(sub[6], []).append(sub[13])
            pairs.setdefault(sub[13], []).append(sub[6])
    clustered, clusters, k = set(), {}, 0
    for p in pairs:
        if p in clustered:
            continue
        k += 1
        todo, members = [p], set()
        while todo:
            nxt = []
            for q in todo:
                if q in clustered:
                    continue
                clustered.add(q)
                members.update(pairs[q])
                nxt.extend(pairs[q])
            todo = nxt
        clusters[f"cluster_{k}"] = members
    out = Path(str(p2p).replace("_pixel2pixel.txt", "") + "_clusters.bedpe")
    with open(out, "w") as o:
        o.write("chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tname\n")
        for cid, mem in clusters.items():
            v = np.array([[int(pixels[m][i]) for i in (1, 2, 4, 5)] for m in mem])
            o.write(f"{chrom}\t{v[:,0].min()}\t{v[:,1].max()}\t{chrom}\t"
                    f"{v[:,2].min()}\t{v[:,3].max()}\t{cid}\n")
    return out


def run_step2(params, tool, hic, chrom, length, outdir, resume=False):
    outdir = ensure_dir(outdir)
    manifest = outdir / "step2_manifest.json"
    if resume and manifest.exists():
        log.info("Step2: manifest exists, skipped")
        return json.loads(manifest.read_text())
    log.info("Step2: dumping %s bp matrices", params.dump_resolutions)
    files = dump_matrices(tool, hic, chrom, params.dump_resolutions, params.norm_order, outdir, resume)
    need = {params.low_res, params.mid_res, *params.score_res}
    missing = need - set(files)
    if missing:
        raise RuntimeError(f"Required resolutions missing in .hic: {sorted(missing)}")
    low, mid = params.low_res, params.mid_res
    low_bed = canny_detect(files[low], low, params, chrom, zero_cross=True, mode="low", oe_mask=False)
    mid_bed = canny_detect(files[mid], mid, params, chrom, zero_cross=(mid == low), mode="mid", oe_mask=True)
    remake = low_res_to_mid_res(low_bed, mid, chrom)
    fill = diagonal_fill(mid_bed, mid, length, chrom)
    merged = merge_bedpe(remake, fill, chrom)
    filtered = rri_gaussian_filter(merged, files[mid], mid, length, params, chrom)
    clusters = None
    if params.run_clusters:
        try:
            clusters = str(cluster_pixels(tool, filtered, mid, chrom))
        except Exception as e:  # auxiliary only
            log.warning("  pixel clustering skipped: %s", e)
    info = {"matrices": {str(k): str(v) for k, v in files.items()},
            "feature_bedpe": str(filtered), "clusters": clusters}
    manifest.write_text(json.dumps(info, indent=2))
    return info
