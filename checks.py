# Sanity checks and spot calculations. Run: python checks.py 1,3,5  (no args = all)
import re
import sys
import numpy as np
import pandas as pd

import csst_manufacturer_data as md

TOL = 0.01          # relative tolerance for a scaling-identity pass
EXACT_TOL = 1e-9
L_TOL = 1e-6        # tolerance when matching a scaled length to a tabulated one

S_NG, S_PROPANE = 0.60, 1.52
INWC_TO_PSI = md.INWC_TO_PSI
PROPANE_BTU_PER_SCF = md.PROPANE_BTU_PER_SCF

# Parflex sizes that pass the scaling identity (check 1).
PARFLEX_IDENTITY_PASS = {39, 62}

BRANDS = ["gastite", "parflex", "tracpipe", "csa", "parker"]
PROPANE_BRANDS = ["gastite", "parflex", "tracpipe"]


def _numeric_lengths(pipe):
    keep, lengths = [], []
    for i, v in enumerate(pipe["lengths_ft"]):
        try:
            lengths.append(float(v))
            keep.append(i)
        except (TypeError, ValueError):
            continue        # Parker's "30S" straight-tube column
    return np.array(lengths, dtype=float), keep


def _natural_gas(pipes):
    return {k: v for k, v in pipes.items() if v["gas"] == "natural_gas"}


# ------------------------------------------------- 1) scaling identity

# Q(dP, L) == Q(k*dP, k*L) holds exactly for any table generated from an
# equation in dP/L, and fails for measured data or the compressible GFE.
def _identity_rows(pipe):
    drops = np.array([c[1] for c in pipe["conditions"]], dtype=float)
    lengths, keep = _numeric_lengths(pipe)
    flows = np.array(pipe["flows"], dtype=float)[:, keep]

    rows = []
    for i in range(len(drops)):
        for j in range(len(drops)):
            if i == j or drops[i] <= 0:
                continue
            k = drops[j] / drops[i]
            if not np.isfinite(k) or k <= 0:
                continue
            for li, L in enumerate(lengths):
                target = L * k
                match = np.where(np.abs(lengths - target) <= L_TOL * max(target, 1.0))[0]
                if match.size == 0:
                    continue
                q1, q2 = flows[i, li], flows[j, match[0]]
                if not (np.isfinite(q1) and np.isfinite(q2)) or q1 == 0:
                    continue
                rel = abs(q2 - q1) / abs(q1)
                rows.append(dict(rel=rel, passed=rel <= TOL, exact=rel <= EXACT_TOL))
    return rows


def _identity_stats(rows):
    n = len(rows)
    n_pass = sum(r["passed"] for r in rows)
    n_exact = sum(r["exact"] for r in rows)
    rel = [r["rel"] for r in rows]
    return [n, n_pass, 100 * n_pass / n, n_exact, 100 * n_exact / n,
            100 * float(np.median(rel)), 100 * float(np.max(rel))]


def check_scaling_identity():
    per_pipe, per_brand = [], []
    for brand in ["gastite", "parflex", "tracpipe", "parker"]:
        pipes = md.load(brand)
        brand_rows = []
        for label, pipe in pipes.items():
            rows = _identity_rows(pipe)
            if not rows:
                print(f"NOTE: {brand} / {label} has no admissible pairs")
                continue
            brand_rows.extend(rows)
            per_pipe.append([brand, label, pipe["gas"], pipe["ehd"]] + _identity_stats(rows))
        if brand_rows:
            per_brand.append([brand] + _identity_stats(brand_rows))

    stat_cols = ["n pairs", "n pass", "pass %", "n exact", "exact %",
                 "median |diff| %", "max |diff| %"]
    print(f"\n--- 1) Scaling identity Q(dP, L) == Q(k*dP, k*L), tolerance {TOL*100:g}% ---")
    print("\nPer pipe:")
    print(pd.DataFrame(per_pipe, columns=["source", "pipe", "gas", "EHD"] + stat_cols)
          .round(2).to_string(index=False))
    print("\nPer manufacturer:")
    print(pd.DataFrame(per_brand, columns=["source"] + stat_cols).round(2).to_string(index=False))
    print("\nA pass rate near 100% means the table was generated from an equation in dP/L.")


# ------------------------------------------------- 2) SG multiplier

# Published propane capacity should equal published NG capacity x multiplier.
# sqrt(S_NG/S_propane) is the physics; x2520/1000 converts CFH to MBtu/h.
MULT_DISPLAY = np.sqrt(S_NG / S_PROPANE) * (PROPANE_BTU_PER_SCF / 1000)

# Same drop but a different nominal pressure class, so not the same schedule.
EXCLUDED_PAIRS = {("tracpipe", (0.50, round(1.0 * INWC_TO_PSI, 9)),
                   (0.43, round(1.0 * INWC_TO_PSI, 9)))}


def _table_header(brand, raw_id):
    text = open(md.os.path.join(md.HERE, md.SOURCES[brand]["file"])).read()
    m = re.search(r"\[TABLE " + re.escape(raw_id) + r"\](.*?)(?=\n\[TABLE |\Z)", text, re.S)
    if m is None:
        raise ValueError(f"Table {raw_id!r} not found; see pressure_conditions.txt.")
    block = m.group(1)
    gas = re.search(r"gas\s*=\s*(\S+)", block).group(1)
    p_inlet = float(re.search(r"P_inlet_psi\s*=\s*([\d.]+)", block).group(1))
    m_psi = re.search(r"dP_psi\s*=\s*([\d.]+)", block)
    if m_psi is not None:
        return gas, p_inlet, float(m_psi.group(1)), False
    drop_inwc = float(re.search(r"dP_inWC\s*=\s*([\d.]+)", block).group(1))
    return gas, p_inlet, drop_inwc * INWC_TO_PSI, True


# Low-pressure schedules match on drop alone, elevated ones on inlet and drop.
def _schedule_key(is_low, p_inlet, drop_psi):
    return ("low", round(drop_psi, 9)) if is_low else \
        ("high", round(p_inlet, 9), round(drop_psi, 9))


def _cond_key(cond):
    return _schedule_key(cond[3].endswith("inWC"), cond[0], cond[1])


# Re-pivot per-EHD data back into the published grid (rows = length, cols = size).
def _grid(pipes, gas, key, ehd_label):
    cols, lengths, matched = {}, None, None
    for pipe in pipes.values():
        if pipe["gas"] != gas:
            continue
        for i, c in enumerate(pipe["conditions"]):
            if _cond_key(c) == key:
                cols[pipe["ehd"]] = np.asarray(pipe["flows"][i])
                lengths, matched = pipe["lengths_ft"], c
                break
    if not cols:
        return None
    ehds = sorted(cols)
    df = pd.DataFrame(np.column_stack([cols[e] for e in ehds]),
                      index=pd.Index(lengths, name="L (ft)"),
                      columns=pd.Index([ehd_label[e] for e in ehds], name="pipe size"))
    return df, matched


def check_sg_multiplier(table="parflex_9"):
    brand, raw_id = table.split("_", 1)
    if brand not in PROPANE_BRANDS:
        raise ValueError(f"Unknown brand {brand!r}; expected one of {PROPANE_BRANDS}")

    gas, p_inlet, drop_psi, is_low = _table_header(brand, raw_id)
    key = _schedule_key(is_low, p_inlet, drop_psi)
    desc = (f"drop {drop_psi / INWC_TO_PSI:g} inWC" if is_low else f"drop {drop_psi:g} psi")
    print(f"\n--- 2) SG multiplier, {table!r} ({brand}, {gas}, inlet {p_inlet:g} psi, {desc}) ---")
    print(f"MULT = sqrt({S_NG}/{S_PROPANE}) * {PROPANE_BTU_PER_SCF:g}/1000 = {MULT_DISPLAY:.6f}")

    pipes = md.load(brand)
    ehd_label = md.SOURCES[brand]["ehd_label"]
    ng = _grid(pipes, "natural_gas", key, ehd_label)
    pr = _grid(pipes, "propane", key, ehd_label)
    if pr:                                  # undo the parser's MBtu/h -> SCFH conversion
        pr = (pr[0] * (PROPANE_BTU_PER_SCF / 1000), pr[1])

    excluded = bool(ng and pr and (brand, (round(ng[1][0], 9), round(ng[1][1], 9)),
                                   (round(pr[1][0], 9), round(pr[1][1], 9))) in EXCLUDED_PAIRS)

    predicted = None
    if ng:
        print("\nNG published (CFH):")
        print(ng[0])
        predicted = (ng[0] * MULT_DISPLAY).round(1)
        print("\nNG x multiplier -> predicted propane (MBtu/h):")
        print(predicted)
    else:
        print("\n(no natural-gas table at this schedule)")

    if excluded:
        print("\n(propane counterpart excluded: same drop, different pressure class)")
    elif pr:
        print("\nPropane published (MBtu/h):")
        print(pr[0])
        if predicted is not None:
            L = predicted.index.intersection(pr[0].index)
            C = predicted.columns.intersection(pr[0].columns)
            print("\n%diff = (predicted - published) / published * 100:")
            print(((predicted.loc[L, C] - pr[0].loc[L, C]) / pr[0].loc[L, C] * 100).round(2))
    else:
        print("\n(no propane table at this matching schedule)")


# ------------------------------------------------- 3-5) section 3.6.1

# Flat arrays of Q (SCFH-equiv), dP/ft (inWC/ft) and L (ft) for one pipe.
def _cells(pipe):
    lengths, keep = _numeric_lengths(pipe)
    flows = np.array(pipe["flows"], dtype=float)[:, keep]
    dp_per_ft = np.array([c[2] for c in pipe["conditions"]])[:, None] / lengths[None, :]
    L_grid = np.broadcast_to(lengths[None, :], flows.shape)
    ok = np.isfinite(flows) & (flows > 0) & np.isfinite(dp_per_ft) & (dp_per_ft > 0)
    return flows[ok], dp_per_ft[ok], L_grid[ok]


def _power_law(Q, dP):
    b, log_a = np.polyfit(np.log(Q), np.log(dP), 1)
    return float(np.exp(log_a)), float(b)


# Cross-manufacturer spread at matched EHD, evaluated at a common shared flow.
def check_spread():
    by_ehd = {}
    for brand in BRANDS:
        for pipe in _natural_gas(md.load(brand)).values():
            Q, dP, _ = _cells(pipe)
            if Q.size < 3:
                continue
            a, b = _power_law(Q, dP)
            by_ehd.setdefault(int(pipe["ehd"]), {})[brand] = dict(
                a=a, b=b, qmin=Q.min(), qmax=Q.max())

    rows = []
    for ehd in sorted(by_ehd):
        fits = by_ehd[ehd]
        if len(fits) < 2:
            continue
        lo = max(f["qmin"] for f in fits.values())
        hi = min(f["qmax"] for f in fits.values())
        if not hi > lo:
            rows.append([ehd, len(fits), np.nan, np.nan, "no shared Q range"])
            continue
        q = np.sqrt(lo * hi)
        vals = {k: f["a"] * q ** f["b"] for k, f in fits.items()}
        v = np.array(list(vals.values()))
        rows.append([ehd, len(fits), q, (v.max() - v.min()) / v.mean() * 100,
                     f"{max(vals, key=vals.get)} / {min(vals, key=vals.get)}"])

    df = pd.DataFrame(rows, columns=["EHD", "n sources", "Q eval (SCFH)", "spread %",
                                     "highest / lowest"])
    print("\n--- 3) Cross-manufacturer spread at matched EHD, natural gas ---")
    print(df.round(1).to_string(index=False))
    finite = df["spread %"].dropna()
    if finite.size:
        print(f"\nRange across {finite.size} multi-source EHDs: {finite.min():.1f}% to "
              f"{finite.max():.1f}% (median {finite.median():.1f}%)")
        print("Widest:")
        print(df.loc[df["spread %"].nlargest(3).index].round(1).to_string(index=False))


def _fit_generator(records):
    Q = np.concatenate([r[0] for r in records])
    G = np.concatenate([r[1] for r in records])
    D = np.concatenate([r[2] for r in records])
    ok = np.isfinite(Q) & np.isfinite(G) & np.isfinite(D) & (Q > 0) & (G > 0) & (D > 0)
    y = np.log(Q[ok])
    X = np.column_stack([np.ones(ok.sum()), np.log(D[ok]), np.log(G[ok])])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_res = np.sum((y - X @ coef) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return (float(np.exp(coef[0])), float(coef[1]), float(coef[2]),
            float(1 - ss_res / ss_tot if ss_tot > 0 else np.nan), int(ok.sum()))


# Generator-equation fits Q = C*D^p*(dP/L)^q; p and q are unit-invariant.
def check_generator_fits():
    rows = []
    for brand in BRANDS:
        groups = {"all": []}
        if brand == "parflex":
            groups.update({"identity passes": [], "identity fails": []})
        for pipe in _natural_gas(md.load(brand)).values():
            Q, dP, _ = _cells(pipe)
            if Q.size < 3:
                continue
            rec = (Q, dP, np.full(Q.shape, pipe["ehd"] * md.EHD_TO_MM))
            groups["all"].append(rec)
            if brand == "parflex":
                key = ("identity passes" if int(pipe["ehd"]) in PARFLEX_IDENTITY_PASS
                       else "identity fails")
                groups[key].append(rec)
        for name, recs in groups.items():
            if recs:
                C, p, q, r2, n = _fit_generator(recs)
                rows.append([brand, name, p, q, r2, n])

    print("\n--- 4) Generator-equation fits Q = C*D^p*(dP/L)^q ---")
    print(pd.DataFrame(rows, columns=["source", "subset", "p", "q", "R2", "n points"])
          .round(3).to_string(index=False))
    print("\np and q are invariant to unit choice; only C changes.")


GAS = {"NG": dict(S=0.60, mu=1.2e-5, b=37.26),
       "H2": dict(S=0.0696, mu=8.76e-6, b=12.1)}
RE_THRESHOLD = 4000.0


# Re_H2/Re_NG at equal appliance input; D and A cancel, so one constant covers all.
def re_scale_h2():
    ng, h2 = GAS["NG"], GAS["H2"]
    return (h2["S"] / ng["S"]) * (ng["b"] / h2["b"]) * (ng["mu"] / h2["mu"])


def check_h2_coverage():
    scale = re_scale_h2()
    print(f"\n--- 5) Hydrogen Re coverage over the published NG envelope ---")
    print(f"Re_H2/Re_NG at equal appliance input = {scale:.4f}")
    for brand in BRANDS:
        rows, tot, below = [], 0, 0
        for pipe in _natural_gas(md.load(brand)).values():
            Re = md.pipe_arrays(pipe)["Re"]
            ok = np.isfinite(Re)
            n_below = int((Re[ok] * scale < RE_THRESHOLD).sum())
            rows.append([int(pipe["ehd"]), int(ok.sum()), n_below, 100 * n_below / ok.sum()])
            tot += int(ok.sum())
            below += n_below
        print(f"\n{brand}:")
        print(pd.DataFrame(rows, columns=["EHD", "n cells", "below Re=4000", "% below"])
              .sort_values("EHD").round(1).to_string(index=False))
        print(f"Overall: {below} of {tot} cells below Re = {RE_THRESHOLD:g} "
              f"({100 * below / tot:.1f}%)")


# ------------------------------------------------- 6) exponent consistency

# Published generator exponents (p, q) from check 4, and the Polyflo n each implies.
PUBLISHED_FITS = {
    "Gastite":          (2.863, 0.495),
    "Parflex":          (2.677, 0.481),
    "TracPipe":         (2.892, 0.491),
    "CSA/ANSI LC 1:23": (2.959, 0.508),
    "Parker Hannifin":  (2.893, 0.484),
}


def check_exponent_consistency():
    rows = []
    for name, (p_obs, q) in PUBLISHED_FITS.items():
        n_from_q = 2 - 1 / q
        n_from_pq = 5 - p_obs / q
        p_pred = (5 - n_from_q) / (2 - n_from_q)
        rows.append([name, p_obs, q, n_from_q, n_from_pq, p_pred, p_obs - p_pred])
    df = pd.DataFrame(rows, columns=["source", "p obs", "q", "n from q", "n from p,q",
                                     "p predicted", "gap in p"])
    print("\n--- 6) Exponent consistency of the generator fits ---")
    print(df.round(3).to_string(index=False))
    print(f"\ngap range: {df['gap in p'].min():.2f} to {df['gap in p'].max():.2f}")


# ------------------------------------------------- 7) Polyflo constants

# a0 is one scalar for every cell, and K depends only on the gas, not the schedule.
def check_polyflo_constants():
    import master_table_new as mt
    print(f"\n--- 7) Polyflo closed-form constants ---")
    grid = np.full(mt.GRID, mt.a0, dtype=float)
    print(f"\na0 grid (constant = {mt.a0}):")
    print(mt.to_df(grid, mt.NPS_labels))
    print(f"All {grid.size} cells == a0: {np.all(grid == mt.a0)}")

    rows = []
    for gname, g in mt.GASES.items():
        K = (1.1494e-3 * (mt.Tb / mt.Pb) * mt.a0 ** -0.5
             * (1250 * g['S'] * mt.Pb / (27 * np.pi * mt.R_air * mt.Tb * g['mu'])) ** (mt.n / 2))
        rows.append([g['S'], g['mu'], K])
    print("\nK = 1.1494e-3*(Tb/Pb)*a0^-1/2*[1250*S*Pb/(27*pi*R_air*Tb*mu)]^(n/2):")
    print(pd.DataFrame(rows, index=list(mt.GASES), columns=["S", "mu (Pa.s)", "K"])
          .to_string(float_format=lambda v: f"{v:.6g}"))
    print("\nK has no P1/P2 term, so it is identical for low- and high-pressure schedules;\n"
          "what changes is the (P1^2-P2^2) term downstream.")


# ------------------------------------------------- 8) generator spot checks

# Hand-check of each brand's pooled generator equation at EHD 31 against its table.
SPOT_CASES = [
    ("Gastite",  19.99185, 2.86264, 0.49510, 100, 1.0, "Table 7.10"),
    ("Parflex",  10.28488, 2.67728, 0.48098, 100, 1.5, "Table 10"),
    ("TracPipe", 23.87437, 2.89187, 0.49135, 100, 1.0, "Table N-3"),
    ("Parker",   25.28203, 2.90095, 0.48514,  30, 1.0, "Table N-3"),
]


def check_generator_spot():
    D = 31 * md.EHD_TO_MM / 1000.0            # EHD 31 bore, m
    rows = []
    for name, C, p, q, L_ft, dP_psi, source in SPOT_CASES:
        Q_m3s = C * D ** p * (dP_psi * 6894.76 / (L_ft / 3.281)) ** q
        rows.append([name, source, L_ft, dP_psi, Q_m3s * 3600 * 35.314666721])
    print("\n--- 8) Generator equation spot checks, EHD 31 ---")
    print(pd.DataFrame(rows, columns=["source", "table", "L (ft)", "dP (psi)", "Q (CFH)"])
          .round(1).to_string(index=False))


# ------------------------------------------------- 9) shifted Polyflo factor k

# Laminar branch of the GFE in closed form: f = 64/Re gives Q = C^2*X*D^4*K_Re/64,
# relative density cancelling. Returns (Q_lam, Q_polyflo, Q_gfe_churchill, Re) in m3/day.
def _three_capacities(mt, cfg, gas, eps_mm, ncols=6):
    S, mu = gas['S'], gas['mu']
    P1, P2 = mt.get_pressures(cfg)
    D = mt.D_list_mm_steel[None, :ncols]                   # NPS 1/2 to 2
    X = (P1 ** 2 - P2 ** 2) / (S * mt.Tf * cfg['F'] * mt.L_km)
    C = 1.1494e-3 * (mt.Tb / mt.Pb)
    K_re = 4 * S * mt.Pb * 1000 / (mt.R_air * mt.Tb) / (np.pi * mu * 86400 / 1000)
    Q_lam = C ** 2 * X * D ** 4 * K_re / 64
    Q_poly = mt.polyflo_Q_day(S, mu, P1, P2, cfg['F'], D_mm=D)
    Q_gfe, Re, _ = mt.solve_gfe(S, mu, eps_mm, P1, P2, cfg['F'], D_mm=D,
                                GRID=(mt.L_list_m.size, ncols))
    return Q_lam, Q_poly, Q_gfe, Re


SHIFT_TABLES = (1, 2, 3, 4, 5)          # in-scope schedules, inlet <= 69 kPa gauge
LAM_TOL = 1e-4                          # Churchill sits a hair above 64/Re in laminar flow


# Smallest k with min(Q_lam, Q_poly*k^-1/(2-n)) <= Q_gfe at every in-scope cell,
# i.e. f = max(64/Re, k*a0*Re^-n) never below the GFE-Churchill reference.
def shifted_polyflo_k(eps_mm=None):
    import master_table_new as mt
    eps_mm = mt.EPS_MM if eps_mm is None else eps_mm
    a = 1 / (2 - mt.n)
    k, where = 1.0, None
    for t in SHIFT_TABLES:
        for g in ("NG", "H2"):
            Q_lam, Q_poly, Q_gfe, Re = _three_capacities(mt, mt.A_TABLES[t], mt.GASES[g], eps_mm)
            need = np.where(Q_lam <= Q_gfe * (1 + LAM_TOL), 0.0, (Q_poly / Q_gfe) ** (1 / a))
            i = np.unravel_index(np.argmax(need), need.shape)
            if need[i] > k:
                k, where = float(need[i]), (f"A.{t}", g, f"{mt.L_list_m[i[0]]:g} m",
                                            f"NPS {mt.NPS_labels[i[1]]}", float(Re[i]))
    return k, where


def check_shifted_polyflo():
    import master_table_new as mt
    rows = []
    for eps in (0.0015, 0.015, mt.EPS_MM, 0.09, 0.15):
        k, where = shifted_polyflo_k(eps)
        rows.append([eps, k, k ** (-1 / (2 - mt.n)), mt.Re_polyflo_laminar * k ** (-1 / (1 - mt.n)),
                     " / ".join(map(str, where[:4])) if where else "-",
                     round(where[4]) if where else np.nan])
    print("\n--- 9) Shifted Polyflo f = max(64/Re, k*a0*Re^-n), A.1-A.5, NPS 1/2-2, NG and H2 ---")
    print(pd.DataFrame(rows, columns=["eps (mm)", "k", "capacity factor k^-1/(2-n)", "Re*",
                                      "cell that sets k", "Re there"]).round(4).to_string(index=False))


# ------------------------------------------------- 10) one-at-a-time sensitivity

# H2 capacity at the Table A.1 schedule, NPS 1/2 to 2, GFE-Churchill, one input changed at a time.
def check_oat_sensitivity():
    import master_table_new as mt
    base_gas, cfg = dict(mt.GASES["H2"]), dict(mt.A_TABLES[1])

    def run(gas=base_gas, cfg=cfg, eps=mt.EPS_MM, table=None):
        c = mt.A_TABLES[table] if table else cfg
        return _three_capacities(mt, c, gas, eps)[2:]

    Q0, Re0 = run()
    cases = [("roughness 0.045 -> 0.0015 mm (copper-like)", dict(eps=0.0015)),
             ("roughness 0.045 -> 0.09 mm (commercial steel, upper value)", dict(eps=0.09)),
             ("fitting factor F 1.2 -> 1.0", dict(cfg={**cfg, "F": 1.0})),
             ("H2 viscosity -5%", dict(gas={**base_gas, "mu": base_gas["mu"] * 0.95})),
             ("H2 viscosity +5%", dict(gas={**base_gas, "mu": base_gas["mu"] * 1.05})),
             ("schedule A.1 -> A.2", dict(table=2)),
             ("schedule A.1 -> A.3", dict(table=3))]
    rows = [["baseline (A.1, eps 0.045 mm, F 1.2)", 0.0, 0.0, 0.0,
             100 * np.mean(Re0 < 1361), 100 * np.mean(Re0 < 2300)]]
    for name, kw in cases:
        Q, Re = run(**kw)
        d = (Q / Q0 - 1) * 100
        rows.append([name, float(np.median(d)), float(d.min()), float(d.max()),
                     100 * np.mean(Re < 1361), 100 * np.mean(Re < 2300)])
    lhv = (274.0 / 324.0 - 1) * 100
    rows.append(["heating value HHV -> LHV (kW rating only)", lhv, lhv, lhv,
                 rows[0][4], rows[0][5]])
    print("\n--- 10) One-at-a-time sensitivity, H2, Table A.1 schedule, NPS 1/2-2, GFE-Churchill ---")
    print(pd.DataFrame(rows, columns=["change", "median dQ %", "min dQ %", "max dQ %",
                                      "% cells Re<1361", "% cells Re<2300"])
          .round(1).to_string(index=False))


# ------------------------------------------------- 11) three-branch sizing law

# f = max(64/Re, k*a0*Re^-n, f_Churchill(size) above the transition peak). k only has to clear
# the transition peak; past each size's switch Re its own Churchill curve takes over.
def check_three_branch(eps_mm=0.09, k_use=1.18):
    import master_table_new as mt
    from friction import churchill_f, polyflo_f
    from scipy.optimize import brentq
    D = mt.D_list_mm_steel[:6]
    Re = np.logspace(np.log10(2300), 4, 20000)
    k_peak = max(float((churchill_f(Re, eps_mm, d) / polyflo_f(Re)).max()) for d in D)
    sw = [brentq(lambda R: churchill_f(R, eps_mm, d) - k_use * polyflo_f(R), 2e4, 1e9) for d in D]

    C = 1.1494e-3 * (mt.Tb / mt.Pb)
    worst, n_ch, n_tot, margin = -np.inf, 0, 0, []
    for t in SHIFT_TABLES:
        for g in ("NG", "H2"):
            gas, cfg = mt.GASES[g], mt.A_TABLES[t]
            P1, P2 = mt.get_pressures(cfg)
            X = (P1 ** 2 - P2 ** 2) / (gas['S'] * mt.Tf * cfg['F'] * mt.L_km)
            K_re = 4 * gas['S'] * mt.Pb * 1000 / (mt.R_air * mt.Tb) / (np.pi * gas['mu'] * 86400 / 1000)
            f = np.full((mt.L_list_m.size, 6), 0.03)
            for _ in range(500):                      # Picard, under-relaxed
                Q = C * np.sqrt(X / f) * D[None, :] ** 2.5
                R = K_re * Q / D[None, :]
                f_ch = churchill_f(R, eps_mm, D[None, :])
                f = 0.5 * f + 0.5 * np.maximum.reduce(
                    [64 / R, k_use * polyflo_f(R), np.where(R > 2e4, f_ch, 0.0)])
            Q_ref, _, _ = mt.solve_gfe(gas['S'], gas['mu'], eps_mm, P1, P2, cfg['F'],
                                       D_mm=D[None, :], GRID=(mt.L_list_m.size, 6))
            over = (Q / Q_ref - 1) * 100
            worst = max(worst, float(over.max()))
            margin += list(-over.ravel())
            n_ch += int(((R > 2e4) & (f_ch > k_use * polyflo_f(R))).sum())
            n_tot += over.size
    print(f"\n--- 11) Three-branch law, eps = {eps_mm} mm, A.1-A.5, NPS 1/2-2, NG and H2 ---")
    print(f"k needed to clear the transition peak: {k_peak:.4f}; used k = {k_use} "
          f"(capacity factor {k_use ** (-1 / (2 - mt.n)):.4f}, "
          f"Re* = {mt.Re_polyflo_laminar * k_use ** (-1 / (1 - mt.n)):.0f})")
    print("switch to Churchill above Re: " +
          ", ".join(f"NPS {nps} {r:,.0f}" for nps, r in zip(mt.NPS_labels[:6], sw)))
    print(f"max capacity over-prediction vs reference: {worst:.4f}%   "
          f"cells on the Churchill branch: {n_ch}/{n_tot} ({100 * n_ch / n_tot:.1f}%)   "
          f"margin median {np.median(margin):.1f}%, 90th pct {np.percentile(margin, 90):.1f}%")


# ------------------------------------------------- 12) hydrogen velocity and Re range

# v = Re*mu/(rho*D) for H2 at the Table A.1 and A.5 mean pressures; steel on the Schedule 40 ID,
# CSST on D_eff. Then the Re and velocity each schedule and the CSST data actually reach.
D_EFF_MM = {13: 11.4922, 18: 16.1712, 23: 21.3971, 31: 28.8794}
L_MIN_M = 9.0                            # shortest run considered, 30 ft
L_MAX_M = 76.2                           # longest run considered, 250 ft (last table length within it: 70 m)


def _p_mean(mt, cfg):
    P1, P2 = mt.get_pressures(cfg)
    return (2 / 3) * (P1 + P2 - P1 * P2 / (P1 + P2))


def _rho(mt, gas, p_kpa):
    return gas['S'] * p_kpa * 1000 / (mt.R_air * mt.Tf)


def check_h2_velocity_range():
    import master_table_new as mt
    h2 = mt.GASES['H2']
    p1, p5 = _p_mean(mt, mt.A_TABLES[1]), _p_mean(mt, mt.A_TABLES[5])
    pipes = [(f"NPS {n}", d) for n, d in zip(mt.NPS_labels[:6], mt.D_list_mm_steel[:6])]
    pipes += [(f"EHD {e}", d) for e, d in D_EFF_MM.items()]
    rows = []
    for name, d in pipes:
        v = lambda Re, p: Re * h2['mu'] / (_rho(mt, h2, p) * d / 1000)
        rows.append([name, d, v(1361, p1), v(2300, p1), v(2e4, p1), v(2e4, p5),
                     v(2e4, p1) / mt.speed_of_sound(h2)])
    print(f"\n--- 12) H2 velocity (m/s) at given Re; mean pressure A.1 {p1:.1f} kPa, A.5 {p5:.1f} kPa ---")
    print(pd.DataFrame(rows, columns=["pipe", "D (mm)", "Re 1361, A.1", "Re 2300, A.1",
                                      "Re 20000, A.1", "Re 20000, A.5", "Mach, Re 20000 A.1"])
          .round(3).to_string(index=False))

    L = mt.L_list_m
    rows = []
    for t in (1, 2, 3, 4, 5):
        r = mt.build_all(mt.A_TABLES[t], h2)
        Re, V = r['Re_gfe'][:, :6], r['V_gfe'][:, :6]
        m = (L >= L_MIN_M) & (L <= L_MAX_M)
        i = np.unravel_index(np.argmax(np.where(m[:, None], Re, 0)), Re.shape)
        rows.append([f"A.{t}", Re[m].max(), V[m].max(), f"NPS {mt.NPS_labels[i[1]]}, {L[i[0]]:g} m",
                     100 * np.mean(Re[m] > 2e4)])
    print("\nH2 over Tables A.1-A.5, NPS 1/2-2, runs of 30 ft (9 m) and longer, GFE-Churchill:")
    print(pd.DataFrame(rows, columns=["schedule", "max Re", "max V (m/s)", "at", "% cells Re > 20000"])
          .round(1).to_string(index=False))

    import report_figures as rf
    from csst_friction_models import parker_valid_mask
    rows = []
    for label, pipe in md.load('parker').items():
        e = int(pipe['ehd'])
        pipe['id_mm'] = rf.D_EFF_MM[e]
        arr = md.pipe_arrays(pipe)
        keep = parker_valid_mask(label, pipe, arr)
        Re, V = np.where(keep, arr['Re'], np.nan), np.where(keep, arr['V'], np.nan)
        rows.append([e, np.nanmax(Re), np.nanmax(Re) * 0.49, np.nanmax(Re) * 0.52,
                     np.nanmax(V) * 3.1, np.nanmax(V) * 3.3])
    print("\nCSST, H2 at the energy of the largest measured NG flow (Manufacturer C, D_eff):")
    print(pd.DataFrame(rows, columns=["EHD", "NG max Re", "H2 Re HHV", "H2 Re LHV",
                                      "H2 V HHV (m/s)", "H2 V LHV (m/s)"]).round(0).to_string(index=False))


# ------------------------------------------------- 13) sizing options over the hydrogen cells

# Deviation of a capacity from GFE-Churchill at fixed pressure drop: Re^2*f is fixed, so
# Q_option/Q_ref = Re_option/Re_ref for the same gas and bore. Positive = non-conservative.
def _dev_fixed_dp(Re_opt, f_opt, eps_mm, D_mm):
    from friction import churchill_f
    from scipy.optimize import brentq
    C = Re_opt ** 2 * f_opt
    Re_ref = brentq(lambda R: R ** 2 * churchill_f(R, eps_mm, D_mm) - C, 1e-3, 1e9)
    return (Re_opt / Re_ref - 1) * 100


def _option_a_f(Re):
    from friction import polyflo_f
    return np.maximum(64 / Re, polyflo_f(Re))


def check_sizing_options():
    import master_table_new as mt
    from friction import churchill_f, polyflo_f
    from scipy.optimize import brentq
    kf = 1.18 ** (-1 / (2 - mt.n))
    L = mt.L_list_m
    m = ((L >= L_MIN_M) & (L <= L_MAX_M))[:, None]
    D = mt.D_list_mm_steel[:6]

    # cell by cell, H2, A.1-A.5, NPS 1/2-2, runs of 30 ft and longer
    re_max = np.zeros(6)
    print("\n--- 13) Sizing options against GFE-Churchill, H2 cells of A.1-A.5, NPS 1/2-2, L >= 9 m ---")
    for eps in (0.0015, 0.015, mt.EPS_MM, 0.09):
        wA, wB, wA_hi = (-1e9, None), (-1e9, None), (-1e9, None)
        for t in SHIFT_TABLES:
            Q_lam, Q_poly, Q_gfe, Re = _three_capacities(mt, mt.A_TABLES[t], mt.GASES['H2'], eps)
            A = np.where(m, (np.minimum(Q_lam, Q_poly) / Q_gfe - 1) * 100, -1e9)
            B = np.where(m, (np.minimum(Q_lam, Q_poly * kf) / Q_gfe - 1) * 100, -1e9)
            Ah = np.where(Re > 1e4, A, -1e9)
            if eps == mt.EPS_MM:
                re_max = np.maximum(re_max, np.where(m, Re, 0).max(axis=0))
            for arr, w, key in ((A, wA, 'A'), (B, wB, 'B'), (Ah, wA_hi, 'Ah')):
                i = np.unravel_index(np.argmax(arr), arr.shape)
                if arr[i] > w[0]:
                    w = (arr[i], f"A.{t}, {L[i[0]]:g} m, NPS {mt.NPS_labels[i[1]]}, Re {Re[i]:,.0f}")
                    if key == 'A': wA = w
                    elif key == 'B': wB = w
                    else: wA_hi = w
        print(f"eps {eps} mm: Option A worst {wA[0]:+.2f}% ({wA[1]}); above Re 1e4 {wA_hi[0]:+.2f}% "
              f"({wA_hi[1]}); Option B worst {wB[0]:+.3f}% ({wB[1]})")

    # per size along the curve, up to the highest H2 Re of that size
    rows = []
    for j, (nps, d) in enumerate(zip(mt.NPS_labels[:6], D)):
        R = np.logspace(np.log10(1500), np.log10(re_max[j]), 400)
        row = [f"NPS {nps}", re_max[j]]
        for eps in (mt.EPS_MM, 0.09):
            row.append(max(_dev_fixed_dp(r, _option_a_f(r), eps, d) for r in R))
        for eps in (mt.EPS_MM, 0.09):
            g = lambda r: churchill_f(r, eps, d) - 1.18 * polyflo_f(r)
            row.append(brentq(g, 2e4, 1e9) if g(1e9) > 0 else np.inf)
        rows.append(row)
    print("\nPer size, over Re up to the highest H2 cell Re of that size (Option B limit: first Re above "
          "which Churchill exceeds the shifted line):")
    print(pd.DataFrame(rows, columns=["size", "highest H2 Re", "Option A max dev %, eps 0.045",
                                      "Option A max dev %, eps 0.09", "Option B limit Re, eps 0.045",
                                      "Option B limit Re, eps 0.09"]).round(2).to_string(index=False))

    # the hump is not a roughness effect: its peak at fixed dP, and the highest eps for which
    # Polyflo stays at or above Churchill from the end of the hump up to Re_top
    rows = []
    for eps in (1e-6, 0.0015, 0.015, mt.EPS_MM, 0.09):
        R = np.logspace(np.log10(2500), np.log10(6000), 300)
        rows.append([eps] + [max(_dev_fixed_dp(r, _option_a_f(r), eps, d) for r in R) for d in D])
    print("\nOption A hump peak (% capacity) by roughness, NPS 1/2 to 2:")
    print(pd.DataFrame(rows, columns=["eps (mm)"] + [f"NPS {n}" for n in mt.NPS_labels[:6]])
          .round(2).to_string(index=False))

    def conservative_after_hump(eps, d, re_top):
        R = np.logspace(np.log10(3500), np.log10(re_top), 3000)
        g = polyflo_f(R) - churchill_f(R, eps, d)
        i0 = np.argmax(g >= 0)
        return g[i0] >= 0 and np.all(g[i0:] >= -1e-12)

    for re_top in (2e4, 2e5):
        lim = []
        for d in D:
            lo, hi = 1e-4, 0.3
            for _ in range(50):
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if conservative_after_hump(mid, d, re_top) else (lo, mid)
            lim.append(lo)
        print(f"highest eps (mm) with Polyflo >= Churchill from the end of the hump to Re {re_top:g}: "
              + ", ".join(f"NPS {n} {e:.4f} (eps/D {e / d:.5f})" for n, e, d in zip(mt.NPS_labels[:6], lim, D)))


# ------------------------------------------------- 14) Manufacturer C table against its measurements

# Published Parflex capacity at the measured drop and length (log-log in length), over the measured
# flow. Positive = table above measurement, i.e. non-conservative.
def check_table_vs_measured():
    from csst_friction_models import parker_valid_mask
    pk, pf = md.load('parker'), md.load('parflex')
    rows = []
    for label, p in pk.items():
        t = pf[f'{label} (natural_gas)']
        Lt = np.array(t['lengths_ft'], float)
        keep = parker_valid_mask(label, p, md.pipe_arrays(dict(p)))
        dev = []
        for ci, c in enumerate(p['conditions']):
            tj = [j for j, tc in enumerate(t['conditions']) if abs(tc[2] - c[2]) / c[2] < 0.02][0]
            Qt = np.array(t['flows'][tj], float)
            for li, L in enumerate(p['lengths_ft']):
                if isinstance(L, str) or not keep[ci, li]:
                    continue
                q = np.exp(np.interp(np.log(L), np.log(Lt), np.log(Qt)))
                dev.append((q / p['flows'][ci][li] - 1) * 100)
        dev = np.array(dev)
        rows.append([int(p['ehd']), dev.size, np.median(dev), dev.min(), dev.max(), 100 * np.mean(dev > 0)])
    print("\n--- 14) Manufacturer C published table against its own measurements (NG) ---")
    print(pd.DataFrame(rows, columns=["EHD", "cells", "median %", "min %", "max %", "% cells table above"])
          .round(1).to_string(index=False))


# ------------------------------------------------- 15) laboratory test matrix

# One row per measurement point. Planning values only: gas properties at 15.6 C and 1 atm, CSST
# friction from the Manufacturer C trend on D_eff, steel from Churchill at 0.045 mm.
TEST_GASES = {'He': dict(S=0.1382, mu=1.94e-5), 'N2': dict(S=0.967, mu=1.74e-5),
              'NG': dict(S=0.60, mu=1.2e-5)}
TAP_SPACINGS_M = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
DP_MIN_PA, DP_MAX_PA = 100.0, 10e3
RE_T1_HE = (250, 500, 750, 1000, 1361, 1750, 2300, 3000, 4000, 5000)
RE_T1_N2 = (1000, 1361, 1750, 2300, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 25000)
RE_T3_HE = (500, 750, 1000, 1200, 1361, 1500, 1750, 2000, 2300, 2600, 3000, 4000, 5000)
RE_T3_N2 = (1000, 1361, 1750, 2300, 2600, 3000, 3300, 3600, 4000, 4500, 5000, 7500, 10000, 15000, 20000)
RE_T5 = (('He', 1000), ('He', 3000), ('N2', 10000), ('N2', 20000))
V_T6 = {'CSST': (20, 40, 60, 80, 100), 'steel': (20, 40, 60, 80, 100, 150)}
LC1_LENGTHS_FT = (30, 50, 80, 100, 150, 200, 250)
LC1_CONDITIONS = ((0.5, 0.5 * INWC_TO_PSI, '0.5 psig / 0.5 in. w.c.'), (2.0, 1.5, '2.0 psig / 1.5 psi'),
                  (5.0, 3.5, '5.0 psig / 3.5 psi'))
PSI_TO_KPA = 6.894757


def _csst_f_model():
    import report_figures as rf
    out = {}
    for e, (Re, f) in rf._parker_deff().items():
        o = np.argsort(Re)
        out[e] = (np.log(Re[o]), np.log(f[o]))
    def f_of(e, Re):
        lr, lf = out[e]
        x = np.log(Re)
        if x < lr[0]:
            return float(np.exp(lf[0] + lr[0] - x))      # laminar slope -1 below the data
        return float(np.exp(np.interp(x, lr, lf)))       # flat above the data
    return f_of


def _point(gas, D_mm, Re, f, L_dev_m, p_atm=101.325):
    import master_table_new as mt
    g = TEST_GASES[gas]
    rho = g['S'] * p_atm * 1000 / (mt.R_air * mt.Tf)
    D = D_mm / 1000
    v = Re * g['mu'] / (rho * D)
    q_std = v * np.pi / 4 * D ** 2 * 60000                 # std L/min, near atmospheric
    dpl = f / D * 0.5 * rho * v ** 2                       # Pa/m
    tap = next((s for s in TAP_SPACINGS_M if dpl * s >= DP_MIN_PA), TAP_SPACINGS_M[-1])
    tap = min(tap, max(TAP_SPACINGS_M[0], DP_MAX_PA / dpl)) if dpl * tap > DP_MAX_PA else tap
    p_in = dpl * (L_dev_m + tap + 10 * D) / 1000           # kPa gauge, outlet vented
    return dict(velocity_m_s=v, flow_std_L_min=q_std, flow_scfh=q_std * 60 / 28.3168,
                tap_spacing_m=tap, dp_taps_Pa=dpl * tap, inlet_kPa_g=p_in)


def build_test_matrix():
    import master_table_new as mt
    from friction import churchill_f
    f_csst = _csst_f_model()
    rows = []

    def add(**kw):
        rows.append(kw)

    for e, d in D_EFF_MM.items():                                        # T1
        for gas, res in (('He', RE_T1_HE), ('N2', RE_T1_N2)):
            for Re in res:
                L_dev = max(50 * d, 0.06 * Re * d if Re < 2300 else 0) / 1000
                pt = _point(gas, d, Re, f_csst(e, Re), L_dev)
                if pt['dp_taps_Pa'] < DP_MIN_PA:
                    continue                                   # left to the other gas
                add(test='T1', priority=1 if e in (13, 18) else 2, specimen='CSST, Manufacturer C',
                    size=f'EHD {e}', D_mm=d, gas=gas, target_Re=Re, dev_length_m=L_dev, **pt)

    for label, p in md.load('parker').items():                             # T2
        e = int(p['ehd'])
        for ci, c in enumerate(p['conditions']):
            for li, Lft in enumerate(p['lengths_ft']):
                if isinstance(Lft, str):
                    continue
                q = p['flows'][ci][li]
                arr = md.pipe_arrays(dict(p, id_mm=D_EFF_MM[e]))
                add(test='T2', priority=1 if e in (13, 18) else 2, specimen='CSST, Manufacturer C',
                    size=f'EHD {e}', D_mm=D_EFF_MM[e], gas='NG', run_length_ft=Lft,
                    inlet_kPa_g=c[0] * PSI_TO_KPA, drop_Pa=c[2] * INWC_TO_PSI * PSI_TO_KPA * 1000,
                    flow_scfh=q, flow_std_L_min=q * 28.3168 / 60, target_Re=arr['Re'][ci, li],
                    velocity_m_s=arr['V'][ci, li])

    for nps, d in zip(mt.NPS_labels[:3], mt.D_list_mm_steel[:3]):         # T3
        for gas, res in (('He', RE_T3_HE), ('N2', RE_T3_N2)):
            for Re in res:
                L_dev = max(50 * d, 0.06 * Re * d if Re < 2300 else 0) / 1000
                pt = _point(gas, d, Re, float(churchill_f(Re, mt.EPS_MM, d)), L_dev)
                if pt['dp_taps_Pa'] < DP_MIN_PA:
                    continue
                add(test='T3', priority=2, specimen='Schedule 40 steel', size=f'NPS {nps}', D_mm=d,
                    gas=gas, target_Re=Re, dev_length_m=L_dev, **pt)

    for e, d in D_EFF_MM.items():                                        # T4, He at the LC 1 conditions
        g = TEST_GASES['He']
        for p_in, dp_psi, lab in LC1_CONDITIONS:
            P1 = mt.Patm_kPa + p_in * PSI_TO_KPA
            P2 = P1 - dp_psi * PSI_TO_KPA
            rho_m = g['S'] * 0.5 * (P1 + P2) * 1000 / (mt.R_air * mt.Tf)
            for Lft in LC1_LENGTHS_FT:
                L = Lft * 0.3048
                v, f = 5.0, 0.2
                for _ in range(200):                      # fixed point on v with the C friction trend
                    v = np.sqrt((P1 - P2) * 1000 * 2 * d / 1000 / (f * L * rho_m))
                    Re = rho_m * v * d / 1000 / g['mu']
                    f = 0.5 * f + 0.5 * f_csst(e, Re)
                q_std = v * np.pi / 4 * (d / 1000) ** 2 * 0.5 * (P1 + P2) / mt.Pb * 60000
                add(test='T4', priority=2, specimen='CSST with four 90Â° bends and two end fittings',
                    size=f'EHD {e}', D_mm=d, gas='He', run_length_ft=Lft, inlet_kPa_g=p_in * PSI_TO_KPA,
                    drop_Pa=dp_psi * PSI_TO_KPA * 1000, target_Re=Re, velocity_m_s=v,
                    flow_std_L_min=q_std, flow_scfh=q_std * 60 / 28.3168, condition=lab)

    fittings = [(f'{k}, NPS {n}', d) for n, d in zip(mt.NPS_labels[:3], mt.D_list_mm_steel[:3])
                for k in ('90Â° elbow', 'Tee, run', 'Tee, branch', 'Ball valve')]
    fittings += [(f'End fittings, EHD {e}', d) for e, d in D_EFF_MM.items()]
    for name, d in fittings:                                              # T5
        for gas, Re in RE_T5:
            pt = _point(gas, d, Re, 0.0, 50 * d / 1000)
            add(test='T5', priority=3, specimen=name, size=name.split(', ')[-1], D_mm=d, gas=gas,
                target_Re=Re, velocity_m_s=pt['velocity_m_s'], flow_std_L_min=pt['flow_std_L_min'],
                flow_scfh=pt['flow_scfh'])

    runs = [(f'CSST run with fittings, EHD {e}', d, V_T6['CSST']) for e, d in D_EFF_MM.items()]
    runs += [(f'Steel run with fittings, NPS {n}', d, V_T6['steel'])
             for n, d in zip(mt.NPS_labels[:3], mt.D_list_mm_steel[:3])]
    g = TEST_GASES['He']                          # sound speed closest to hydrogen of the test gases
    rho = g['S'] * mt.Patm_kPa * 1000 / (mt.R_air * mt.Tf)
    for name, d, vs in runs:                                              # T6
        for v in vs:
            q_std = v * np.pi / 4 * (d / 1000) ** 2 * 60000
            add(test='T6', priority=3, specimen=name, size=name.split(', ')[-1], D_mm=d, gas='He',
                velocity_m_s=v, target_Re=rho * v * d / 1000 / g['mu'], flow_std_L_min=q_std,
                flow_scfh=q_std * 60 / 28.3168)
    df = pd.DataFrame(rows)
    df.insert(0, 'point', np.arange(1, len(df) + 1))
    return df


def check_test_matrix():
    import os
    df = build_test_matrix()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "test_matrix.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.round(4).to_csv(out, index=False)
    print(f"\n--- 15) Laboratory test matrix: {len(df)} points written to {out} ---")
    print(df.groupby(['test', 'gas']).agg(points=('point', 'size'), Re_min=('target_Re', 'min'),
                                         Re_max=('target_Re', 'max'), v_max=('velocity_m_s', 'max'))
          .round(1).to_string())


# ------------------------------------------------- 16) Re and velocity in every cell, NG and H2

# Every cell of Tables A.1-A.5, NPS 1/2-2, runs of 30 to 250 ft (9 to 70 m in the SI tables), for
# natural gas and hydrogen on GFE-Churchill at eps = 0.045 mm, mean pressure. One row per cell.
def check_re_velocity_cells():
    import os
    import master_table_new as mt
    L = mt.L_list_m
    keep = (L >= L_MIN_M) & (L <= L_MAX_M)
    rows = []
    for gas_name in ('NG', 'H2'):
        gas = mt.GASES[gas_name]
        for t in (1, 2, 3, 4, 5):
            r = mt.build_all(mt.A_TABLES[t], gas)
            for j in range(6):
                for i in np.where(keep)[0]:
                    Re, V = r['Re_gfe'][i, j], r['V_gfe'][i, j]
                    rows.append(dict(gas=gas_name, table=f"A.{t}", nps=mt.NPS_labels[j], run_m=L[i],
                                     run_ft=L[i] / 0.3048, Re=Re, velocity_m_s=V, velocity_ft_s=V / 0.3048,
                                     Mach=r['Mach_gfe'][i, j], P_mean_kPa_abs=r['P_avg'],
                                     regime=('laminar' if Re < 2300 else 'transitional' if Re < 4000
                                             else 'turbulent'),
                                     capacity_kW=r['Q_kW_gfe'][i, j]))
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "re_velocity_by_cell_30_250ft.csv")
    df.round(4).to_csv(out, index=False)
    print(f"\n--- 16) Re and velocity by cell, 30 to 250 ft: {len(df)} cells written to {out} ---")
    g = df.groupby(['gas', 'table'])
    print(g.agg(Re_min=('Re', 'min'), Re_max=('Re', 'max'), v_min=('velocity_m_s', 'min'),
                v_max=('velocity_m_s', 'max'), Mach_max=('Mach', 'max'),
                pct_Re_gt_2e4=('Re', lambda s: 100 * (s > 2e4).mean())).round(2).to_string())
    h = df[(df.gas == 'H2') & (df.Re > 2e4)]
    print("\nH2 cells above Re = 20,000, by schedule and size (shortest to longest run, m):")
    for (t, n), s in h.groupby(['table', 'nps']):
        print(f"  {t}, NPS {n}: {s.run_m.min():g} to {s.run_m.max():g} m ({len(s)} cells), Re {s.Re.min():,.0f} to {s.Re.max():,.0f}")
    print("\nCells by regime (%), by gas and schedule:")
    print(pd.crosstab([df.gas, df.table], df.regime, normalize='index').mul(100).round(1).to_string())


CHECKS = {
    1: check_scaling_identity,
    2: check_sg_multiplier,
    3: check_spread,
    4: check_generator_fits,
    5: check_h2_coverage,
    6: check_exponent_consistency,
    7: check_polyflo_constants,
    8: check_generator_spot,
    9: check_shifted_polyflo,
    10: check_oat_sensitivity,
    11: check_three_branch,
    12: check_h2_velocity_range,
    13: check_sizing_options,
    14: check_table_vs_measured,
    15: check_test_matrix,
    16: check_re_velocity_cells,
}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 60)
    arg = next((a for a in sys.argv[1:] if re.fullmatch(r"\d+(,\d+)*", a)), None)
    for k in sorted({int(x) for x in arg.split(",")} if arg else CHECKS):
        CHECKS[k]()

