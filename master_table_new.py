# Tables A.1-A.7 (steel) and B.1-B.5 (propane), plus copper A.8b-A.14b: Q, V, Re
# from the General Flow Equation closed two ways - Polyflo (f = a0*Re^-n, closed
# form) and Churchill (1977), continuous across laminar/transition/turbulent.
# Run: python master_table_new.py           selected table below
#      python master_table_new.py census    Re/velocity census, all configs
#      python master_table_new.py batch     pickle all 26 configs, no output
import os
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from friction import churchill_f

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 12)

a0 = 0.140370          # Polyflo friction law f = a0*Re^-n, fit to CSA Table A.5
n = 0.151571
alpha = 2 / (2 - n)

Tb = 288.6             # K, base temperature
Pb = 101.325           # kPa, base pressure
R_air = 287.05         # J/(kg.K), air
z = 0.2778             # MJ/h -> kW
Tf = Tb                # isothermal
Patm_kPa = 101.325

EPS_MM = 0.045         # commercial steel roughness

GASES = {
    'NG':      dict(S=0.60,   mu=1.2e-5,       b=37.258945808, GAMMA=1.31),
    'H2':      dict(S=0.0696, mu=8.76e-6,      b=12.1,         GAMMA=1.41),
    'butane':  dict(S=2.00,   mu=0.0071524e-3, b=121.46416333, GAMMA=1.09),
    'propane': dict(S=1.52,   mu=8.0e-6,       b=93.892543436, GAMMA=1.13),
}

NPS_labels = ['1/2', '3/4', '1', '1-1/4', '1-1/2', '2', '2-1/2', '3', '4']
D_list_in_steel = np.array([0.622, 0.824, 1.049, 1.380, 1.610, 2.067, 2.469, 3.068, 4.026])
D_list_mm_steel = D_list_in_steel * 25.4
L_list_m = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 35, 40, 45, 50, 60, 70, 80, 90,
                     100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600])

D_mm = D_list_mm_steel[None, :]
L_km = (L_list_m / 1000.0)[:, None]
GRID = (L_list_m.size, D_list_in_steel.size)

A_TABLES = {
    1: dict(regime='low',  deltaP_Pa=125, F=1.2),
    2: dict(regime='low',  deltaP_Pa=250, F=1.2),
    3: dict(regime='high', P_gauge=14,  drop_kPa=7,  F=1.2),
    4: dict(regime='high', P_gauge=14,  drop_kPa=10, F=1.2),
    5: dict(regime='high', P_gauge=34,  drop_kPa=17, F=1.0),
    6: dict(regime='high', P_gauge=70,  drop_kPa=35, F=1.0),
    7: dict(regime='high', P_gauge=140, drop_kPa=70, F=1.0),
}

# Annex B - propane, its own pressure schedule.
B_TABLES = {
    1: dict(regime='low',  deltaP_Pa=250,            F=1.2),
    2: dict(regime='high', P_gauge=14,  drop_kPa=7,  F=1.2),
    3: dict(regime='high', P_gauge=34,  drop_kPa=17, F=1.0),
    4: dict(regime='high', P_gauge=70,  drop_kPa=35, F=1.0),
    5: dict(regime='high', P_gauge=140, drop_kPa=70, F=1.0),
}

# Copper (Type K), CSA Tables A.8b-A.14b, same 1-7 schedule numbering as A_TABLES.
NPS_labels_cu = ['1/4', '3/8', '1/2', '5/8', '3/4', '1', '1-1/4']
OD_labels_cu = [9.5, 13, 16, 19, 22, 29, 35]
D_list_mm_cu = np.array([7.747, 10.2108, 13.3858, 16.5608, 18.923, 25.273, 31.623])
EPS_MM_CU = 0.0015
A_TABLES_CU = dict(A_TABLES)

MATERIALS = {
    'steel': dict(NPS_labels=NPS_labels, OD_labels=None, D_list_mm=D_list_mm_steel,
                  EPS_MM=EPS_MM, A_TABLES=A_TABLES,
                  GRID=(L_list_m.size, len(NPS_labels)),
                  table_name=lambda t: f'A.{t}'),
    'copper': dict(NPS_labels=NPS_labels_cu, OD_labels=OD_labels_cu, D_list_mm=D_list_mm_cu,
                   EPS_MM=EPS_MM_CU, A_TABLES=A_TABLES_CU,
                   GRID=(L_list_m.size, len(NPS_labels_cu)),
                   table_name=lambda t: f'A.{t + 7}b'),
}

# Polyflo-laminar crossover: Re where f = a0*Re^-n meets f = 64/Re.
Re_polyflo_laminar = (64 / a0) ** (1 / (1 - n))

# ------------------------------------------------------------------ selectors
run_table = 'A'
A_table_number = 5      # 1-7 = steel A.1-A.7, 8-14 = copper A.8b-A.14b
A_table_gas = 'H2'      # NG, H2, butane
B_table_number = 3


def get_pressures(cfg):
    if cfg['regime'] == 'low':
        return Patm_kPa + cfg['deltaP_Pa'] / 1000, Patm_kPa
    P1 = cfg['P_gauge'] + Patm_kPa
    return P1, P1 - cfg['drop_kPa']


# Dynaflow closed form: the GFE solved under f = a0*Re^-n, no iteration.
def polyflo_Q_day(S, mu, P1, P2, F, D_mm=D_mm):
    L_eff = F * L_km
    leading = (1.1494e-3 * (Tb / Pb) * a0 ** -0.5
               * (1250 * S * Pb / (np.pi * R_air * Tb * mu * 27)) ** (n / 2)) ** alpha
    return (leading * ((P1 ** 2 - P2 ** 2) / (S * Tf * L_eff)) ** (alpha / 2)
            * D_mm ** (alpha * (2.5 - n / 2)))


friction_closure = churchill_f


# GFE with Churchill friction, iterated to a fixed point.
def solve_gfe(S, mu, eps_mm, P1, P2, F, f0=0.02, tol=1e-12, maxiter=300,
              D_mm=D_mm, GRID=GRID):
    L_eff = F * L_km
    rho_std = S * (Pb * 1000) / (R_air * Tb)

    def q_and_re(f):
        Q_day = (1.1494e-3 * (Tb / Pb)
                 * np.sqrt((P1 ** 2 - P2 ** 2) / (S * Tf * L_eff * f)) * D_mm ** 2.5)
        return Q_day, 4 * rho_std * (Q_day / 86400) / (np.pi * (D_mm / 1000) * mu)

    f = np.full(GRID, f0, dtype=float)
    for _ in range(maxiter):
        f_new = friction_closure(q_and_re(f)[1], eps_mm, D_mm)
        step = np.max(np.abs(f_new / f - 1))
        f = f_new
        if step < tol:
            break
    else:
        print(f"WARNING: solve_gfe did not converge (final step={step:.2e})")

    Q_day, Re = q_and_re(f)
    return Q_day, Re, f


# V at flowing conditions, using the Menon mean pipeline pressure.
def flowing_velocity(Q_m3h_std, P1, P2, D_mm=D_mm):
    P_avg = (2 / 3) * (P1 + P2 - (P1 * P2) / (P1 + P2))
    Q_actual = Q_m3h_std * (Pb / P_avg) * (Tf / Tb)
    A = (np.pi / 4) * (D_mm / 1000) ** 2
    return Q_actual / (3600 * A), P_avg


def rho_std_of(S):
    return S * (Pb * 1000) / (R_air * Tb)


def speed_of_sound(gas):
    return np.sqrt(gas['GAMMA'] * (R_air / gas['S']) * Tf)


# Everything for one (table config, gas) pair, on the selected material.
def build_all(cfg, gas, material='steel'):
    m = MATERIALS[material]
    D_mm_m, GRID_m, EPS_m = m['D_list_mm'][None, :], m['GRID'], m['EPS_MM']
    S, mu, b, F = gas['S'], gas['mu'], gas['b'], cfg['F']
    P1, P2 = get_pressures(cfg)
    rho_std = rho_std_of(S)

    Q_day_poly = polyflo_Q_day(S, mu, P1, P2, F, D_mm=D_mm_m)
    Q_m3h_poly = Q_day_poly / 24
    Re_poly = 4 * rho_std * (Q_day_poly / 86400) / (np.pi * (D_mm_m / 1000) * mu)

    Q_day_gfe, Re_gfe, f_gfe = solve_gfe(S, mu, EPS_m, P1, P2, F, D_mm=D_mm_m, GRID=GRID_m)
    Q_m3h_gfe = Q_day_gfe / 24

    V_poly, P_avg = flowing_velocity(Q_m3h_poly, P1, P2, D_mm=D_mm_m)
    V_gfe, _ = flowing_velocity(Q_m3h_gfe, P1, P2, D_mm=D_mm_m)
    a = speed_of_sound(gas)

    return dict(Q_kW_poly=np.round(Q_m3h_poly * b * z), Q_kW_gfe=np.round(Q_m3h_gfe * b * z),
                Q_m3h_poly=Q_m3h_poly, Q_m3h_gfe=Q_m3h_gfe,
                V_poly=V_poly, V_gfe=V_gfe, Re_poly=Re_poly, Re_gfe=Re_gfe,
                a=a, Mach_poly=V_poly / a, Mach_gfe=V_gfe / a, f_gfe=f_gfe,
                diff_pct=(Q_m3h_gfe - Q_m3h_poly) / Q_m3h_poly * 100,
                P_avg=P_avg, P1=P1, P2=P2)


# Smallest size whose interpolated capacity at L_m meets load_kW.
def size_pipe(load_kW, L_m, gas_name, annex, table_number, method, material='steel'):
    if material == 'steel':
        cfg = (A_TABLES if annex == 'A' else B_TABLES)[table_number]
        col_labels = NPS_labels
    else:
        cfg = MATERIALS[material]['A_TABLES'][table_number]
        col_labels = MATERIALS[material]['OD_labels']
    res = build_all(cfg, GASES[gas_name], material=material)
    Q_kW = res['Q_kW_poly'] if method == 'polyflo' else res['Q_kW_gfe']

    for j, label in enumerate(col_labels):
        cap = np.interp(L_m, L_list_m, Q_kW[:, j])
        if cap >= load_kW:
            return label, cap, cap - load_kW
    return None


def to_df(mat, col_labels=None, col_name='NPS'):
    return pd.DataFrame(mat, index=pd.Index(L_list_m, name='L (m)'),
                        columns=pd.Index(col_labels if col_labels is not None else NPS_labels,
                                         name=col_name))


def show(mat, title, col_labels=None, col_name='NPS'):
    print(f"\n--- {title} ---")
    print(to_df(mat, col_labels, col_name))


# Negative = Polyflo over-predicts capacity, unsafe.
def flag_positive(diff_pct):
    return np.vectorize(lambda v: f"*{v:.2f}*" if v < 0 else f"{v:.2f}")(diff_pct)


# All (label, cfg, gas) triples across both annexes sharing cfg's pressure condition.
def matching_entries(cfg):
    entries = [(f'A.{t} ({g})', c, g) for t, c in A_TABLES.items() if c == cfg
               for g in ['NG', 'H2', 'butane']]
    entries += [(f'B.{t} (propane)', c, 'propane') for t, c in B_TABLES.items() if c == cfg]
    return entries


# ------------------------------------------------------------------ census

def all_configs(material='steel'):
    m = MATERIALS[material]
    out = {}
    if material == 'steel':
        for t, cfg in A_TABLES.items():
            for g in ['NG', 'H2', 'butane']:
                out[f'A.{t} ({g})'] = build_all(cfg, GASES[g])
        for t, cfg in B_TABLES.items():
            out[f'B.{t} (propane)'] = build_all(cfg, GASES['propane'])
    else:
        for t, cfg in m['A_TABLES'].items():
            for g in ['NG', 'H2']:
                out[f"{m['table_name'](t)} ({g})"] = build_all(cfg, GASES[g], material=material)
    return out


# Re spread, laminar/transitional coverage and Polyflo-vs-GFE error per config.
def census(configs):
    rows = []
    for res in configs.values():
        Re, diff = res['Re_gfe'], res['diff_pct']
        rows.append([Re.min(), Re.max(),
                     100 * (Re < 4000).mean(), 100 * (Re < 2300).mean(),
                     100 * (Re < Re_polyflo_laminar).mean(),
                     np.sqrt(np.mean(diff ** 2)), np.max(np.abs(diff)),
                     100 * (diff < 0).mean()])
    return pd.DataFrame(rows, index=list(configs),
                        columns=['Re min', 'Re max', '%cells Re<4000', '%cells Re<2300',
                                 f'%cells Re<{Re_polyflo_laminar:.0f}', 'RMSE %',
                                 'max abs error %', '% Polyflo over-predicts'])


# Erosional velocity and compressibility screen per config.
def velocity_census(configs):
    rows = []
    for res in configs.values():
        V, M = res['V_gfe'], res['Mach_gfe']
        rows.append([V.max(), 100 * (V > 20).mean(), 100 * (V > 30).mean(), res['a'],
                     M.max(), 100 * (M > 0.3).mean(), 100 * (M > 0.5).mean(),
                     100 * (M > 1.0).mean()])
    return pd.DataFrame(rows, index=list(configs),
                        columns=['V_max (m/s)', '%cells V>20', '%cells V>30', 'a (m/s)',
                                 'Mach max', '%cells Mach>0.3', '%cells Mach>0.5',
                                 '%cells Mach>1.0'])


def print_census():
    steel = all_configs('steel')
    print("\n--- Steel census (A.1-A.7 NG/H2/butane, B.1-B.5 propane) ---")
    print(census(steel))
    print("\n--- Velocity screen (erosional V>20/V>30 vs compressibility Mach) ---")
    print(velocity_census(steel))

    # Whether the Mach>0.3 exceedance is confined to the largest diameters.
    for t in range(3, 8):
        breakdown = {}
        for g in ['NG', 'H2', 'butane']:
            count = (steel[f'A.{t} ({g})']['Mach_gfe'] > 0.3).sum(axis=0)
            breakdown[g] = [f'{c:d} ({100 * c / len(L_list_m):.0f}%)' for c in count]
        print(f"\n--- A.{t}: cells with Mach>0.3 by NPS (of {len(L_list_m)} lengths) ---")
        print(pd.DataFrame(breakdown, index=pd.Index(NPS_labels, name='NPS')))

    print("\n--- Copper census (A.8b-A.14b, NG/H2) ---")
    print(census(all_configs('copper')))


# ------------------------------------------------------------------ report

def _moody_figure(res_by_gas=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    Re_sweep = np.logspace(np.log10(500), np.log10(2e7), 400)
    for j in range(len(NPS_labels)):
        ax.plot(Re_sweep, friction_closure(Re_sweep, EPS_MM, D_list_mm_steel[j]),
                color='tab:blue', linewidth=1.3, alpha=0.75, zorder=3)
    ax.plot([], [], color='tab:blue', linewidth=2, label='GFE: Churchill (1977), all NPS')

    ax.plot(Re_sweep, a0 * Re_sweep ** -n, color='black', linewidth=1.8,
            linestyle='--', label='Polyflo f = a0*Re^-n')
    ax.plot(Re_polyflo_laminar, 64 / Re_polyflo_laminar, marker='o', color='0.35',
            markersize=5, zorder=5)
    ax.annotate(f'Polyflo-laminar crossover \n Re = {Re_polyflo_laminar:.0f}',
                xy=(Re_polyflo_laminar, 64 / Re_polyflo_laminar), xytext=(6000, 0.095),
                arrowprops=dict(arrowstyle='->', color='0.35', linewidth=1.3),
                fontsize=9, color='0.35', ha='left')
    for x in (2300, 4000):
        ax.axvline(x, color='0.7', linestyle=':', linewidth=1, zorder=1)

    gas_colors = {'NG': 'tab:green', 'H2': 'tab:cyan', 'butane': 'tab:red', 'propane': 'tab:purple'}
    gas_y = {'NG': 0.0098, 'H2': 0.0113, 'butane': 0.0128, 'propane': 0.0143}
    gas_tables = {'NG': A_TABLES, 'H2': A_TABLES, 'butane': A_TABLES, 'propane': B_TABLES}
    for gname, tables in gas_tables.items():
        g = GASES[gname]
        Re_min, Re_max = np.inf, -np.inf
        for cfg in tables.values():
            P1, P2 = get_pressures(cfg)
            _, Re_c, _ = solve_gfe(g['S'], g['mu'], EPS_MM, P1, P2, cfg['F'])
            Re_min, Re_max = min(Re_min, Re_c.min()), max(Re_max, Re_c.max())
        ax.annotate('', xy=(Re_max, gas_y[gname]), xytext=(Re_min, gas_y[gname]),
                    arrowprops=dict(arrowstyle='<->', color=gas_colors[gname], linewidth=1.6))
        ax.plot([], [], color=gas_colors[gname], linewidth=1.6, label=f'{gname} Re range')

    ax.annotate('increasing D', xy=(1.5e5, 0.0175), xytext=(1.5e5, 0.045),
                arrowprops=dict(arrowstyle='->', color='0.35', linewidth=1.3),
                fontsize=9, color='0.35', ha='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(20, 2e7)
    ax.set_ylim(0.009, 0.145)
    ax.set_xlabel('Reynolds number Re')
    ax.set_ylabel('Darcy friction factor f')
    ax.set_title(f'Moody diagram - A.1-A.7 & B.1-B.5, NG/H2/butane/propane, eps={EPS_MM} mm')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()


def _re_crossings(L, Re, thresholds=(2300, 4000)):
    out = {}
    for thresh in thresholds:
        crossings = [L[k] + (thresh - Re[k]) * (L[k + 1] - L[k]) / (Re[k + 1] - Re[k])
                     for k in np.where(np.diff(np.sign(Re - thresh)))[0]]
        if crossings:
            out[thresh] = crossings
    return out


def report(cfg, gas_name, table_label, col_labels, col_name, material):
    gas = GASES[gas_name]
    res = build_all(cfg, gas, material=material)

    print("=" * 100)
    print(f"{table_label}  [regime={cfg['regime']}, F={cfg['F']}]")
    print("=" * 100)
    show(res['Q_kW_poly'], 'Q (kW) - Polyflo', col_labels, col_name)
    show(res['Q_kW_gfe'], 'Q (kW) - GFE (Churchill 1977)', col_labels, col_name)
    show(np.round(res['Q_m3h_poly'], 2), 'Q (m3/h) - Polyflo', col_labels, col_name)
    show(np.round(res['Q_m3h_gfe'], 2), 'Q (m3/h) - GFE', col_labels, col_name)
    print("\n--- %diff = (Q_gfe - Q_polyflo)/Q_polyflo*100 "
          "[*flagged* = Polyflo over-predicts, unsafe] ---")
    print(to_df(flag_positive(res['diff_pct']), col_labels, col_name))
    show(np.round(res['V_poly'], 3), 'V (m/s) - Polyflo', col_labels, col_name)
    show(np.round(res['V_gfe'], 3), 'V (m/s) - GFE', col_labels, col_name)
    show(np.round(res['Re_poly']).astype(int), 'Re - Polyflo', col_labels, col_name)
    show(np.round(res['Re_gfe']).astype(int), 'Re - GFE', col_labels, col_name)

    regime = np.where(res['Re_gfe'] < 2300, 'L', np.where(res['Re_gfe'] < 4000, 'T', 'U'))
    show(regime, 'Regime map - GFE (L<2300, T 2300-4000, U>=4000)', col_labels, col_name)
    for code, name in (('L', 'LAM'), ('T', 'TRANS'), ('U', 'TURB')):
        k = np.sum(regime == code)
        print(f"{name:6s} {k:3d} / {regime.size}  ({100 * k / regime.size:.1f}%)")

    # Copper shares steel's schedule VALUES, so dict equality would cross-match.
    matched = matching_entries(cfg) if material == 'steel' else [(table_label, cfg, gas_name)]
    note = (' (matched across annexes)' if material == 'steel'
            and len({lbl[0] for lbl, _, _ in matched}) > 1 else '')

    summary, cen = [], []
    for label, entry_cfg, gname in matched:
        r = res if label == table_label else build_all(entry_cfg, GASES[gname], material=material)
        Re_f, diff = r['Re_gfe'], r['diff_pct']
        summary.append([MATERIALS[material]['EPS_MM'], np.sqrt(np.mean((diff / 100) ** 2)) * 100,
                        np.max(np.abs(diff)), (Re_f < 2300).mean() * 100,
                        ((Re_f >= 2300) & (Re_f < 4000)).mean() * 100,
                        (Re_f >= 4000).mean() * 100])
        cen.append([100 * (Re_f < 4000).mean(), 100 * (Re_f < 2300).mean(),
                    100 * (Re_f < Re_polyflo_laminar).mean()])
    labels = [lbl for lbl, _, _ in matched]

    print(f"\n--- Capacity summary{note} ---")
    print(pd.DataFrame(summary, index=labels,
                       columns=['eps (mm)', 'RMSE (%)', 'max|err| (%)', '%LAM', '%TRN', '%OK']))
    print(f"\n--- Re census{note} ---")
    print(f"Polyflo-laminar crossover: Re = {Re_polyflo_laminar:.1f}")
    print(pd.DataFrame(cen, index=labels,
                       columns=['%cells Re<4000', '%cells Re<2300',
                                f'%cells Re<{Re_polyflo_laminar:.0f}']))

    if material == 'steel':
        _moody_figure()

    col_colors = plt.cm.tab10(np.arange(len(col_labels)))
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for j in range(len(col_labels)):
        ax2.scatter(res['Re_gfe'][:, j], res['diff_pct'][:, j], s=12, alpha=0.6,
                    color=col_colors[j], label=col_labels[j])
    ax2.set_xscale('log')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.axhspan(-3, 3, color='grey', alpha=0.15)
    ax2.set_xlabel('Reynolds number Re')
    ax2.set_ylabel('% error = (Q_gfe - Q_polyflo)/Q_polyflo*100')
    ax2.set_title(f'{table_label}: GFE vs Polyflo percent error')
    ax2.legend(title=col_name, fontsize=8, loc='best')
    fig2.tight_layout()

    # Dotted = Churchill/GFE, dashed = Polyflo, matching the CSST figures.
    src_colors = {'poly': '#1baf7a', 'gfe': '#eb6834'}
    n_cols = 3
    fig3, axes3 = plt.subplots(-(-len(col_labels) // n_cols), n_cols,
                               figsize=(13, 3.3 * -(-len(col_labels) // n_cols)))
    axes3 = np.atleast_1d(axes3).ravel()
    for j, label in enumerate(col_labels):
        ax3 = axes3[j]
        ax3.plot(L_list_m, res['Q_m3h_poly'][:, j], color=src_colors['poly'],
                 linewidth=1.5, linestyle='--')
        ax3.plot(L_list_m, res['Q_m3h_gfe'][:, j], color=src_colors['gfe'],
                 linewidth=1.5, linestyle=':')
        for src, re_arr in (('poly', res['Re_poly'][:, j]), ('gfe', res['Re_gfe'][:, j])):
            for thresh, ls in ((2300, ':'), (4000, '-.')):
                for xc in _re_crossings(L_list_m, re_arr).get(thresh, []):
                    ax3.axvline(xc, color=src_colors[src], linewidth=0.9, linestyle=ls, alpha=0.8)
        ax3.set_title(f'{col_name} {label}', fontsize=9)
        ax3.set_xlabel('L (m)', fontsize=8)
        ax3.set_ylabel('Q (m³/h)', fontsize=8)
        ax3.tick_params(labelsize=7)
        ax3.set_xlim(10, 200)
    for ax3 in axes3[len(col_labels):]:
        ax3.set_visible(False)

    handles = [plt.Line2D([0], [0], color=src_colors['poly'], linestyle='--', linewidth=1.5,
                          label='Q Polyflo'),
               plt.Line2D([0], [0], color=src_colors['gfe'], linestyle=':', linewidth=1.5,
                          label='Q GFE (Churchill)')]
    handles += [plt.Line2D([0], [0], color='grey', linewidth=0.9, linestyle=ls,
                           label=f'Re = {t} (color = source)') for t, ls in ((2300, ':'), (4000, '-.'))]
    fig3.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, framealpha=0.9)
    fig3.suptitle(f'{table_label}: Q Polyflo vs Q GFE/Churchill vs pipe length', fontsize=11)
    fig3.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'table'

    if mode == 'census':
        print_census()

    elif mode == 'batch':
        os.makedirs('output', exist_ok=True)
        with open('output/all_configs.pkl', 'wb') as fh:
            pickle.dump(all_configs('steel'), fh)

    else:
        material, col_name = 'steel', 'NPS'
        if run_table == 'A':
            local_num = A_table_number if A_table_number <= 7 else A_table_number - 7
            if A_table_number > 7:
                material, col_name = 'copper', 'OD (mm)'
            m = MATERIALS[material]
            cfg = m['A_TABLES'][local_num]
            col_labels = m['NPS_labels'] if material == 'steel' else m['OD_labels']
            gas_name = A_table_gas
            table_label = f"{m['table_name'](local_num)} ({A_table_gas})"
        else:
            cfg, col_labels, gas_name = B_TABLES[B_table_number], NPS_labels, 'propane'
            table_label = f'B.{B_table_number} (propane)'

        report(cfg, gas_name, table_label, col_labels, col_name, material)
