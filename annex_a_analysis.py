# Annex A analysis, steel (A.1-A.7) and copper (A.8b-A.14b): Polyflo vs
# GFE-Churchill comparison tables, conservatism figures, H2 capacity tables,
# capacity-ratio maps and worked sizing examples.
# Run: python annex_a_analysis.py export plots h2_tables ...   (no args = all)
import csv
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import xlsxwriter

from friction import churchill_f, colebrook_f, polyflo_f
from master_table_new import (A_TABLES, GASES, L_list_m, MATERIALS, NPS_labels,
                              Pb, R_air, Re_polyflo_laminar, Tb, build_all,
                              size_pipe, z)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
TABLES_DIR = os.path.join(OUT, 'annex_a_tables')
PLOTS_DIR = os.path.join(OUT, 'plots')
COPPER_DIR = os.path.join(OUT, 'copper')
MAPS_DIR = os.path.join(OUT, 'capacity_ratio_maps')

CU = MATERIALS['copper']
OD_labels = CU['OD_labels']
cu_table_name = CU['table_name']

GASES_2 = ['NG', 'H2']
GAS_COLORS = {'NG': 'tab:green', 'H2': 'tab:cyan'}
TABLE_COLORS = {t: c for t, c in zip(range(1, 8), plt.cm.tab10.colors)}

GREEN = PatternFill('solid', fgColor='C6EFCE')
YELLOW = PatternFill('solid', fgColor='FFEB9C')
RED = PatternFill('solid', fgColor='FFC7CE')
HEADER_FILL = PatternFill('solid', fgColor='D9D9D9')
HEADER_FONT = Font(bold=True)
WRAP = Alignment(wrap_text=True, vertical='center', horizontal='center')
CENTER = Alignment(horizontal='center')
THIN = Side(style='thin', color='808080')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

SCHEDULE_TEXT = {
    1: 'inlet below 1.75 kPa, 125 Pa drop',
    2: 'inlet 1.75 to 3.50 kPa, 250 Pa drop',
    3: '14 kPa inlet, 7 kPa drop',
    4: '14 kPa inlet, 10 kPa drop',
    5: '34 kPa inlet, 17 kPa drop',
    6: '70 kPa inlet, 35 kPa drop',
    7: '140 kPa inlet, 70 kPa drop',
}

SUMMARY_HEADERS = ['Table', 'Gas', '%Laminar', '%Transitional', '%Turbulent',
                   'RMS |diff%|', 'min diff%', 'max diff%', '% non-conservative',
                   'mean_non_cons', '%Re 4000-10000']


def regime_of(re):
    return 'Laminar' if re < 2300 else ('Transitional' if re < 4000 else 'Turbulent')


def regime_letter(re):
    return 'L' if re < 2300 else ('T' if re < 4000 else 'U')


# Fill keyed to the absolute Polyflo-vs-Churchill capacity difference.
def fill_for(abs_pct):
    if not np.isfinite(abs_pct):
        return None
    return RED if abs_pct > 10 else (YELLOW if abs_pct >= 5 else GREEN)


def _cols_for(material):
    m = MATERIALS[material]
    labels = m['NPS_labels'] if material == 'steel' else m['OD_labels']
    header = 'L (m) \\ NPS' if material == 'steel' else 'L (m) \\ OD (mm)'
    return m, labels, header


def _summary_row(label, gas_name, diffs, re_vals):
    n = len(diffs)
    if not n:
        return [label, gas_name] + [float('nan')] * 9
    neg = diffs[diffs < 0]
    return [label, gas_name,
            100 * np.sum(re_vals < 2300) / n,
            100 * np.sum((re_vals >= 2300) & (re_vals < 4000)) / n,
            100 * np.sum(re_vals >= 4000) / n,
            float(np.sqrt(np.mean(diffs ** 2))), float(diffs.min()), float(diffs.max()),
            100 * np.sum(diffs < 0) / n,
            float(neg.mean()) if neg.size else float('nan'),
            100 * np.sum((re_vals >= 4000) & (re_vals <= 10000)) / n]


# =================================================================== export

# One worksheet per (table, gas): Q_poly, Q_churchill, signed diff%, Re/regime.
def _build_sheet(wb, table_num, gas_name, res, csv_rows, material):
    m, col_labels, col_header = _cols_for(material)
    ws = wb.create_sheet(f"{m['table_name'](table_num).replace('.', '')}_{gas_name}")
    ws.cell(row=1, column=1, value=col_header).font = HEADER_FONT
    for j, col in enumerate(col_labels):
        c = ws.cell(row=1, column=2 + j, value=col)
        c.font = HEADER_FONT
        c.alignment = WRAP

    regime_style = {'Laminar': dict(b=True), 'Transitional': dict(i=True), 'Turbulent': dict()}
    for i, L in enumerate(L_list_m):
        r = 2 + i
        ws.cell(row=r, column=1, value=float(L)).font = HEADER_FONT
        for j, col in enumerate(col_labels):
            qp, qg, re = res['Q_kW_poly'][i, j], res['Q_kW_gfe'][i, j], res['Re_gfe'][i, j]
            if np.isnan(qp) or np.isnan(qg) or np.isnan(re):
                continue
            diff_pct = res['diff_pct'][i, j]
            regime, re_int = regime_of(re), int(round(re))
            pct_line = f'{diff_pct:+.1f}%\n'
            parts = [f'Q_poly {qp:.0f} kW\nQ_churchill {qg:.0f} kW\n']
            # Underline where Polyflo over-predicts capacity against Churchill.
            parts.append(TextBlock(InlineFont(u='single'), pct_line) if diff_pct < 0 else pct_line)
            parts.append(TextBlock(InlineFont(**regime_style[regime]),
                                   f'Re = {re_int} ({regime})'))
            cell = ws.cell(row=r, column=2 + j)
            cell.value = CellRichText(parts)
            cell.alignment = WRAP
            cell.fill = fill_for(abs(diff_pct))
            csv_rows.append([table_num, gas_name, col, float(L), qp, qg,
                             (qg - qp) / qp * 100, diff_pct, re_int, regime])

    ws.freeze_panes = 'B2'
    ws.column_dimensions['A'].width = 10
    for j in range(len(col_labels)):
        ws.column_dimensions[get_column_letter(2 + j)].width = 20
    ws.row_dimensions[1].height = 20
    for i in range(len(L_list_m)):
        ws.row_dimensions[2 + i].height = 60


# Companion sheet: the same grid holding just Re, a regime letter and a
# non-conservative asterisk, with the same |diff%| fill.
def _build_re_sheet(wb, table_num, gas_name, res, material):
    m, col_labels, col_header = _cols_for(material)
    ws = wb.create_sheet(f"{m['table_name'](table_num).replace('.', '')}_{gas_name}_Re")
    ws.cell(row=1, column=1, value=col_header).font = HEADER_FONT
    for j, col in enumerate(col_labels):
        c = ws.cell(row=1, column=2 + j, value=col)
        c.font = HEADER_FONT
        c.alignment = WRAP

    for i, L in enumerate(L_list_m):
        r = 2 + i
        ws.cell(row=r, column=1, value=float(L)).font = HEADER_FONT
        for j in range(len(col_labels)):
            qp, qg, re = res['Q_kW_poly'][i, j], res['Q_kW_gfe'][i, j], res['Re_gfe'][i, j]
            if np.isnan(qp) or np.isnan(qg) or np.isnan(re):
                continue
            diff_pct = res['diff_pct'][i, j]
            label = f'{int(round(re))} {regime_letter(re)}' + ('*' if diff_pct < 0 else '')
            cell = ws.cell(row=r, column=2 + j, value=label)
            cell.alignment = WRAP
            cell.fill = fill_for(abs(diff_pct))

    ws.cell(row=2 + len(L_list_m) + 1, column=1, value=(
        'L = laminar, T = transitional, U = turbulent; '
        '* = non-conservative (Polyflo overpredicts capacity vs Churchill).'))
    ws.freeze_panes = 'B2'
    ws.column_dimensions['A'].width = 10
    for j in range(len(col_labels)):
        ws.column_dimensions[get_column_letter(2 + j)].width = 10
    ws.row_dimensions[1].height = 20
    for i in range(len(L_list_m)):
        ws.row_dimensions[2 + i].height = 16


def export(tables=(1, 2, 3, 4, 5, 6, 7), gases=GASES_2, material='steel'):
    os.makedirs(TABLES_DIR, exist_ok=True)
    tables_dict = MATERIALS[material]['A_TABLES']
    wb = Workbook()
    wb.remove(wb.active)

    csv_rows, summary_rows = [], []
    for table_num in sorted(tables):
        for gas_name in gases:
            res = build_all(tables_dict[table_num], GASES[gas_name], material=material)
            n_before = len(csv_rows)
            _build_sheet(wb, table_num, gas_name, res, csv_rows, material)
            _build_re_sheet(wb, table_num, gas_name, res, material)
            new = csv_rows[n_before:]
            summary_rows.append(_summary_row(table_num, gas_name,
                                             np.array([r[7] for r in new]),
                                             np.array([r[8] for r in new])))

    ws = wb.create_sheet('summary')
    for j, h in enumerate(SUMMARY_HEADERS):
        ws.cell(row=1, column=1 + j, value=h).font = HEADER_FONT
        ws.column_dimensions[get_column_letter(1 + j)].width = 14
    for i, row in enumerate(summary_rows):
        for j, val in enumerate(row):
            ws.cell(row=2 + i, column=1 + j, value=val)
    ws.freeze_panes = 'A2'

    suffix = '' if material == 'steel' else '_copper'
    xlsx_path = os.path.join(TABLES_DIR, f'annex_a_tables{suffix}.xlsx')
    wb.save(xlsx_path)

    csv_path = os.path.join(TABLES_DIR, f'annex_a_tables{suffix}.csv')
    col_name = 'NPS' if material == 'steel' else 'OD_mm'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['table', 'gas', col_name, 'L_m', 'Q_poly', 'Q_churchill',
                    'diff_pct_rounded', 'diff_pct_raw', 'Re', 'regime'])
        w.writerows(csv_rows)

    print(xlsx_path)
    print(csv_path)
    return xlsx_path, csv_path, summary_rows


def export_all():
    for material in ('steel', 'copper'):
        _, _, summary = export(material=material)
        print('\t'.join(SUMMARY_HEADERS))
        for row in summary:
            print('\t'.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in row))


# ================================================== conservatism, steel

# diff_pct_raw is used throughout: diff_pct_rounded produces false horizontal
# stripes at low Re where capacities are single-digit kW.
def load_rows(csv_path, col_key='NPS'):
    with open(csv_path, newline='') as fh:
        return [dict(table=int(r['table']), gas=r['gas'], col=r[col_key],
                     re=float(r['Re']), diff=float(r['diff_pct_raw']))
                for r in csv.DictReader(fh)]


# Per (table, gas): x = %cells in the Re band, y = mean non-conservatism over
# diff<0 cells, s = %cells non-conservative by more than 5%.
def _triplets(rows, gas, re_lo, re_hi):
    out = []
    for t in sorted({r['table'] for r in rows if r['gas'] == gas}):
        cell = [r for r in rows if r['gas'] == gas and r['table'] == t]
        n = len(cell)
        neg = [r['diff'] for r in cell if r['diff'] < 0]
        out.append((t,
                    100 * sum(1 for r in cell if re_lo <= r['re'] <= re_hi) / n,
                    sum(neg) / len(neg) if neg else float('nan'),
                    100 * sum(1 for r in cell if r['diff'] < -5) / n))
    return out


def plot_diff_vs_re(rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    colors = {c: plt.cm.viridis(i / (len(NPS_labels) - 1)) for i, c in enumerate(NPS_labels)}
    y_min, y_max = -30, 30
    true_min = min(r['diff'] for r in rows)

    for ax, gas in zip(axes, GASES_2):
        ax.axhspan(-5, 5, color='green', alpha=0.15, zorder=0)
        ax.axhspan(5, 10, color='darkorange', alpha=0.18, zorder=0)
        ax.axhspan(-10, -5, color='darkorange', alpha=0.18, zorder=0)
        ax.axhline(0, color='black', linewidth=0.8, zorder=1)
        ax.axvline(Re_polyflo_laminar, color='black', linestyle=':', linewidth=1.2, zorder=1)
        ax.text(Re_polyflo_laminar, y_max - 1, 'Polyflo-laminar crossover',
                rotation=90, va='top', ha='right', fontsize=8)
        for nps in NPS_labels:
            pts = [r for r in rows if r['gas'] == gas and r['col'] == nps]
            if pts:
                ax.scatter([p['re'] for p in pts], [p['diff'] for p in pts],
                           s=10, color=colors[nps], label=nps, zorder=2)
        ax.set_xscale('log')
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Re')
        ax.set_title(gas)
        ax.grid(True, alpha=0.3, which='both')

    axes[0].set_ylabel('diff_pct_raw (%), signed, + = Churchill > Polyflo')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='NPS', loc='center left',
               bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.suptitle('Q_poly vs Q_churchill percent difference vs Reynolds number, A.1-A.7 pooled')
    fig.text(0.5, 0.01, f'y clipped to [{y_min}, {y_max}]; deeper laminar points not '
                        f'shown (true min = {true_min:.1f}%)',
             ha='center', fontsize=8, style='italic')
    fig.tight_layout(rect=(0, 0.03, 0.88, 1))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_conservatism_vs_table(rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    all_tables = sorted({r['table'] for r in rows})
    for ax, gas in zip(axes, GASES_2):
        tables = sorted({r['table'] for r in rows if r['gas'] == gas})
        non_cons, severe = [], []
        for t in tables:
            cell = [r for r in rows if r['gas'] == gas and r['table'] == t]
            non_cons.append(100 * sum(1 for r in cell if r['diff'] < 0) / len(cell))
            severe.append(100 * sum(1 for r in cell if r['diff'] < -5) / len(cell))
        ax.plot(tables, non_cons, marker='o', color=GAS_COLORS[gas], linestyle='-',
                label='% non-conservative')
        ax.plot(tables, severe, marker='s', color=GAS_COLORS[gas], linestyle='--',
                label='% non-cons. by >5%')
        ax.set_xticks(all_tables)
        ax.set_xticklabels([f'A.{t}' for t in all_tables])
        ax.set_xlabel('Annex A table (pressure schedule)')
        ax.set_title(gas)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel('% of cells')
    fig.suptitle('Non-conservatism vs table')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_severe_vs_re_band(rows, re_lo, re_hi, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    corr = {}
    for ax, gas in zip(axes, GASES_2):
        tr = _triplets(rows, gas, re_lo, re_hi)
        x = [p[1] for p in tr]
        y = [p[3] for p in tr]
        ax.scatter(x, y, color=GAS_COLORS[gas], zorder=3)
        for t, xi, yi in zip([p[0] for p in tr], x, y):
            ax.annotate(f'A.{t}', (xi, yi), textcoords='offset points',
                        xytext=(6, 4), fontsize=9)
        ax.set_xlabel(f'% cells with Re in [{re_lo}, {re_hi}]')
        ax.set_title(gas)
        ax.grid(True, alpha=0.3)
        corr[gas] = float(np.corrcoef(x, y)[0, 1])
    axes[0].set_ylabel('% non-conservative by >5%')
    fig.suptitle(f'Severe non-conservatism vs share of cells with Re in [{re_lo}, {re_hi}]')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return corr


# One label per point, stacked downward so none overlap and leaders never cross.
def _declutter(ax, triplets, x_pad, min_gap):
    prev = None
    for t, x, y, _s in sorted(triplets, key=lambda p: -p[2]):
        ly = y if prev is None else min(y, prev - min_gap)
        prev = ly
        ax.annotate(f'A.{t}', xy=(x, y), xytext=(x + x_pad, ly), textcoords='data',
                    fontsize=9, ha='left', va='center', clip_on=False,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.6,
                                    shrinkA=0, shrinkB=4))


def plot_mean_non_cons(rows, re_lo, re_hi, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    all_tr = {g: _triplets(rows, g, re_lo, re_hi) for g in GASES_2}
    y_all = [p[2] for tr in all_tr.values() for p in tr]
    y_lo, y_hi = min(y_all), max(y_all)
    y_pad, min_gap = 0.08 * (y_hi - y_lo), 0.075 * (y_hi - y_lo)

    for ax, gas in zip(axes, GASES_2):
        tr = all_tr[gas]
        for t, x, y, s in tr:
            ax.scatter(x, y, s=40 + 20 * s, color=TABLE_COLORS[t], zorder=3)
        x_max = max(p[1] for p in tr)
        ax.set_xlim(-0.02 * x_max, 1.35 * x_max)
        _declutter(ax, tr, x_pad=0.025 * x_max, min_gap=min_gap)
        ax.set_xlabel(f'% of cells with Re in [{re_lo}, {re_hi}]')
        ax.set_title(gas)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(y_lo - y_pad, y_hi + y_pad)
    axes[0].set_ylabel('mean non-conservatism (%)')
    fig.suptitle(f'Mean non-conservatism vs share of cells with Re in [{re_lo}, {re_hi}]')
    fig.legend(handles=[Line2D([0], [0], marker='o', linestyle='', color=TABLE_COLORS[t],
                               markersize=8, label=f'A.{t}') for t in range(1, 8)],
               title='Table', loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return all_tr


def first_crossing_report(rows):
    print('\nFirst Re >= 4000 (ascending) where |diff_pct_raw| leaves the +/-5% band:')
    for gas in GASES_2:
        for nps in NPS_labels:
            pts = sorted(((r['re'], r['diff']) for r in rows
                          if r['gas'] == gas and r['col'] == nps), key=lambda p: p[0])
            if not pts:
                continue
            if pts[0][0] > 4000:
                print(f'  {gas:>2} NPS {nps:>6}: data-limited (lowest Re = {pts[0][0]:.0f})')
                continue
            crossing = next((re for re, d in pts if re >= 4000 and abs(d) > 5), None)
            print(f'  {gas:>2} NPS {nps:>6}: ' +
                  (f'first crosses at Re = {crossing:.0f}' if crossing
                   else 'stays within +/-5% band for all Re >= 4000'))


def steel_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    rows = load_rows(os.path.join(TABLES_DIR, 'annex_a_tables.csv'), 'NPS')

    p1 = os.path.join(PLOTS_DIR, 'diff_vs_Re_by_gas.png')
    p2 = os.path.join(PLOTS_DIR, 'conservatism_vs_table.png')
    plot_diff_vs_re(rows, p1)
    plot_conservatism_vs_table(rows, p2)
    print(p1)
    print(p2)

    p3 = os.path.join(PLOTS_DIR, 'severe_vs_re_4000_25000.png')
    print(p3, plot_severe_vs_re_band(rows, 4000, 25000, p3))

    p4 = os.path.join(PLOTS_DIR, 'mean_non_cons_vs_re_band.png')
    tr = plot_mean_non_cons(rows, 4000, 10000, p4)
    print(p4)
    print('\ngas\ttable\tx\ty\ts')
    for gas in GASES_2:
        for t, x, y, s in tr[gas]:
            print(f'{gas}\tA.{t}\t{x:.1f}\t{y:.1f}\t{s:.1f}')

    first_crossing_report(rows)


# ================================================== conservatism, copper

CHURCHILL_COLOR = '#1f77b4'
COLEBROOK_COLOR = '#8ec6e6'
POLYFLO_COLOR = '#d2691e'
NG_RANGE_COLOR = '#1f7a63'
H2_RANGE_COLOR = '#c0398f'
AGREEMENT_YELLOW = 'gold'
FIG12_YELLOW = '#F5E6A8'
FIG12_LIGHT_BLUE = '#BFDCEE'
NG_BAR_COLOR = '#1f4e8c'
H2_BAR_COLOR = '#e07b28'
LAM_BG, TRN_BG, TUR_BG = '#D6E6F5', '#FBE8CC', '#DCEEDC'


def fig8_moody(rows, out_path):
    D_list_mm, EPS_MM = CU['D_list_mm'], CU['EPS_MM']
    fig, ax = plt.subplots(figsize=(9, 7))
    Re_sweep = np.logspace(np.log10(500), np.log10(2e7), 400)

    for D in D_list_mm:
        ax.plot(Re_sweep, churchill_f(Re_sweep, EPS_MM, D), color=CHURCHILL_COLOR,
                linewidth=1.3, zorder=3)
    Re_turb = Re_sweep[Re_sweep >= 2300]
    for D in D_list_mm:
        ax.plot(Re_turb, colebrook_f(Re_turb, EPS_MM, D), color=COLEBROOK_COLOR,
                linewidth=1.1, linestyle=':', zorder=2)
    Re_lam = Re_sweep[Re_sweep <= 2300]
    ax.plot(Re_lam, 64 / Re_lam, color='black', linestyle='-.', linewidth=1.6, zorder=4)
    ax.plot(Re_sweep, polyflo_f(Re_sweep), color=POLYFLO_COLOR, linewidth=2.0,
            linestyle='--', zorder=4)
    ax.plot(Re_polyflo_laminar, 64 / Re_polyflo_laminar, marker='o', markerfacecolor='none',
            markeredgecolor=POLYFLO_COLOR, markeredgewidth=1.6, markersize=8, zorder=5)

    for x, lbl in ((2300, 'Re 2 300'), (4000, 'Re 4 000')):
        ax.axvline(x, color='0.6', linestyle=':', linewidth=1, zorder=1)
        ax.text(x, 0.135, lbl, rotation=90, va='top', ha='right', fontsize=8, color='0.5')
    ax.axvspan(4000, 10000, color=AGREEMENT_YELLOW, alpha=0.2, zorder=0)

    for gname, y, mark, color in (('NG', 0.0105, 's', NG_RANGE_COLOR),
                                  ('H2', 0.0097, '^', H2_RANGE_COLOR)):
        pts = [r['re'] for r in rows if r['gas'] == gname]
        ax.plot([min(pts), max(pts)], [y, y], color=color, linewidth=2.0,
                marker=mark, markersize=6, zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(20, 2e7)
    ax.set_ylim(0.009, 0.145)
    ax.set_xlabel('Reynolds number (dimensionless), increasing to the right')
    ax.set_ylabel('Darcy friction factor (dimensionless), increasing upward')
    ax.grid(True, which='both', alpha=0.25)
    handles = [
        Patch(facecolor=AGREEMENT_YELLOW, alpha=0.2, label='Agreement band, Re 4 000 to 10 000'),
        Line2D([0], [0], color='black', lw=1.6, linestyle='-.',
               label='Laminar friction factor, 64 divided by Reynolds number'),
        Line2D([0], [0], color=CHURCHILL_COLOR, lw=1.6,
               label='Churchill closure, seven copper pipe sizes'),
        Line2D([0], [0], color=POLYFLO_COLOR, marker='o', markerfacecolor='none', linestyle='',
               markersize=7, label=f'Polyflo meets laminar line, Re = {Re_polyflo_laminar:.0f}'),
        Line2D([0], [0], color=COLEBROOK_COLOR, lw=1.3, linestyle=':',
               label='Colebrook-White closure, same seven sizes'),
        Line2D([0], [0], color=NG_RANGE_COLOR, lw=2.0, marker='s', markersize=6,
               label='Natural gas operating range, Annex A schedules'),
        Line2D([0], [0], color=POLYFLO_COLOR, lw=2.0, linestyle='--',
               label='Polyflo power law used by the sizing equation'),
        Line2D([0], [0], color=H2_RANGE_COLOR, lw=2.0, marker='^', markersize=6,
               label='Hydrogen operating range, Annex A schedules'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig9_capacity_ratio(out_path):
    D_list_mm, EPS_MM = CU['D_list_mm'], CU['EPS_MM']
    fig, ax = plt.subplots(figsize=(9, 6))
    Re_sweep = np.logspace(np.log10(60), np.log10(2.5e7), 600)

    ratios = []
    for j, D in enumerate(D_list_mm):
        ratio = np.sqrt(churchill_f(Re_sweep, EPS_MM, D) / polyflo_f(Re_sweep))
        ratios.append(ratio)
        ax.plot(Re_sweep, ratio, color=plt.cm.viridis(j / (len(D_list_mm) - 1)),
                linewidth=1.8, label=f'{OD_labels[j]}', zorder=2)

    ax.axhline(1.0, color='black', linewidth=1.0, zorder=3)
    ax.axhspan(0.95, 1.05, color='0.7', alpha=0.35, zorder=0)
    ax.axvspan(4000, 10000, color=AGREEMENT_YELLOW, alpha=0.3, zorder=0)
    ax.axvline(Re_polyflo_laminar, color=POLYFLO_COLOR, linestyle='--', linewidth=1.4, zorder=3)

    blend = ax.get_xaxis_transform()
    ax.text(Re_polyflo_laminar * 0.8, 0.82,
            f'Re = {Re_polyflo_laminar:.0f}\nsizing law crosses\nthe laminar line',
            rotation=90, va='top', ha='right', fontsize=8.5, color=POLYFLO_COLOR, transform=blend)
    ax.text((4000 * 10000) ** 0.5, 0.82, 'agreement band\nRe 4 000 to 10 000',
            rotation=90, va='top', ha='center', fontsize=8.5, color='0.35', transform=blend)
    ax.text(0.03, 0.92, 'over-predicts capacity\n(not conservative)', fontsize=9.5,
            color='0.25', transform=ax.transAxes, va='top')
    ax.text(0.62, 0.12, 'under-predicts capacity\n(conservative)', fontsize=9.5,
            color='0.25', transform=ax.transAxes, va='bottom')

    ax.set_xscale('log')
    ax.set_ylim(min(0.75, min(r.min() for r in ratios)) * 0.95,
                max(r.max() for r in ratios) * 1.05)
    ax.set_xlabel('Reynolds number (dimensionless), increasing to the right')
    ax.set_ylabel('Capacity ratio, sizing equation over flow equation\n'
                  '(dimensionless), increasing upward')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(title='Copper pipe size, OD (mm), Type K', fontsize=8, loc='upper right',
              ncol=3, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig11_re_spans(rows, out_path):
    schedules = [1, 3, 7]                      # A.8b, A.10b, A.14b
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
    y_pos = np.arange(len(OD_labels))
    x_hi = max(r['re'] for r in rows if r['table'] in schedules) * 2

    for ax, tbl in zip(axes, schedules):
        ax.axvspan(1, 2300, color=LAM_BG, zorder=0)
        ax.axvspan(2300, 4000, color=TRN_BG, zorder=0)
        ax.axvspan(4000, x_hi, color=TUR_BG, zorder=0)
        for j, od in enumerate(OD_labels):
            for gname, offset, color, hatch in (('NG', 0.18, NG_BAR_COLOR, None),
                                                ('H2', -0.18, H2_BAR_COLOR, '///')):
                pts = [r['re'] for r in rows if r['table'] == tbl
                       and r['gas'] == gname and r['col'] == str(od)]
                if not pts:
                    continue
                ax.barh(y_pos[j] + offset, max(pts) - min(pts), left=min(pts), height=0.32,
                        color=color, hatch=hatch, edgecolor='white', linewidth=0.5, zorder=2)
        ax.set_xscale('log')
        ax.set_xlim(1, x_hi)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'OD {od}' for od in OD_labels])
        ax.set_xlabel('Reynolds number (dimensionless)')
        ax.set_title(f'schedule of Table {cu_table_name(tbl)}', fontsize=10)

    axes[0].set_ylabel('Nominal pipe size, Type K copper')
    fig.legend(handles=[Patch(facecolor=NG_BAR_COLOR, label='natural gas'),
                        Patch(facecolor=H2_BAR_COLOR, hatch='///', label='hydrogen'),
                        Patch(facecolor=LAM_BG, label='laminar, Re < 2,300'),
                        Patch(facecolor=TRN_BG, label='transitional'),
                        Patch(facecolor=TUR_BG, label='turbulent, Re > 4,000')],
               loc='lower center', ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.03), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig12_diff_vs_re(rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    y_min, y_max = -30, 30
    panel = {'NG': 'Natural gas', 'H2': 'Hydrogen'}

    for ax, gas in zip(axes, GASES_2):
        ax.axhspan(-5, 5, color=FIG12_YELLOW, alpha=0.7, zorder=0)
        ax.axvspan(4000, 10000, color=FIG12_LIGHT_BLUE, alpha=0.7, zorder=0)
        ax.axhline(0, color='black', linewidth=0.9, zorder=1)
        ax.axvline(Re_polyflo_laminar, color=POLYFLO_COLOR, linestyle='--',
                   linewidth=1.4, zorder=1)
        for j, od in enumerate(OD_labels):
            pts = [r for r in rows if r['gas'] == gas and r['col'] == str(od)]
            if not pts:
                continue
            color = plt.cm.viridis(j / (len(OD_labels) - 1))
            re_arr = np.array([p['re'] for p in pts])
            diff_arr = np.array([p['diff'] for p in pts])
            clipped = np.abs(diff_arr) > y_max
            ax.scatter(re_arr[~clipped], diff_arr[~clipped], s=8, color=color, zorder=2)
            if clipped.any():
                ax.scatter(re_arr[clipped], np.sign(diff_arr[clipped]) * y_max * 0.98,
                           s=30, marker='v', facecolor='none', edgecolor=color,
                           linewidth=1.1, zorder=3)
        ax.set_xscale('log')
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Reynolds number (dimensionless), increasing to the right')
        ax.grid(True, alpha=0.3, which='both')
        ax.text(0.96, 0.05, panel[gas], transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    axes[0].set_ylabel('Capacity difference (percent)\n'
                       'above zero: sizing equation is conservative\n'
                       'below zero: sizing equation over-predicts capacity')
    size_handles = [Line2D([0], [0], marker='o', linestyle='',
                           color=plt.cm.viridis(j / (len(OD_labels) - 1)), label=f'{od}')
                    for j, od in enumerate(OD_labels)]
    feature_handles = [
        Line2D([0], [0], color=POLYFLO_COLOR, linestyle='--', lw=1.4,
               label=f'Power law meets laminar line, Re = {Re_polyflo_laminar:.0f}'),
        Patch(facecolor=FIG12_YELLOW, alpha=0.7, label='Within 5 percent of the flow equation'),
        Patch(facecolor=FIG12_LIGHT_BLUE, alpha=0.7, label='Agreement band, Re 4 000 to 10 000'),
        Line2D([0], [0], marker='v', linestyle='', markerfacecolor='none',
               markeredgecolor='0.3', label='Point beyond the axis range, drawn at the edge'),
    ]
    fig.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.2, wspace=0.06)
    leg1 = fig.legend(handles=size_handles, title='OD (mm)', loc='upper center',
                      ncol=len(OD_labels), bbox_to_anchor=(0.28, 0.1), fontsize=8,
                      title_fontsize=8.5)
    fig.legend(handles=feature_handles, loc='upper center', ncol=1,
               bbox_to_anchor=(0.78, 0.14), fontsize=8)
    fig.add_artist(leg1)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig13_mean_non_cons(rows, re_lo, re_hi, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    all_tr = {g: _triplets(rows, g, re_lo, re_hi) for g in GASES_2}
    y_all = [p[2] for tr in all_tr.values() for p in tr if not np.isnan(p[2])]
    y_lo, y_hi = min(y_all), max(y_all)
    y_pad = 0.08 * (y_hi - y_lo)

    for ax, gas in zip(axes, GASES_2):
        for t, x, y, s in all_tr[gas]:
            if np.isnan(y):
                continue
            ax.scatter(x, y, s=40 + 20 * s, color=TABLE_COLORS[t], zorder=3)
            ax.annotate(cu_table_name(t), (x, y), textcoords='offset points',
                        xytext=(6, 4), fontsize=8)
        ax.set_xlabel(f'% cells with Re in [{re_lo}, {re_hi}]')
        ax.set_title(gas)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(y_lo - y_pad, y_hi + y_pad)
    axes[0].set_ylabel('mean non-conservatism (%)')
    fig.suptitle(f'Copper: mean non-conservatism vs share of cells with Re in [{re_lo}, {re_hi}]')
    fig.legend(handles=[Line2D([0], [0], marker='o', linestyle='', color=TABLE_COLORS[t],
                               markersize=8, label=cu_table_name(t)) for t in range(1, 8)],
               title='Table', loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return all_tr


# No copper Figure 10 analogue: the Polyflo law and the laminar line are
# functions of Re alone, so the steel version already covers it.
def copper_plots():
    os.makedirs(COPPER_DIR, exist_ok=True)
    rows = load_rows(os.path.join(TABLES_DIR, 'annex_a_tables_copper.csv'), 'OD_mm')
    fig8_moody(rows, os.path.join(COPPER_DIR, 'fig8_moody_copper.png'))
    fig9_capacity_ratio(os.path.join(COPPER_DIR, 'fig9_capacity_ratio_copper.png'))
    fig11_re_spans(rows, os.path.join(COPPER_DIR, 'fig11_re_spans_copper.png'))
    fig12_diff_vs_re(rows, os.path.join(COPPER_DIR, 'fig12_diff_vs_re_copper.png'))
    tr = fig13_mean_non_cons(rows, 4000, 10000,
                             os.path.join(COPPER_DIR, 'fig13_mean_non_cons_copper.png'))
    print('Figures 8/9/11/12/13 (copper) written to', COPPER_DIR)
    for g in GASES_2:
        print(g, tr[g])


# ======================================================= H2 capacity tables

FONT_NAME, FONT_SIZE = 'Arial', 8


def _f(**kwargs):
    return Font(name=FONT_NAME, size=FONT_SIZE, **kwargs)


# Fill = |diff%| band; font = regime, underlined where Polyflo over-predicts.
def _write_h2_sheet(ws, tbl_num, res):
    Q, diff_signed, Re = res['Q_kW_gfe'], res['diff_pct'], res['Re_gfe']
    diff = np.abs(diff_signed)

    ws.cell(row=1, column=1, value=f'Table A.{tbl_num} - hydrogen capacity, kW, Schedule 40 '
                                   f'steel, {SCHEDULE_TEXT[tbl_num]}').font = _f(bold=True)
    c = ws.cell(row=3, column=1, value='L (m) / NPS')
    c.font = _f(bold=True)
    c.border = BORDER
    for j, nps in enumerate(NPS_labels):
        c = ws.cell(row=3, column=2 + j, value=nps)
        c.font = _f(bold=True)
        c.alignment = CENTER
        c.border = BORDER

    for i, L in enumerate(L_list_m):
        r = 4 + i
        c = ws.cell(row=r, column=1, value=int(L))
        c.font = _f(bold=True)
        c.alignment = CENTER
        c.border = BORDER
        for j in range(len(NPS_labels)):
            q, re = Q[i, j], Re[i, j]
            cell = ws.cell(row=r, column=2 + j, value=int(q) if np.isfinite(q) else None)
            cell.alignment = CENTER
            cell.border = BORDER
            f = fill_for(diff[i, j])
            if f:
                cell.fill = f
            cell.font = (_f(bold=bool(re < 2300), italic=bool(2300 <= re < 4000),
                            underline='single' if diff_signed[i, j] < 0 else None)
                         if np.isfinite(re) else _f())

    ws.cell(row=4 + len(L_list_m) + 1, column=1, value=(
        'Fill: green below 5%, yellow 5 to 10%, red 10% or more, absolute difference '
        'between the Polyflo and GFE-Churchill capacity at that cell. Font: bold laminar '
        '(Re below 2300), italic transitional (2300 to 4000), plain turbulent (above 4000); '
        'underline where non-conservative.')).font = _f()
    ws.column_dimensions['A'].width = 11
    for j in range(len(NPS_labels)):
        ws.column_dimensions[get_column_letter(2 + j)].width = 9
    ws.freeze_panes = 'B4'


def h2_tables():
    os.makedirs(OUT, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    rows = []
    for tbl_num, cfg in A_TABLES.items():
        res = build_all(cfg, GASES['H2'])
        _write_h2_sheet(wb.create_sheet(title=f'A.{tbl_num}'), tbl_num, res)
        Q, Re = res['Q_kW_gfe'], res['Re_gfe']
        diff, n = np.abs(res['diff_pct']), Q.size
        rows.append([f'A.{tbl_num}', SCHEDULE_TEXT[tbl_num],
                     round(100 * np.sum(diff < 5) / n, 1),
                     round(100 * np.sum((diff >= 5) & (diff < 10)) / n, 1),
                     round(100 * np.sum(diff >= 10) / n, 1),
                     round(100 * np.sum(Re < 2300) / n, 1),
                     round(100 * np.sum((Re >= 2300) & (Re < 4000)) / n, 1),
                     int(np.nanmin(Q)), int(np.nanmax(Q))])

    headers = ['table', 'schedule', '% green', '% yellow', '% red',
               '% laminar', '% transitional', 'Q min (kW)', 'Q max (kW)']
    ws = wb.create_sheet(title='summary', index=0)
    ws.cell(row=1, column=1,
            value='Hydrogen capacity tables, GFE-Churchill at eps = 0.045 mm').font = _f(bold=True)
    for j, col in enumerate(headers):
        ws.cell(row=3, column=1 + j, value=col).font = _f(bold=True)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            ws.cell(row=4 + i, column=1 + j, value=val).font = _f()
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 32

    path = os.path.join(OUT, 'hydrogen_capacity_tables.xlsx')
    wb.save(path)
    print(f'Saved: {path}')
    for row in rows:
        print(row)


# ======================================================= capacity ratio maps

RATIO_BANDS = [(-np.inf, 0.50, '#8F3D3D', '#FFFFFF', 'below 0.50'),
               (0.50, 0.75, '#D55E00', '#FFFFFF', '0.50 to 0.74'),
               (0.75, 0.90, '#E69F00', '#173753', '0.75 to 0.89'),
               (0.90, 1.00, '#C6DBEF', '#173753', '0.90 to 0.99'),
               (1.00, 1.05, '#56B4E9', '#173753', '1.00 to 1.04'),
               (1.05, np.inf, '#0072B2', '#FFFFFF', '1.05 and above')]


def _ratio_band(v):
    for i, (lo, hi, *_r) in enumerate(RATIO_BANDS):
        if lo <= v < hi:
            return i
    return len(RATIO_BANDS) - 1


def _regime_grid(Re):
    r = np.full(Re.shape, 'U', dtype='<U1')
    r[Re < 4000] = 'T'
    r[Re < 2300] = 'L'
    return r


def _schedule_text(cfg):
    if cfg['regime'] == 'low':
        tail = 'inlet below 1.75 kPa' if cfg['deltaP_Pa'] == 125 else 'inlet 1.75 to 3.50 kPa'
        return f"{cfg['deltaP_Pa']} Pa pressure drop, {tail}"
    return f"{cfg['P_gauge']} kPa inlet, {cfg['drop_kPa']} kPa drop"


# H2 (GFE-Churchill) over NG (Polyflo), both unrounded, with the H2 regime letter.
def ratio_map(tbl_num):
    os.makedirs(MAPS_DIR, exist_ok=True)
    material = 'steel' if tbl_num <= 7 else 'copper'
    local = tbl_num if tbl_num <= 7 else tbl_num - 7
    m = MATERIALS[material]
    cfg = m['A_TABLES'][local]
    col_labels = m['NPS_labels'] if material == 'steel' else m['OD_labels']
    col_unit = ('Nominal pipe size, NPS (in.)' if material == 'steel'
                else 'Outside diameter, OD (mm)')
    col_short = 'NPS' if material == 'steel' else 'OD (mm)'
    bore_label = 'Bore (in.)' if material == 'steel' else 'Bore (mm)'
    table_disp = m['table_name'](local)
    D_list = m['D_list_mm'] if material == 'copper' else m['D_list_mm'] / 25.4
    in_scope = 6 if material == 'steel' else len(col_labels)

    res_ng = build_all(cfg, GASES['NG'], material=material)
    res_h2 = build_all(cfg, GASES['H2'], material=material)
    NG = res_ng['Q_m3h_poly'] * GASES['NG']['b'] * z
    H2 = res_h2['Q_m3h_gfe'] * GASES['H2']['b'] * z
    RATIO = H2 / NG
    Re_h2, Re_ng = res_h2['Re_gfe'], res_ng['Re_poly']
    REG_H2, REG_NG = _regime_grid(Re_h2), _regime_grid(Re_ng)

    path = os.path.join(MAPS_DIR, f"RP0302_{table_disp.replace('.', '')}"
                                  f"_capacity_ratio_regime_map.xlsx")
    wb = xlsxwriter.Workbook(path)
    NAVY = '#173753'
    f_title = wb.add_format({'bold': True, 'font_size': 14, 'font_color': NAVY,
                             'font_name': 'Calibri'})
    f_sub = wb.add_format({'font_size': 10, 'font_color': '#5A6470', 'font_name': 'Calibri'})
    f_hdr = wb.add_format({'bold': True, 'font_size': 11, 'font_color': '#FFFFFF',
                           'bg_color': NAVY, 'align': 'center', 'valign': 'vcenter',
                           'border': 1, 'border_color': '#FFFFFF', 'font_name': 'Calibri'})
    f_row = wb.add_format({'bold': True, 'font_size': 11, 'font_color': NAVY,
                           'align': 'center', 'valign': 'vcenter', 'border': 1,
                           'border_color': '#FFFFFF', 'bg_color': '#F4F6F8',
                           'font_name': 'Calibri'})
    f_note = wb.add_format({'font_size': 10, 'font_color': NAVY, 'text_wrap': True,
                            'valign': 'top', 'font_name': 'Calibri'})
    f_leg = wb.add_format({'font_size': 10, 'font_color': NAVY, 'valign': 'vcenter',
                           'font_name': 'Calibri'})
    f_leg_h = wb.add_format({'bold': True, 'font_size': 11, 'font_color': NAVY,
                             'font_name': 'Calibri'})

    cell_fmt, cell_fmt_edge = [], []
    for _lo, _hi, bg, fg, _lab in RATIO_BANDS:
        base = dict(font_size=11, bg_color=bg, font_color=fg, align='center',
                    valign='vcenter', border=1, border_color='#FFFFFF', font_name='Calibri')
        cell_fmt.append(wb.add_format(base))
        cell_fmt_edge.append(wb.add_format({**base, 'right': 5, 'right_color': NAVY}))
    leg_fmt = [wb.add_format({'bg_color': bg, 'border': 1, 'border_color': '#FFFFFF'})
               for _lo, _hi, bg, _fg, _l in RATIO_BANDS]

    n_cols = len(col_labels)
    ws = wb.add_worksheet('Capacity ratio map')
    ws.hide_gridlines(2)
    ws.set_column(0, 0, 11)
    ws.set_column(1, n_cols, 10.5)
    ws.set_column(n_cols + 1, n_cols + 1, 2.5)
    ws.set_column(n_cols + 2, n_cols + 2, 4)
    ws.set_column(n_cols + 3, n_cols + 3, 30)
    ws.write(0, 0, f'Hydrogen-to-natural-gas capacity ratio and hydrogen flow regime, '
                   f'CSA B149.1 Table {table_disp} conditions', f_title)
    ws.write(1, 0, 'Hydrogen General Flow Equation with Churchill friction divided by the '
                   'Polyflo reconstruction of the natural-gas table basis. Both sides '
                   'unrounded. Letter gives the hydrogen flow regime.', f_sub)

    R0 = 3
    ws.merge_range(R0, 1, R0, n_cols, col_unit, f_hdr)
    ws.write(R0 + 1, 0, 'Length (m)', f_hdr)
    for j, s in enumerate(col_labels):
        ws.write(R0 + 1, 1 + j, s, f_hdr)
    ws.set_row(R0, 20)
    ws.set_row(R0 + 1, 20)
    for i, L in enumerate(L_list_m):
        r = R0 + 2 + i
        ws.set_row(r, 17)
        ws.write(r, 0, int(L), f_row)
        for j in range(n_cols):
            k = _ratio_band(RATIO[i, j])
            fmt = cell_fmt_edge[k] if j == in_scope - 1 else cell_fmt[k]
            ws.write(r, 1 + j, f'{RATIO[i, j]:.2f} {REG_H2[i, j]}', fmt)

    LR = R0 + 1
    ws.write(LR, n_cols + 2, 'Capacity ratio', f_leg_h)
    for k, (_lo, _hi, _bg, _fg, lab) in enumerate(RATIO_BANDS):
        ws.write_blank(LR + 1 + k, n_cols + 2, None, leg_fmt[k])
        ws.write(LR + 1 + k, n_cols + 3, lab, f_leg)
    ws.write(LR + 8, n_cols + 2, 'Hydrogen flow regime', f_leg_h)
    for k, lab in enumerate(['L  laminar, Re below 2,300',
                             'T  transitional, 2,300 to 4,000',
                             'U  turbulent, Re 4,000 or more']):
        ws.write(LR + 9 + k, n_cols + 3, lab, f_leg)
    ws.write(LR + 13, n_cols + 2, 'Reading a cell', f_leg_h)
    ws.merge_range(LR + 14, n_cols + 2, LR + 17, n_cols + 3,
                   '0.91 T means hydrogen carries 91% of the natural-gas capacity at that '
                   'length and size, and the hydrogen solution is transitional.', f_note)

    lam_h2 = float(100 * (REG_H2 == 'L').mean())
    lam_ng = float(100 * (REG_NG == 'L').mean())
    turb_ng_not_h2 = int(((REG_NG == 'U') & (REG_H2 != 'U')).sum())
    notes = [
        f'Basis. CSA B149.1 Annex A Table {table_disp} schedule: {_schedule_text(cfg)}, '
        f'fitting allowance factor F = {cfg["F"]}, '
        f'{"Schedule 40 steel bore" if material == "steel" else "Type K copper bore"}, '
        f'roughness {m["EPS_MM"]} mm, base conditions 288.6 K and 101.325 kPa.',
        'Numerator. Hydrogen capacity from the General Flow Equation closed with the '
        'Churchill (1977) correlation, solved cell by cell, unrounded.',
        'Denominator. Natural-gas capacity from the Polyflo closed form on the same '
        'schedule, unrounded. This reconstructs the basis of the published table rather '
        'than reproducing its printed values, so the whole-kilowatt rounding floor does '
        'not distort the pattern. The two differ because Polyflo and the Churchill-closed '
        "GFE do not agree on this table's conditions; the ratio is therefore not a "
        'same-model gas-property comparison.',
        f'Regime letter. Hydrogen Reynolds number at the solved capacity. {lam_h2:.1f}% of '
        f'cells are laminar for hydrogen against {lam_ng:.1f}% for natural gas, and '
        f'{turb_ng_not_h2} cells that are turbulent for natural gas are transitional or '
        f'laminar for hydrogen.',
        ('Scope. The heavy rule after NPS 2 marks the end of the size range investigated '
         'in this report. NPS 2-1/2 to NPS 4 are carried for context only. Confirmation '
         'testing remains recommended (3.6.4).') if material == 'steel' else
        ('Scope. All copper OD sizes in this table are within the domestic-scope range '
         'investigated.'),
    ]
    rr = R0 + 2 + len(L_list_m) + 1
    for t in notes:
        span = max(2, -(-len(t) // 118))
        ws.merge_range(rr, 0, rr + span - 1, n_cols, t, f_note)
        for k in range(span):
            ws.set_row(rr + k, 14)
        rr += span + 1
    ws.freeze_panes(R0 + 2, 1)
    ws.set_landscape()
    ws.set_paper(9)
    ws.fit_to_pages(1, 1)
    ws.set_margins(0.4, 0.4, 0.4, 0.4)
    ws.print_area(0, 0, rr, n_cols + 3)

    ws2 = wb.add_worksheet('Underlying values')
    ws2.hide_gridlines(2)
    cols = ['Length (m)', col_short, bore_label, 'H2 capacity, GFE-Churchill (kW)',
            'NG capacity, Polyflo (kW)', 'Capacity ratio', 'H2 Reynolds number',
            'H2 regime', 'NG Reynolds number', 'NG regime']
    for j, (c, w) in enumerate(zip(cols, [11, 10, 10, 26, 22, 14, 18, 11, 18, 11])):
        ws2.write(0, j, c, f_hdr)
        ws2.set_column(j, j, w)
    f_n2 = wb.add_format({'num_format': '0.00', 'font_name': 'Calibri', 'font_size': 11})
    f_n3 = wb.add_format({'num_format': '0.000', 'font_name': 'Calibri', 'font_size': 11})
    f_n0 = wb.add_format({'num_format': '#,##0', 'font_name': 'Calibri', 'font_size': 11})
    f_txt = wb.add_format({'align': 'center', 'font_name': 'Calibri', 'font_size': 11})
    r = 1
    for i, L in enumerate(L_list_m):
        for j, s in enumerate(col_labels):
            ws2.write_number(r, 0, int(L), f_n0)
            ws2.write_string(r, 1, str(s), f_txt)
            ws2.write_number(r, 2, float(D_list[j]), f_n3)
            ws2.write_number(r, 3, float(H2[i, j]), f_n2)
            ws2.write_number(r, 4, float(NG[i, j]), f_n2)
            ws2.write_number(r, 5, float(RATIO[i, j]), f_n3)
            ws2.write_number(r, 6, float(Re_h2[i, j]), f_n0)
            ws2.write_string(r, 7, str(REG_H2[i, j]), f_txt)
            ws2.write_number(r, 8, float(Re_ng[i, j]), f_n0)
            ws2.write_string(r, 9, str(REG_NG[i, j]), f_txt)
            r += 1
    ws2.autofilter(0, 0, r - 1, len(cols) - 1)
    ws2.freeze_panes(1, 0)
    ws2.set_landscape()
    ws2.fit_to_pages(1, 0)

    ws3 = wb.add_worksheet('Summary')
    ws3.hide_gridlines(2)
    ws3.set_column(0, 0, 52)
    ws3.set_column(1, 1, 16)
    ws3.write(0, 0, 'Summary statistics on the unrounded basis', f_title)
    ws3.write(1, 0, 'These are recomputed from the unrounded capacities on this sheet set, '
                    'and may differ slightly from any whole-kilowatt figures quoted '
                    'elsewhere.', f_sub)
    tb = (REG_H2 == 'U') & (REG_NG == 'U')
    stats = [('Cells in the table', RATIO.size),
             ('Cells turbulent for both gases', int(tb.sum()))]
    if tb.sum() > 0:
        stats += [('Ratio over those cells, minimum', float(RATIO[tb].min())),
                  ('Ratio over those cells, maximum', float(RATIO[tb].max())),
                  ('Ratio over those cells, median', float(np.median(RATIO[tb])))]
    if (REG_H2 == 'T').any():
        stats.append(('Median ratio, hydrogen transitional cells',
                      float(np.median(RATIO[REG_H2 == 'T']))))
    if (REG_H2 == 'L').any():
        stats.append(('Median ratio, hydrogen laminar cells',
                      float(np.median(RATIO[REG_H2 == 'L']))))
    stats += [('Hydrogen cells laminar (%)', lam_h2),
              ('Natural-gas cells laminar (%)', lam_ng),
              ('Turbulent on NG but not on H2 (cells)', turb_ng_not_h2),
              (f'{col_labels[0]} ratio at 3 m', float(RATIO[0, 0])),
              (f'{col_labels[0]} ratio at 50 m', float(RATIO[list(L_list_m).index(50), 0])),
              (f'{col_labels[0]} ratio at 300 m', float(RATIO[list(L_list_m).index(300), 0])),
              (f'{col_labels[-1]} ratio, column minimum', float(RATIO[:, -1].min())),
              (f'{col_labels[-1]} ratio, column maximum', float(RATIO[:, -1].max()))]
    for i, (k, v) in enumerate(stats):
        ws3.write(3 + i, 0, k, f_note)
        ws3.write_number(3 + i, 1, v, f_n3 if isinstance(v, float) else f_n0)

    wb.close()
    return path, RATIO


def ratio_maps():
    for tbl_num in range(1, 15):
        path, RATIO = ratio_map(tbl_num)
        print(f'wrote {path}  cells={RATIO.size}  '
              f'ratio range {RATIO.min():.3f} to {RATIO.max():.3f}')


# ============================================== copper published comparison

# CSA B149.1:25 Tables A.8b-A.14b NG capacities, transcribed from the PDF.
PUBLISHED_CU = {
    1: [  # A.8b
        [7, 15, 30, 52, 74, 158, 285], [5, 10, 21, 36, 51, 109, 196],
        [4, 8, 17, 29, 41, 87, 157], [3, 7, 14, 25, 35, 75, 135],
        [3, 6, 13, 22, 31, 66, 119], [3, 6, 11, 20, 28, 60, 108],
        [2, 5, 10, 18, 26, 55, 100], [2, 5, 10, 17, 24, 51, 93],
        [2, 4, 9, 16, 23, 48, 87], [2, 4, 9, 15, 21, 46, 82],
        [2, 4, 8, 14, 20, 42, 75], [2, 4, 7, 13, 18, 39, 70],
        [2, 3, 7, 12, 17, 37, 66], [2, 3, 7, 11, 16, 35, 62],
        [1, 3, 6, 10, 15, 31, 56], [1, 3, 5, 10, 13, 29, 52],
        [1, 2, 5, 9, 13, 27, 48], [1, 2, 5, 8, 12, 25, 45],
        [1, 2, 4, 8, 11, 24, 43], [1, 2, 4, 7, 10, 21, 38],
        [1, 2, 4, 6, 9, 19, 34], [1, 2, 3, 6, 8, 18, 32],
        [1, 2, 3, 5, 8, 16, 29], [1, 1, 3, 5, 7, 14, 26],
        [1, 1, 2, 4, 6, 13, 24], [1, 1, 2, 4, 6, 12, 22],
        [1, 1, 2, 4, 5, 11, 20], [None, 1, 2, 3, 5, 10, 18],
        [None, 1, 2, 3, 4, 9, 16],
    ],
    2: [  # A.9b
        [10, 21, 44, 76, 108, 231, 415], [7, 15, 30, 52, 74, 158, 285],
        [6, 12, 24, 42, 60, 127, 229], [5, 10, 21, 36, 51, 109, 196],
        [4, 9, 18, 32, 45, 97, 174], [4, 8, 17, 29, 41, 87, 157],
        [4, 7, 15, 27, 38, 80, 145], [3, 7, 14, 25, 35, 75, 135],
        [3, 7, 13, 23, 33, 70, 126], [3, 6, 13, 22, 31, 66, 119],
        [3, 6, 12, 20, 29, 61, 110], [3, 5, 11, 19, 27, 57, 102],
        [2, 5, 10, 18, 25, 53, 96], [2, 5, 9, 17, 24, 50, 91],
        [2, 4, 9, 15, 21, 46, 82], [2, 4, 8, 14, 20, 42, 76],
        [2, 4, 7, 13, 18, 39, 70], [2, 3, 7, 12, 17, 37, 66],
        [2, 3, 7, 11, 16, 35, 62], [1, 3, 6, 10, 14, 31, 55],
        [1, 3, 5, 9, 13, 28, 50], [1, 2, 5, 8, 12, 26, 46],
        [1, 2, 4, 8, 11, 24, 43], [1, 2, 4, 7, 10, 21, 38],
        [1, 2, 4, 6, 9, 19, 34], [1, 2, 3, 6, 8, 18, 32],
        [1, 2, 3, 5, 8, 16, 29], [1, 1, 3, 5, 7, 14, 26],
        [1, 1, 2, 4, 6, 13, 24],
    ],
    3: [  # A.10b
        [66, 137, 278, 486, 690, 1474, 2653], [46, 94, 191, 334, 474, 1013, 1824],
        [37, 75, 154, 268, 381, 813, 1464], [31, 65, 131, 230, 326, 696, 1253],
        [28, 57, 116, 204, 289, 617, 1111], [25, 52, 106, 184, 262, 559, 1006],
        [23, 48, 97, 170, 241, 514, 926], [22, 44, 90, 158, 224, 478, 861],
        [20, 42, 85, 148, 210, 449, 808], [19, 39, 80, 140, 199, 424, 763],
        [18, 36, 74, 129, 183, 390, 702], [16, 34, 69, 120, 170, 363, 653],
        [15, 32, 64, 112, 159, 341, 613], [14, 30, 61, 106, 151, 322, 579],
        [13, 27, 55, 96, 136, 291, 525], [12, 25, 51, 88, 126, 268, 483],
        [11, 23, 47, 82, 117, 249, 449], [11, 22, 44, 77, 110, 234, 421],
        [10, 21, 42, 73, 103, 221, 398], [9, 18, 37, 65, 92, 196, 353],
        [8, 16, 34, 59, 83, 178, 320], [7, 15, 31, 54, 76, 163, 294],
        [7, 14, 29, 50, 71, 152, 274], [6, 12, 25, 44, 63, 135, 242],
        [5, 11, 23, 40, 57, 122, 220], [5, 10, 21, 37, 53, 112, 202],
        [5, 10, 20, 34, 49, 104, 188], [4, 9, 17, 31, 43, 93, 167],
        [4, 8, 16, 28, 39, 84, 151],
    ],
    4: [  # A.11b
        [80, 165, 335, 586, 831, 1774, 3194], [55, 113, 230, 402, 571, 1220, 2196],
        [44, 91, 185, 323, 458, 979, 1763], [38, 78, 158, 277, 392, 838, 1509],
        [33, 69, 140, 245, 348, 743, 1337], [30, 62, 127, 222, 315, 673, 1212],
        [28, 57, 117, 204, 290, 619, 1115], [26, 53, 109, 190, 270, 576, 1037],
        [24, 50, 102, 178, 253, 541, 973], [23, 47, 96, 168, 239, 511, 919],
        [21, 44, 89, 155, 220, 470, 846], [20, 41, 83, 144, 205, 437, 787],
        [18, 38, 77, 135, 192, 410, 738], [17, 36, 73, 128, 181, 387, 697],
        [16, 33, 66, 116, 164, 351, 632], [15, 30, 61, 107, 151, 323, 581],
        [14, 28, 57, 99, 141, 300, 541], [13, 26, 53, 93, 132, 282, 507],
        [12, 25, 50, 88, 125, 266, 479], [11, 22, 45, 78, 110, 236, 425],
        [10, 20, 40, 71, 100, 214, 385], [9, 18, 37, 65, 92, 197, 354],
        [8, 17, 35, 60, 86, 183, 329], [7, 15, 31, 53, 76, 162, 292],
        [7, 14, 28, 48, 69, 147, 264], [6, 13, 26, 45, 63, 135, 243],
        [6, 12, 24, 41, 59, 126, 226], [5, 10, 21, 37, 52, 111, 201],
        [5, 9, 19, 33, 47, 101, 182],
    ],
    5: [  # A.12b
        [127, 261, 531, 929, 1317, 2814, 5066], [87, 179, 365, 638, 905, 1934, 3482],
        [70, 144, 293, 512, 727, 1553, 2796], [60, 123, 251, 439, 622, 1329, 2393],
        [53, 109, 222, 389, 552, 1178, 2121], [48, 99, 202, 352, 500, 1067, 1922],
        [44, 91, 185, 324, 460, 982, 1768], [41, 85, 172, 301, 428, 914, 1645],
        [39, 80, 162, 283, 401, 857, 1543], [36, 75, 153, 267, 379, 810, 1458],
        [34, 69, 141, 246, 349, 745, 1341], [31, 64, 131, 229, 324, 693, 1248],
        [29, 60, 123, 215, 304, 650, 1171], [28, 57, 116, 203, 288, 614, 1106],
        [25, 52, 105, 184, 261, 556, 1002], [23, 48, 97, 169, 240, 512, 922],
        [21, 44, 90, 157, 223, 476, 857], [20, 41, 84, 147, 209, 447, 805],
        [19, 39, 80, 139, 198, 422, 760], [17, 35, 71, 123, 175, 374, 674],
        [15, 31, 64, 112, 159, 339, 610], [14, 29, 59, 103, 146, 312, 561],
        [13, 27, 55, 96, 136, 290, 522], [12, 24, 49, 85, 120, 257, 463],
        [10, 22, 44, 77, 109, 233, 419], [10, 20, 40, 71, 100, 214, 386],
        [9, 19, 38, 66, 93, 199, 359], [8, 16, 33, 58, 83, 177, 318],
        [7, 15, 30, 53, 75, 160, 288],
    ],
    6: [  # A.13b
        [208, 428, 872, 1523, 2161, 4617, 8311], [143, 294, 599, 1047, 1485, 3173, 5712],
        [115, 236, 481, 841, 1193, 2548, 4587], [98, 202, 412, 720, 1021, 2181, 3926],
        [87, 179, 365, 638, 905, 1933, 3480], [79, 163, 331, 578, 820, 1751, 3153],
        [72, 150, 304, 532, 754, 1611, 2901], [67, 139, 283, 495, 702, 1499, 2698],
        [63, 131, 266, 464, 658, 1406, 2532], [60, 123, 251, 438, 622, 1328, 2392],
        [55, 113, 231, 403, 572, 1222, 2200], [51, 106, 215, 375, 532, 1137, 2047],
        [48, 99, 201, 352, 499, 1067, 1920], [45, 94, 190, 333, 472, 1008, 1814],
        [41, 85, 172, 301, 427, 913, 1644], [38, 78, 159, 277, 393, 840, 1512],
        [35, 73, 148, 258, 366, 781, 1407], [33, 68, 138, 242, 343, 733, 1320],
        [31, 64, 131, 229, 324, 693, 1247], [28, 57, 116, 203, 287, 614, 1105],
        [25, 52, 105, 184, 260, 556, 1001], [23, 47, 97, 169, 240, 512, 921],
        [21, 44, 90, 157, 223, 476, 857], [19, 39, 80, 139, 197, 422, 759],
        [17, 35, 72, 126, 179, 382, 688], [16, 33, 66, 116, 165, 352, 633],
        [15, 30, 62, 108, 153, 327, 589], [13, 27, 55, 96, 136, 290, 522],
        [12, 24, 50, 87, 123, 263, 473],
    ],
    7: [  # A.14b
        [354, 731, 1487, 2598, 3686, 7874, 14175], [243, 502, 1022, 1786, 2533, 5412, 9742],
        [195, 403, 820, 1434, 2034, 4346, 7823], [167, 345, 702, 1227, 1741, 3719, 6696],
        [148, 306, 622, 1088, 1543, 3296, 5934], [134, 277, 564, 986, 1398, 2987, 5377],
        [124, 255, 519, 907, 1286, 2748, 4947], [115, 237, 483, 844, 1197, 2556, 4602],
        [108, 223, 453, 791, 1123, 2398, 4318], [102, 210, 428, 748, 1061, 2266, 4079],
        [94, 193, 394, 688, 976, 2084, 3752], [87, 180, 366, 640, 908, 1939, 3491],
        [82, 169, 344, 600, 852, 1819, 3275], [77, 159, 324, 567, 805, 1719, 3094],
        [70, 145, 294, 514, 729, 1557, 2803], [64, 133, 270, 473, 671, 1433, 2579],
        [60, 124, 252, 440, 624, 1333, 2399], [56, 116, 236, 413, 585, 1250, 2251],
        [53, 110, 223, 390, 553, 1181, 2126], [47, 97, 198, 345, 490, 1047, 1885],
        [43, 88, 179, 313, 444, 949, 1708], [39, 81, 165, 288, 409, 873, 1571],
        [37, 75, 153, 268, 380, 812, 1461], [32, 67, 136, 237, 337, 719, 1295],
        [29, 60, 123, 215, 305, 652, 1173], [27, 56, 113, 198, 281, 600, 1080],
        [25, 52, 105, 184, 261, 558, 1004], [22, 46, 93, 163, 231, 494, 890],
        [20, 42, 85, 148, 210, 448, 807],
    ],
}


def copper_published():
    os.makedirs(COPPER_DIR, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    n_od = len(OD_labels)

    for tbl_num in range(1, 8):
        grid = PUBLISHED_CU[tbl_num]
        ws = wb.create_sheet(cu_table_name(tbl_num).replace('.', ''))
        Q_ng = build_all(CU['A_TABLES'][tbl_num], GASES['NG'], material='copper')['Q_kW_poly']
        Q_h2 = build_all(CU['A_TABLES'][tbl_num], GASES['H2'], material='copper')['Q_kW_poly']

        ws.cell(row=1, column=1,
                value=f'{cu_table_name(tbl_num)}: Polyflo NG capacity (kW) vs published, '
                      f'and Polyflo H2 capacity, copper Type K').font = HEADER_FONT
        c = ws.cell(row=3, column=1, value='L (m) \\ OD (mm)')
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        for j, od in enumerate(OD_labels):
            for off, suffix in ((0, 'NG poly/pub'), (n_od, 'H2 poly')):
                c = ws.cell(row=3, column=2 + off + j, value=f'{od} {suffix}')
                c.font = HEADER_FONT
                c.fill = HEADER_FILL
                c.alignment = Alignment(horizontal='center', wrap_text=True)

        for i, L in enumerate(L_list_m):
            r = 4 + i
            c = ws.cell(row=r, column=1, value=int(L))
            c.font = HEADER_FONT
            c.fill = HEADER_FILL
            c.alignment = CENTER
            for j in range(n_od):
                q_poly, pub = float(Q_ng[i, j]), grid[i][j]
                cell = ws.cell(row=r, column=2 + j)
                cell.alignment = CENTER
                cell.border = BORDER
                if pub is None:
                    cell.value = f'{q_poly:.0f} / -'
                else:
                    cell.value = f'{q_poly:.0f} / {pub}'
                    cell.fill = GREEN if int(q_poly) == pub else RED
                cell2 = ws.cell(row=r, column=2 + n_od + j, value=f'{float(Q_h2[i, j]):.0f}')
                cell2.alignment = CENTER
                cell2.border = BORDER

        ws.cell(row=4 + len(L_list_m) + 1, column=1, value=(
            'Cell format: computed Polyflo capacity / published CSA capacity (NG columns '
            'only). Green = exact match, red = mismatch. "-" = CSA prints a dash (below '
            'the 0.5 kW rounding floor).'))
        ws.column_dimensions['A'].width = 10
        for j in range(2 * n_od):
            ws.column_dimensions[get_column_letter(2 + j)].width = 13
        ws.freeze_panes = 'B4'

    path = os.path.join(COPPER_DIR, 'copper_A8b_A14b_capacities.xlsx')
    wb.save(path)
    print(path)


# ==================================================== copper tables 12-19

CAPACITY_BANDS = [(0.50, PatternFill('solid', fgColor='C00000')),
                  (0.75, RED), (0.90, YELLOW), (1.00, GREEN),
                  (1.05, PatternFill('solid', fgColor='9BC2E6')),
                  (np.inf, PatternFill('solid', fgColor='2E75B6'))]


def _fill_for_ratio(ratio):
    for edge, fill in CAPACITY_BANDS:
        if ratio < edge:
            return fill
    return CAPACITY_BANDS[-1][1]


def _write_cu_re_map(ws, tbl_num, res, gas_name):
    EPS_MM = CU['EPS_MM']
    diff_signed, Re = res['diff_pct'], res['Re_gfe']
    diff = np.abs(diff_signed)
    ws.cell(row=1, column=1, value=f'{cu_table_name(tbl_num)} ({gas_name}) - Re map, '
                                   f'Churchill basis, copper Type K, '
                                   f'eps={EPS_MM} mm').font = HEADER_FONT
    c = ws.cell(row=3, column=1, value='L (m) / OD (mm)')
    c.font = HEADER_FONT
    c.fill = HEADER_FILL
    c.border = BORDER
    for j, od in enumerate(OD_labels):
        c = ws.cell(row=3, column=2 + j, value=od)
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = CENTER
        c.border = BORDER

    for i, L in enumerate(L_list_m):
        r = 4 + i
        c = ws.cell(row=r, column=1, value=int(L))
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = CENTER
        c.border = BORDER
        for j in range(len(OD_labels)):
            re = Re[i, j]
            cell = ws.cell(row=r, column=2 + j,
                           value=int(round(re)) if np.isfinite(re) else None)
            cell.alignment = CENTER
            cell.border = BORDER
            f = fill_for(diff[i, j])
            if f:
                cell.fill = f
            if np.isfinite(re):
                cell.font = Font(bold=bool(re < 2300), italic=bool(2300 <= re < 4000),
                                 underline='single' if diff_signed[i, j] < 0 else None)

    ws.cell(row=4 + len(L_list_m) + 1, column=1, value=(
        'Fill: green below 5%, yellow 5 to 10%, red 10% or more, absolute difference '
        'between Polyflo and GFE-Churchill capacity. Font: bold laminar, italic '
        'transitional, plain turbulent; underline where non-conservative.'))
    ws.column_dimensions['A'].width = 11
    for j in range(len(OD_labels)):
        ws.column_dimensions[get_column_letter(2 + j)].width = 9
    ws.freeze_panes = 'B4'


def copper_tables():
    os.makedirs(COPPER_DIR, exist_ok=True)
    D_list_mm, EPS_MM = CU['D_list_mm'], CU['EPS_MM']
    A_CU = CU['A_TABLES']
    wb = Workbook()
    wb.remove(wb.active)

    for tbl_num in (1, 3, 7):                  # A.8b, A.10b, A.14b
        res = build_all(A_CU[tbl_num], GASES['NG'], material='copper')
        _write_cu_re_map(wb.create_sheet(f"{cu_table_name(tbl_num).replace('.', '')}_ReMap"),
                         tbl_num, res, 'NG')

    # Table 15: at matched Re and eps/D the GFE-Polyflo difference is gas-independent.
    ws15 = wb.create_sheet('Table15_gas_independence')
    ws15.cell(row=1, column=1, value='Gas-independence check, copper: at matched Re and '
                                     'eps/D, GFE-Churchill vs Polyflo % difference is '
                                     'gas-independent').font = HEADER_FONT
    headers15 = ['Gas', 'Table', 'OD (mm)', 'L (m)', 'Re', 'GFE-Polyflo diff (%)',
                 'Re deviation from anchor (%)']
    for j, h in enumerate(headers15):
        ws15.cell(row=3, column=1 + j, value=h).font = HEADER_FONT
        ws15.column_dimensions[get_column_letter(1 + j)].width = 16

    j_fixed = OD_labels.index(13)              # hold OD, so eps/D, fixed
    candidates = []
    for tbl_num in range(1, 8):
        for gname in ['NG', 'H2', 'butane', 'propane']:
            res = build_all(A_CU[tbl_num], GASES[gname], material='copper')
            for i, L in enumerate(L_list_m):
                re = res['Re_gfe'][i, j_fixed]
                if np.isfinite(re):
                    candidates.append((gname, tbl_num, OD_labels[j_fixed], L, re,
                                       res['diff_pct'][i, j_fixed]))

    anchor = min(candidates, key=lambda c: abs(c[4] - 25000))
    anchor_re = anchor[4]
    picks, used = [anchor], {anchor[0]}
    for cand in sorted(candidates, key=lambda c: abs(c[4] - anchor_re)):
        if cand[0] not in used and abs(cand[4] - anchor_re) / anchor_re < 0.05:
            picks.append(cand)
            used.add(cand[0])
        if len(used) == 4:
            break

    for i, (gname, tbl_num, od, L, re, diff) in enumerate(picks):
        row = 4 + i
        for col, val in enumerate([gname, cu_table_name(tbl_num), od, float(L),
                                   round(re, 2), round(diff, 4),
                                   round(100 * abs(re - anchor_re) / anchor_re, 2)]):
            ws15.cell(row=row, column=1 + col, value=val)

    ws16 = wb.create_sheet('Table16_summary')
    for j, h in enumerate(SUMMARY_HEADERS):
        ws16.cell(row=1, column=1 + j, value=h).font = HEADER_FONT
        ws16.column_dimensions[get_column_letter(1 + j)].width = 14
    row = 2
    for tbl_num in range(1, 8):
        for gname in GASES_2:
            res = build_all(A_CU[tbl_num], GASES[gname], material='copper')
            for j, v in enumerate(_summary_row(cu_table_name(tbl_num), gname,
                                               res['diff_pct'].ravel(),
                                               res['Re_gfe'].ravel())):
                ws16.cell(row=row, column=1 + j, value=v)
            row += 1
    ws16.freeze_panes = 'A2'

    res_ng = build_all(A_CU[1], GASES['NG'], material='copper')
    res_h2 = build_all(A_CU[1], GASES['H2'], material='copper')

    ws17 = wb.create_sheet('Table17_A8b_capacity')
    ws17.cell(row=1, column=1, value=f'{cu_table_name(1)}: copper capacity, kW, NG '
                                     f'published vs H2 GFE-Churchill').font = HEADER_FONT
    ws17.cell(row=3, column=1, value='L (m) / OD (mm), cell: NG / H2').font = HEADER_FONT
    for j, od in enumerate(OD_labels):
        c = ws17.cell(row=3, column=2 + j, value=od)
        c.font = HEADER_FONT
        c.alignment = CENTER
    for i, L in enumerate(L_list_m):
        r = 4 + i
        ws17.cell(row=r, column=1, value=int(L)).font = HEADER_FONT
        for j in range(len(OD_labels)):
            qng, qh2 = res_ng['Q_kW_gfe'][i, j], res_h2['Q_kW_gfe'][i, j]
            re_h2 = res_h2['Re_gfe'][i, j]
            cell = ws17.cell(row=r, column=2 + j,
                             value=f'{qng:.0f} / {qh2:.0f}'
                             if np.isfinite(qng) and np.isfinite(qh2) else None)
            cell.alignment = CENTER
            if np.isfinite(re_h2):
                cell.font = Font(bold=bool(re_h2 < 2300), italic=bool(2300 <= re_h2 < 4000))
    ws17.column_dimensions['A'].width = 11
    for j in range(len(OD_labels)):
        ws17.column_dimensions[get_column_letter(2 + j)].width = 14

    ws18 = wb.create_sheet('Table18_H2_NG_ratio')
    ws18.cell(row=1, column=1, value=f'{cu_table_name(1)}: H2/NG capacity ratio '
                                     f'(GFE-Churchill, unrounded), copper '
                                     f'Type K').font = HEADER_FONT
    ws18.cell(row=3, column=1, value='L (m) / OD (mm)').font = HEADER_FONT
    for j, od in enumerate(OD_labels):
        c = ws18.cell(row=3, column=2 + j, value=od)
        c.font = HEADER_FONT
        c.alignment = CENTER
    for i, L in enumerate(L_list_m):
        r = 4 + i
        ws18.cell(row=r, column=1, value=int(L)).font = HEADER_FONT
        for j in range(len(OD_labels)):
            qng_kw = res_ng['Q_m3h_gfe'][i, j] * GASES['NG']['b'] * z
            qh2_kw = res_h2['Q_m3h_gfe'][i, j] * GASES['H2']['b'] * z
            ratio = qh2_kw / qng_kw
            cell = ws18.cell(row=r, column=2 + j,
                             value=f"{ratio:.2f} {regime_letter(res_h2['Re_gfe'][i, j])}")
            cell.alignment = CENTER
            cell.fill = _fill_for_ratio(ratio)
    ws18.column_dimensions['A'].width = 11
    for j in range(len(OD_labels)):
        ws18.column_dimensions[get_column_letter(2 + j)].width = 10
    ws18.cell(row=4 + len(L_list_m) + 1, column=1, value=(
        'Ratio bands: <0.50 dark red, 0.50-0.75 red, 0.75-0.90 yellow, 0.90-1.00 green, '
        '1.00-1.05 light blue, >1.05 dark blue. Letter: L laminar, T transitional, '
        'U turbulent (H2).'))

    ws19 = wb.create_sheet('Table19_worked_example')
    i_L, j_OD = list(L_list_m).index(15), OD_labels.index(13)
    cfg = A_CU[1]
    ws19.cell(row=1, column=1, value='Quantity').font = HEADER_FONT
    for gname, col in (('NG', 2), ('H2', 3)):
        gas = GASES[gname]
        res = build_all(cfg, gas, material='copper')
        D_mm_val = D_list_mm[j_OD]
        P1, P2 = 101.325 + 125 / 1000, 101.325
        re = res['Re_gfe'][i_L, j_OD]
        rows19 = [
            ('Nominal size (OD, mm)', OD_labels[j_OD]),
            ('Internal diameter (mm)', D_mm_val),
            ('Pipe roughness (mm)', EPS_MM),
            ('Run length (m)', float(L_list_m[i_L])),
            ('Fitting allowance factor F', cfg['F']),
            ('Effective length used (m)', cfg['F'] * L_list_m[i_L]),
            ('Inlet pressure (kPa absolute)', P1),
            ('Outlet pressure (kPa absolute)', P2),
            ('Pressure drop (Pa)', 125),
            ('Mean pressure (kPa absolute)', (2 / 3) * (P1 + P2 - (P1 * P2) / (P1 + P2))),
            ('Relative density (air=1)', gas['S']),
            ('Dynamic viscosity (Pa.s)', gas['mu']),
            ('Density at base conditions (kg/m3)', gas['S'] * (Pb * 1000) / (R_air * Tb)),
            ('Higher heating value (MJ/m3)', gas['b']),
            ('Churchill friction factor', churchill_f(np.array([re]), EPS_MM, D_mm_val)[0]),
            ('Reynolds number', re),
            ('Flow regime', 'laminar' if re < 2300 else
             ('transitional' if re < 4000 else 'turbulent')),
            ('Volumetric flow at base conditions (m3/h)', res['Q_m3h_gfe'][i_L, j_OD]),
            ('Velocity at mean pressure (m/s)', res['V_gfe'][i_L, j_OD]),
            ('Speed of sound (m/s)', res['a']),
            ('Mach number', res['Mach_gfe'][i_L, j_OD]),
            ('Energy capacity, GFE-Churchill (kW)', res['Q_kW_gfe'][i_L, j_OD]),
            ('Energy capacity, Polyflo (kW)', res['Q_kW_poly'][i_L, j_OD]),
        ]
        ws19.cell(row=1, column=col, value=gname).font = HEADER_FONT
        for r, (lab, val) in enumerate(rows19):
            ws19.cell(row=2 + r, column=1, value=lab)
            ws19.cell(row=2 + r, column=col,
                      value=float(val) if isinstance(val, (int, float, np.floating)) else val)
        qp, qg = res['Q_kW_poly'][i_L, j_OD], res['Q_kW_gfe'][i_L, j_OD]
        ws19.cell(row=2 + len(rows19), column=col, value=(qg - qp) / qp * 100)

    ws19.cell(row=2 + len(rows19), column=1,
              value='Difference, (flow eq - sizing eq)/sizing eq (%)')
    ws19.column_dimensions['A'].width = 42
    ws19.column_dimensions['B'].width = 14
    ws19.column_dimensions['C'].width = 14

    path = os.path.join(COPPER_DIR, 'copper_tables_12_19.xlsx')
    wb.save(path)
    print(path)


# ========================================================= worked examples

# load (kW), length (m), Annex A table number
WORKED_CASES = [(30, 15, 1), (40, 15, 1), (30, 15, 2), (25, 15, 2), (30, 15, 3),
                (30, 15, 4), (150, 40, 5), (600, 100, 6), (600, 100, 7)]


def _classify(re):
    return 'LAM' if re < 2300 else ('TRANS' if re < 4000 else 'TURB')


# Capacity (kW) and Re_gfe at one size, interpolated in L.
def _cap_and_re(label, L_m, gas_name, table_number, method, material='steel'):
    m = MATERIALS[material]
    cfg = A_TABLES[table_number] if material == 'steel' else m['A_TABLES'][table_number]
    col_labels = NPS_labels if material == 'steel' else m['OD_labels']
    res = build_all(cfg, GASES[gas_name], material=material)
    Q = res['Q_kW_poly'] if method == 'polyflo' else res['Q_kW_gfe']
    j = col_labels.index(label)
    return (np.interp(L_m, L_list_m, Q[:, j]),
            np.interp(L_m, L_list_m, res['Re_gfe'][:, j]))


def _report_row(load_kW, L_m, gas_name, table_number, material='steel'):
    nps_poly, _, _ = size_pipe(load_kW, L_m, gas_name, 'A', table_number, 'polyflo',
                               material=material)
    nps_gfe, _, _ = size_pipe(load_kW, L_m, gas_name, 'A', table_number, 'gfe',
                              material=material)
    cap_poly, _ = _cap_and_re(nps_poly, L_m, gas_name, table_number, 'polyflo', material)
    cap_gfe, re_gfe = _cap_and_re(nps_poly, L_m, gas_name, table_number, 'gfe', material)
    short = cap_poly - cap_gfe
    print(f"  {gas_name:6s}  NPS(poly)={nps_poly:6s} NPS(gfe)={nps_gfe:6s} "
          f"cap_poly={cap_poly:7.1f} kW  cap_gfe@poly-NPS={cap_gfe:7.1f} kW  "
          f"shortfall={short:6.1f} kW ({short / cap_poly * 100:5.1f}%)  "
          f"regime={_classify(re_gfe)}")
    return nps_poly, nps_gfe


def worked_examples():
    for load_kW, L_m, table_number in WORKED_CASES:
        print(f"\n=== A.{table_number}: load={load_kW} kW, L={L_m} m ===")
        picks = {g: _report_row(load_kW, L_m, g, table_number) for g in ('H2', 'NG')}
        for gas_name, (nps_poly, nps_gfe) in picks.items():
            if nps_poly != nps_gfe:
                # Under Churchill the GFE pick is not always the bigger pipe.
                direction = ('upsized' if NPS_labels.index(nps_gfe) > NPS_labels.index(nps_poly)
                             else 'downsized')
                cap, re = _cap_and_re(nps_gfe, L_m, gas_name, table_number, 'gfe')
                print(f"  {direction} ({gas_name}) NPS={nps_gfe:6s} cap_gfe={cap:7.1f} kW  "
                      f"margin over load={cap - load_kW:6.1f} kW  regime={_classify(re)}")
            elif (load_kW, L_m, table_number) == (30, 15, 2) and gas_name == 'H2':
                nxt = NPS_labels[NPS_labels.index(nps_poly) + 1]
                cap, re = _cap_and_re(nxt, L_m, gas_name, table_number, 'gfe')
                print(f"  upsized (H2) NPS={nxt:6s} cap_gfe={cap:7.1f} kW  "
                      f"margin over load={cap - load_kW:6.1f} kW  regime={_classify(re)}")


STEPS = {
    'export': export_all,
    'plots': steel_plots,
    'copper_plots': copper_plots,
    'h2_tables': h2_tables,
    'ratio_maps': ratio_maps,
    'published': copper_published,
    'copper_tables': copper_tables,
    'worked': worked_examples,
}


if __name__ == '__main__':
    matplotlib.use('Agg')
    for name in ([a for a in sys.argv[1:] if a in STEPS] or list(STEPS)):
        STEPS[name]()
