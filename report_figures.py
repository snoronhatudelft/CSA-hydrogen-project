# Report figures and the standalone pipe-flow demonstrator.
# Run: python report_figures.py three_ratios | energy_chain | fig10_csst_friction | cfd_f_vs_re | pipe | all
import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from friction import colebrook_f, churchill_f, polyflo_f

# Published Polyflo curve: one style in every figure.
C_POLY, LS_POLY = '#d95f02', (0, (8, 4))
# Shifted Polyflo proposal: one color in every figure.
C_SHIFT = '#2166AC'
import master_table_new as mt
import csst_capacity_comparison as cc
import csst_manufacturer_data as md
from csst_friction_models import parker_valid_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "figures")

NAVY = INK = '#173753'
BLUE_FILL, BLUE_EDGE = '#EEF3FA', '#173753'
TAN_FILL, TAN_EDGE = '#FBF1E6', '#8A5A2A'
RED_FILL, RED_EDGE = '#F9E8E8', '#8F3D3D'
GREEN_FILL, GREEN_EDGE = '#E8F2EA', '#2E6B45'
NUM, GREY = '#7A2E0E', '#5A6470'

plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'savefig.facecolor': 'white',
                     'figure.facecolor': 'white'})


# ------------------------------------------------------- drawing helpers

def canvas(w, h, xmax=100, ymax=100):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def box(ax, x0, y0, x1, y1, fill, edge, lw=2.0, r=1.2, z=2):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle=f'round,pad=0,rounding_size={r}',
                       facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def label(ax, x, y, s, size=11, weight='normal', color=INK, ha='center',
          va='center', z=4, style='normal', lsp=1.25):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va=va, zorder=z, style=style, linespacing=lsp)


def arrow(ax, p0, p1, color=NAVY, lw=2.0, rad=0.0, ls='-', z=3, ms=14):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, linewidth=lw,
                        color=color, zorder=z, linestyle=ls, shrinkA=0, shrinkB=0,
                        connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)
    return a


def _text_bottom(fig, ax, texts):
    # Lowest rendered pixel edge of these text artists, in data (axes) y-units.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ymin = min(t.get_window_extent(renderer).ymin for t in texts)
    return ax.transData.inverted().transform((0, ymin))[1]


def _fit_view(fig, ax, pad=1.0):
    # Shrink the figure to the content's real aspect ratio, then set the
    # axis limits to match. The axes always fills the whole figure, so
    # tightening xlim/ylim alone just re-zooms into the same W x H canvas
    # and the exported size never changes - the figure itself has to shrink.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(ax.texts) + list(ax.patches) + list(ax.lines)
    boxes = [a.get_window_extent(renderer) for a in artists]
    x0 = min(b.x0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    inv = ax.transData.inverted()
    (dx0, dy0), (dx1, dy1) = inv.transform((x0, y0)), inv.transform((x1, y1))
    dx0, dx1, dy0, dy1 = dx0 - pad, dx1 + pad, dy0 - pad, dy1 + pad

    W, _ = fig.get_size_inches()
    fig.set_size_inches(W, W * (dy1 - dy0) / (dx1 - dx0))
    ax.set_xlim(dx0, dx1)
    ax.set_ylim(dy0, dy1)


def _write(fig, name, dpi=300):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print('wrote', path)


# ------------------------------------------------------- 3.1.3 three ratios

# Values are the report's own: Table 1 ratios, 3.1.3 flow/Re, 3.1.4 sonic velocity.
def fig_three_ratios():
    W, H = 12.75, 8.35
    YM = 100 * H / W
    LS, VS, SS2 = 12.0, 18.0, 9.2  # head, value, two-line plain-language subtitle
    BOT_PAD = 0.8  # gap kept between subtitle text and box bottom edge
    GAP = 5.0  # vertical gap between bands, for the junction line and arrow
    fig, ax = canvas(W, H, 100, YM)

    b1t = YM - 1.0
    c1 = [(1.5, 31.0), (34.5, 64.0), (67.5, 97.0)]
    m1 = [(a + b) / 2 for a, b in c1]

    band1 = [('Density', 'Ï Hâ‚‚ / Ï NG  =  0.116', VS,
              'Specific gravity 0.0696 against 0.60\nHydrogen is about one-eighth as dense'),
             ('Dynamic viscosity', 'Î¼ Hâ‚‚ / Î¼ NG  =  0.730', VS,
              'Falls, but far less steeply than density\nso density is what drives the difference'),
             ('Heating value per unit volume', 'Ã— 0.324 HHV    Ã— 0.300 LHV', VS - 4.5,
              'Hâ‚‚ â‰ˆ324/274 Btu/scf, NG 1,000/900 (HHV/LHV)\nAbout a third the heat per cubic foot of NG')]
    band1_subs = []
    for m, (head, val, val_size, sub) in zip(m1, band1):
        label(ax, m, b1t - 2.2, head, size=LS, weight='bold')
        label(ax, m, b1t - 5.6, val, size=val_size, weight='bold', color=NUM)
        band1_subs.append(label(ax, m, b1t - 9.0, sub, size=SS2, color=GREY, lsp=1.6))
    b1b = _text_bottom(fig, ax, band1_subs) - BOT_PAD
    for x0, x1 in c1:
        box(ax, x0, b1b, x1, b1t, BLUE_FILL, BLUE_EDGE, r=0.9)

    b2t = b1b - GAP
    r_x0, r_x1 = 6.0, 48.0
    f_x0, f_x1 = 55.0, 97.0
    label(ax, (r_x0 + r_x1) / 2, b2t - 2.2, 'Combined property group  Ï / Î¼',
          size=LS, weight='bold')
    label(ax, (r_x0 + r_x1) / 2, b2t - 5.6, 'Ã— 0.159', size=VS, weight='bold', color=NUM)
    sub_rho = label(ax, (r_x0 + r_x1) / 2, b2t - 9.0,
                    'Density falls faster than viscosity, so hydrogen\n'
                    'flow is naturally less turbulent at equal speed', size=SS2, color=GREY, lsp=1.6)
    label(ax, (f_x0 + f_x1) / 2, b2t - 2.2,
          'Volumetric flow rate Q and in-pipe velocity v', size=LS, weight='bold')
    label(ax, (f_x0 + f_x1) / 2, b2t - 5.6, 'Ã— 3.09  (HHV)          Ã— 3.28  (LHV)',
          size=VS, weight='bold', color=NUM)
    sub_q = label(ax, (f_x0 + f_x1) / 2, b2t - 9.0,
                 'More hydrogen must flow to deliver the same heat,\n'
                 'raising velocity for a given pipe size', size=SS2, color=GREY, lsp=1.6)
    b2b = _text_bottom(fig, ax, [sub_rho, sub_q]) - BOT_PAD
    box(ax, r_x0, b2b, r_x1, b2t, TAN_FILL, TAN_EDGE, r=0.9)
    box(ax, f_x0, b2b, f_x1, b2t, TAN_FILL, TAN_EDGE, r=0.9)

    b3t = b2b - GAP
    e_x0, e_x1 = 22.0, 81.0
    label(ax, (e_x0 + e_x1) / 2, b3t - 2.2, 'Reynolds number Re', size=LS, weight='bold')
    label(ax, (e_x0 + e_x1) / 2, b3t - 5.6, 'Ã— 0.491  (HHV)    to    Ã— 0.521  (LHV)',
          size=VS, weight='bold', color=NUM)
    sub_re = label(ax, (e_x0 + e_x1) / 2, b3t - 9.4,
                   'Even with more gas flowing, hydrogen ends up less turbulent than natural '
                   'gas overall,\nso some pipes sized as turbulent on natural gas need rechecking for hydrogen',
                   size=SS2, color=GREY, lsp=1.6)
    b3b = _text_bottom(fig, ax, [sub_re]) - BOT_PAD
    box(ax, e_x0, b3b, e_x1, b3t, RED_FILL, RED_EDGE, r=0.9)

    jn1, xj1 = (b1b + b2t) / 2, (r_x0 + r_x1) / 2
    for x in (m1[0], m1[1]):
        ax.plot([x, x], [b1b, jn1], color=NAVY, lw=2.0, zorder=3)
    ax.plot([m1[0], m1[1]], [jn1, jn1], color=NAVY, lw=2.0, zorder=3)
    arrow(ax, (xj1, jn1), (xj1, b2t))
    arrow(ax, (m1[2], b1b), (m1[2], b2t))

    jn2, xj2, xf = (b2b + b3t) / 2, (e_x0 + e_x1) / 2, (f_x0 + f_x1) / 2
    for x in (xj1, xf):
        ax.plot([x, x], [b2b, jn2], color=NAVY, lw=2.0, zorder=3)
    ax.plot([xj1, xf], [jn2, jn2], color=NAVY, lw=2.0, zorder=3)
    arrow(ax, (xj2, jn2), (xj2, b3t))

    _fit_view(fig, ax, pad=0.6)
    _write(fig, 'fig_3_1_3_three_ratios.png')


# ------------------------------------------------------- 3.4 energy chain

# Values are the report's own: 3.4.1 through 3.4.7 and Tables 4-6.
# Sized to be read at the 6.3 in. text width: every font is at least 8 pt once printed.
def fig_energy_chain():
    W, H = 11.5, 10.9
    YM = 100 * H / W
    TS, LS, VS, HL, SS, RS = 16.5, 15.5, 19.5, 13.5, 14.5, 13.0
    fig, ax = canvas(W, H, 100, YM)
    C_H, C_L = '#7A2E0E', '#1F4E79'          # HHV and LHV numbers, one colour each

    label(ax, 1.0, YM - 2.2, 'Same appliance, same pipe: hydrogen needs about three times the gas '
          'flow of natural gas,', size=TS, weight='bold', ha='left')
    label(ax, 1.0, YM - 5.6, 'and a one-step pipe upsize offsets what that flow does to pressure drop '
          'and noise', size=TS, weight='bold', ha='left')

    def pair(x, y, hhv, lhv, dx=10.5, size=VS):
        # HHV and LHV value side by side, each with its own label
        label(ax, x - dx / 2 - 1, y + 2.4, 'HHV basis', size=HL, color=C_H, weight='bold')
        label(ax, x + dx / 2 + 1, y + 2.4, 'LHV basis', size=HL, color=C_L, weight='bold')
        label(ax, x - dx / 2 - 1, y - 0.6, hhv, size=size, weight='bold', color=C_H)
        label(ax, x + dx / 2 + 1, y - 0.6, lhv, size=size, weight='bold', color=C_L)

    r1t, r1b = YM - 9.5, YM - 24.0
    r2t, r2b = YM - 30.5, YM - 57.0
    r3t, r3b = YM - 68.0, YM - 87.0

    a_x0, a_x1, b_x0, b_x1 = 1.0, 37.0, 42.0, 99.0
    box(ax, a_x0, r1b, a_x1, r1t, BLUE_FILL, BLUE_EDGE, r=0.9)
    box(ax, b_x0, r1b, b_x1, r1t, TAN_FILL, TAN_EDGE, r=0.9)
    label(ax, (a_x0 + a_x1) / 2, r1t - 2.8, 'Held the same for both gases', size=LS, weight='bold')
    label(ax, (a_x0 + a_x1) / 2, r1t - 9.0,
          'appliance heat input rate\npipe size and length\nsupply pressure, 7 in. w.c. (1.74 kPa)',
          size=SS, color=INK, lsp=1.35)
    label(ax, (b_x0 + b_x1) / 2, r1t - 2.8, 'Gas flow rate and velocity in the pipe  (Â§3.4.1)',
          size=LS, weight='bold')
    pair((b_x0 + b_x1) / 2, r1t - 9.5, 'Ã— 3.1', 'Ã— 3.3', dx=16)
    arrow(ax, (a_x1, (r1t + r1b) / 2), (b_x0, (r1t + r1b) / 2))

    lanes = [(1.0, 24.5), (26.2, 49.2), (50.9, 73.9), (75.6, 99.0)]
    mids = [(a + b) / 2 for a, b in lanes]
    for x0, x1 in lanes:
        box(ax, x0, r2b, x1, r2t, RED_FILL, RED_EDGE, r=0.9)
    head = ['Reynolds number', 'Pressure drop Î”P', 'Momentum flux ÏvÂ²', 'Fitting noise']
    vals = [('Ã— 0.49', 'Ã— 0.52'), ('Ã— 1.11', 'Ã— 1.26'), ('Ã— 1.11', 'Ã— 1.26'), ('+5 dB', '+7 dB')]
    sub = ['Some sizes move from\nturbulent into\ntransitional or\nlaminar flow',
           'Friction factor held\nconstant; the regime\nshift changes it\nfurther (Â§3.5)',
           'Load on bends and\nfittings; the usual\nerosion screen',
           'From fittings only;\nvalve and regulator\nnoise not included']
    ref = ['Â§3.4.2', 'Â§3.4.2, Eq. (2)', 'Â§3.4.3', 'Â§3.4.4, Table 4']
    for i in range(4):
        label(ax, mids[i], r2t - 2.8, head[i], size=LS, weight='bold')
        pair(mids[i], r2t - 9.0, *vals[i], dx=8.5, size=VS - 1.5)
        label(ax, mids[i], r2t - 18.0, sub[i], size=SS - 0.8, color=GREY, lsp=1.3)
        label(ax, mids[i], r2b + 1.8, ref[i], size=RS, color=RED_EDGE, weight='bold')

    jn = (r1b + r2t) / 2
    ax.plot([(b_x0 + b_x1) / 2] * 2, [r1b, jn], color=NAVY, lw=2.0, zorder=3)
    ax.plot([mids[0], mids[3]], [jn, jn], color=NAVY, lw=2.0, zorder=3)
    for m in mids:
        arrow(ax, (m, jn), (m, r2t))

    mB, mA = (1.0, 49.2), (50.9, 99.0)
    box(ax, mB[0], r3b, mB[1], r3t, GREEN_FILL, GREEN_EDGE, r=0.9)
    box(ax, mA[0], r3b, mA[1], r3t, GREEN_FILL, GREEN_EDGE, r=0.9)
    label(ax, sum(mB) / 2, r3t - 2.8, 'Mitigation B: upsize by one pipe size (Â§3.4.7)',
          size=LS - 0.5, weight='bold', color=GREEN_EDGE)
    label(ax, sum(mB) / 2, r3t - 7.6, 'velocity Ã— 0.57 to 0.62    Î”P Ã— 0.25 to 0.30\n'
          'fitting noise âˆ’13 to âˆ’15 dB', size=SS + 0.5, weight='bold', color=NUM, lsp=1.4)
    label(ax, sum(mB) / 2, r3b + 5.0, 'Relative to the original size (Table 6). Brings pressure\n'
          'drop, momentum flux and noise back to or below\nnatural gas in most cases', size=SS - 0.8, color=GREY, lsp=1.3)
    label(ax, sum(mA) / 2, r3t - 2.8, 'Mitigation A: raise supply pressure (Â§3.4.5)',
          size=LS - 0.5, weight='bold', color=GREEN_EDGE)
    pair(sum(mA) / 2, r3t - 8.4, '4.5 psig', '6.1 psig', dx=16, size=VS - 1.5)
    label(ax, sum(mA) / 2, r3b + 3.6, 'Pressure that returns fitting noise to natural gas.\n'
          'The 5 psig indoor cap admits the HHV case only', size=SS - 0.8, color=GREY, lsp=1.3)

    busB = r3t + (r2b - r3t) * 0.45       # routed so the two paths never cross
    ax.plot([25.1, 25.1], [r3t, busB], color=GREEN_EDGE, lw=1.8, zorder=3)
    ax.plot([25.1, 84.0], [busB, busB], color=GREEN_EDGE, lw=1.8, zorder=3)
    for m in (mids[1], mids[2], 84.0):
        arrow(ax, (m, busB), (m, r2b), color=GREEN_EDGE, lw=1.8, ms=12)
    arrow(ax, (92.0, r3t), (92.0, r2b), color=GREEN_EDGE, lw=1.8, ms=12)

    label(ax, 1.0, r3b - 5.0,
          'Values are hydrogen relative to natural gas, except Mitigation B. HHV and LHV: higher and lower '
          'heating value.\nCSST whistling is a separate acoustic risk that neither mitigation is shown to address '
          '(Â§3.4.6).', size=SS - 0.8, color=INK, ha='left', lsp=1.4)

    _write(fig, 'fig_3_4_equal_energy_chain.png')


# ------------------------------------------------------- Figure 9: capacity ratio

# ratio = Q_polyflo / Q_gfe-churchill at fixed dP/L/D/S = sqrt(f_gfe/f_poly).
# Gas-independent: only Re and eps/D (through f_gfe) enter it, per the
# report's own caption. Markers, one shape per NPS, are the extra
# (non-color) identifier for the accessible version.
NPS_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h']


def _fig_capacity_ratio(accessible, name):
    Re = np.logspace(np.log10(20), np.log10(1e7), 400)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(mt.NPS_labels)))

    fig, ax = plt.subplots(figsize=(9.6, 6.3))

    ax.axhspan(0.95, 1.05, color='0.85', alpha=0.6, zorder=0)
    ax.axhline(1.0, color='black', linewidth=1.0, zorder=1)
    ax.axvspan(4000, 10000, color='gold', alpha=0.25, zorder=0)

    for j, (nps, color) in enumerate(zip(mt.NPS_labels, colors)):
        ratio = np.sqrt(churchill_f(Re, mt.EPS_MM, mt.D_list_mm_steel[j]) / polyflo_f(Re))
        ax.plot(Re, ratio, color=color, linewidth=1.8, label=f'NPS {nps}',
                marker=(NPS_MARKERS[j] if accessible else None),
                markevery=40, markersize=6, zorder=3)

    ax.axvline(mt.Re_polyflo_laminar, color='tab:orange', linestyle='--',
              linewidth=1.4, zorder=2)
    ax.annotate(f'Re = {mt.Re_polyflo_laminar:,.0f}\nsizing law crosses\nthe laminar line',
                xy=(mt.Re_polyflo_laminar, 4.7), xytext=(140, 4.7),
                color='tab:orange', fontsize=9, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='tab:orange', linewidth=1.2))
    ax.annotate('agreement band\nRe 4,000 to 10,000', xy=(6300, 5.3),
                color='darkgoldenrod', fontsize=9, ha='center', va='center')
    ax.text(140, 3.3, 'over-predicts capacity\n(not conservative)',
           color=GREY, fontsize=9.5, ha='left', va='center')
    ax.text(3e5, 0.87, 'under-predicts capacity\n(conservative)',
           color=GREY, fontsize=9.5, ha='center', va='center')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(20, 1e7)
    ax.set_ylim(0.75, 6.5)
    yticks = [0.8, 0.9, 1.0, 1.2, 1.5, 2, 3, 4, 6]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Reynolds number (dimensionless), increasing to the right')
    ax.set_ylabel('Capacity ratio, sizing equation over flow equation\n'
                 '(dimensionless), increasing upward')
    ax.grid(True, which='major', alpha=0.25)
    ax.legend(title='Steel pipe size (Schedule 40)', loc='upper right',
             fontsize=9, ncol=3, framealpha=0.95)
    fig.tight_layout()
    _write(fig, name, dpi=200)


def fig_capacity_ratio():
    """Original: NPS curves distinguished by viridis color alone."""
    _fig_capacity_ratio(accessible=False, name='fig_9_capacity_ratio_original.png')


def fig_capacity_ratio_accessible():
    """Revised per Olga's comment on Figure 9: each NPS curve also carries
    its own marker shape, so color is not the only way to tell curves apart."""
    _fig_capacity_ratio(accessible=True, name='fig_9_capacity_ratio_accessible.png')


# ------------------------------------------------------- Figure 11: Reynolds spans

# The three Annex A pressure schedules Figure 11 spans, per Â§3.5.3 (same
# tables used for Tables 12-14): A.1, A.3, A.7.
RE_SPAN_TABLES = (1, 3, 7)

# Subtitle text per panel. The SI figures are the report's own wording
# (Â§3.5.3 / Table 9: "Tables A.1 (<1.75 kPa inlet / 125 Pa drop), A.3 (14 kPa
# inlet, 7 kPa drop) and A.7 (140 kPa inlet, 70 kPa drop)"). The imperial
# figures in parentheses are not unit-converted from those - they're CSA
# B149.1:25 Annex A's own imperial-unit table of the same number and same
# schedule (A.1/A.3/A.7 in Btu/h, in. w.c. and psig) that A.1/A.3/A.7 here
# report in kW/kPa/Pa (i.e. what the standard itself labels A.1b/A.3b/A.7b).
RE_SPAN_SUBTITLE = {
    1: 'Schedule of Table A.1\n<1.75 kPa inlet / 125 Pa drop  (<7 in. w.c. / 0.5 in. w.c.)',
    3: 'Schedule of Table A.3\n14 kPa inlet / 7 kPa drop  (2 psig / 1 psig)',
    7: 'Schedule of Table A.7\n140 kPa inlet / 70 kPa drop  (20 psig / 10 psig)',
}

# Exact colors sampled from the reference figure (not the lighter BLUE_FILL/
# TAN_FILL/GREEN_FILL palette used elsewhere in this file).
GAS_COLORS = {'NG': '#3183B6', 'H2': '#DE8C4F'}
GAS_HATCH = {'NG': '', 'H2': '///'}
REGIME_BG = {'laminar': '#D4E4F3', 'transitional': '#FDDBB9', 'turbulent': '#D5EECF'}


def _re_spans_by_nps(table_num, gas_name):
    """Per-NPS Reynolds-number span (min to max over all tabulated lengths,
    GFE-Churchill basis) for one gas at one Annex A pressure schedule."""
    cfg = mt.A_TABLES[table_num]
    g = mt.GASES[gas_name]
    P1, P2 = mt.get_pressures(cfg)
    _, Re, _ = mt.solve_gfe(g['S'], g['mu'], mt.EPS_MM, P1, P2, cfg['F'])
    return Re.min(axis=0), Re.max(axis=0)


def _fig_reynolds_spans(accessible, titled, out_name):
    n_nps = len(mt.NPS_labels)
    y = np.arange(n_nps)
    bar_h = 0.32
    offsets = {'NG': +bar_h / 2 + 0.02, 'H2': -bar_h / 2 - 0.02}

    spans = {}
    all_lo, all_hi = np.inf, -np.inf
    for t in RE_SPAN_TABLES:
        for gname in ('NG', 'H2'):
            lo, hi = _re_spans_by_nps(t, gname)
            spans[(t, gname)] = (lo, hi)
            all_lo, all_hi = min(all_lo, lo.min()), max(all_hi, hi.max())
    xlo, xhi = all_lo * 0.7, all_hi * 1.4

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.5), sharey=True,
                             constrained_layout=True)

    major_ticks = [1e1, 1e3, 1e5, 1e7]
    for ax, t in zip(axes, RE_SPAN_TABLES):
        ax.set_box_aspect(1)
        ax.axvspan(xlo, 2300, color=REGIME_BG['laminar'], zorder=0)
        ax.axvspan(2300, 4000, color=REGIME_BG['transitional'], zorder=0)
        ax.axvspan(4000, xhi, color=REGIME_BG['turbulent'], zorder=0)

        for gname in ('NG', 'H2'):
            lo, hi = spans[(t, gname)]
            yy = y + offsets[gname]
            ax.barh(yy, hi - lo, left=lo, height=bar_h,
                    color=GAS_COLORS[gname], edgecolor='black', linewidth=0.5,
                    hatch=(GAS_HATCH[gname] if accessible else None), zorder=3,
                    label=(gname if ax is axes[0] else None))

        ax.set_xscale('log')
        ax.set_xlim(xlo, xhi)
        ax.set_xticks(major_ticks)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(RE_SPAN_SUBTITLE[t] if titled else f'Schedule of Table A.{t}',
                     fontsize=10)
        ax.set_xlabel('Reynolds number Re')
        ax.grid(True, which='major', axis='x', alpha=0.25, zorder=1)
        ax.tick_params(axis='y', left=False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f'NPS {s}' for s in mt.NPS_labels])
    axes[0].set_ylim(-0.6, n_nps - 0.4)  # NPS 1/2 at bottom, NPS 4 at top
    axes[0].tick_params(axis='y', left=True)
    axes[0].legend(loc='lower right', fontsize=9)

    fig.suptitle('Figure 11 - Reynolds-number spans of the published schedules, '
                 'natural gas vs. hydrogen', fontsize=12, fontweight='bold')
    _write(fig, out_name, dpi=200)


def fig_reynolds_spans():
    """Original: NG vs H2 distinguished by color alone, panel titles give
    only the table number, no pressure schedule."""
    _fig_reynolds_spans(accessible=False, titled=False,
                        out_name='fig_11_reynolds_spans_original.png')


def fig_reynolds_spans_accessible():
    """Revised per Olga's OA1 comment on Figure 11: NG vs H2 distinguished by
    hatch pattern in addition to color, and each panel title states its
    pressure schedule (SI, with the imperial-unit Annex A table in
    parentheses)."""
    _fig_reynolds_spans(accessible=True, titled=True,
                        out_name='fig_11_reynolds_spans_accessible.png')


# ------------------------------------------------------- Figure 14: CSST characteristic curves

# Published pressure-gradient-vs-flow curves at equal EHD, Manufacturer C
# (Parker Parflex published sizing table) and D (Parker Hannifin measured
# data) against the CSA/ANSI LC 1 reference. Revised per Olga's OA1 comment
# ("Why is this section including data from Manufacturers A, B and C?") to
# drop Gastite and TracPipe. Built by csst_capacity_comparison.py; this just
# reruns its pipeline and saves into report_figures.py's own output folder.

_FIG14_EHD_MAP = None


def _fig14_ehd_map():
    global _FIG14_EHD_MAP
    if _FIG14_EHD_MAP is None:
        alpha, beta = cc.fit_universal_alpha_beta()
        gastite_data = cc.build_source_data("gastite", alpha, beta)
        parker_data = cc.build_source_data("parker", alpha, beta)
        tracpipe_data = cc.build_source_data("tracpipe", alpha, beta)
        parflex_data = cc.build_source_data("parflex", alpha, beta)
        csa_data = cc.build_source_data("csa", alpha, beta)
        _FIG14_EHD_MAP = cc.build_ehd_pipe_map(gastite_data, parker_data, tracpipe_data,
                                               parflex_data, csa_data)
    return _FIG14_EHD_MAP


def fig14_csd():
    """4-panel version (EHD 13, 18, 23, 31), single row: only sizes where
    Manufacturer C, D and the CSA/ANSI LC 1 reference all have data."""
    os.makedirs(OUT, exist_ok=True)
    cc.plot_figure14_csd(_fig14_ehd_map(), OUT, require_all=True)


def fig14_csd_2x2():
    """Same 4-EHD set as fig14_csd, laid out as a 2x2 grid instead of a
    single row of 4."""
    os.makedirs(OUT, exist_ok=True)
    cc.plot_figure14_csd(_fig14_ehd_map(), OUT, require_all=True, layout="grid")


def fig14_csd_all_ehd():
    """6-panel version (adds EHD 39, 62): every size where the CSA/ANSI LC 1
    reference and at least one of Manufacturer C or D have data."""
    os.makedirs(OUT, exist_ok=True)
    cc.plot_figure14_csd(_fig14_ehd_map(), OUT, require_all=False)


def fig15_capacity_vs_ehd():
    """Capacity at a matched pressure gradient (0.02 in. WC/ft) across every
    EHD any source publishes, plus percent difference from the CSA/ANSI LC 1
    reference. Rebuilt from scratch - the original generating script for
    this figure could not be located - matched to the embedded image's own
    caption conventions. Reuses the same full ehd_map as Figure 14."""
    os.makedirs(OUT, exist_ok=True)
    cc.plot_figure15_capacity_vs_ehd(_fig14_ehd_map(), OUT)


def fig15_capacity_vs_ehd_c_only():
    """Same as fig15_capacity_vs_ehd, but Manufacturer C (published sizing
    tables), Manufacturer C (measurement data) and the CSA/ANSI LC 1
    reference only - Manufacturers A and B dropped."""
    os.makedirs(OUT, exist_ok=True)
    cc.plot_figure15_capacity_vs_ehd(
        _fig14_ehd_map(), OUT,
        brands=["Parker Parflex", "Parker Hannifin", "CSA/ANSI LC 1:23"],
        out_name="fig15_capacity_vs_ehd_matched_gradient_c_only.png")


# Colorblind-safe qualitative palette (Okabe-Ito) and matched line styles,
# one pair per NPS column, so pipe size is readable by shape as well as
# hue. Regime markers (turbulent/transitional/laminar) carry a second,
# independent encoding on top, with white edges for contrast.
_H2_RATIO_COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00']
_H2_RATIO_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
_H2_RATIO_MARKERS = {'U': 'o', 'T': '^', 'L': 's'}
_H2_RATIO_MARKER_LABEL = {'U': 'turbulent', 'T': 'transitional', 'L': 'laminar'}


def fig16_h2_ratio_vs_length(tbl_num=1, n_nps=6,
                             out_name="fig16_h2_ratio_vs_length_nps2.png"):
    """Hydrogen capacity over natural-gas capacity against run length, one
    line per NPS, marker shape gives the hydrogen flow regime. Scoped to
    NPS 1/2 through NPS 2 (n_nps=6 of mt.NPS_labels), matching Tables 17-18.
    Rebuilt from scratch - no generating script for this figure could be
    found - with a colorblind-safe palette, per-size line styles and direct
    end-of-line labels so identification does not rely on color alone."""
    os.makedirs(OUT, exist_ok=True)
    cfg = mt.A_TABLES[tbl_num]
    res_ng = mt.build_all(cfg, mt.GASES['NG'])
    res_h2 = mt.build_all(cfg, mt.GASES['H2'])
    NG = res_ng['Q_m3h_poly'] * mt.GASES['NG']['b'] * mt.z
    H2 = res_h2['Q_m3h_gfe'] * mt.GASES['H2']['b'] * mt.z
    ratio = (H2 / NG)[:, :n_nps]
    re_h2 = res_h2['Re_gfe'][:, :n_nps]
    regime = np.full(re_h2.shape, 'U', dtype='<U1')
    regime[re_h2 < 4000] = 'T'
    regime[re_h2 < 2300] = 'L'
    sizes = mt.NPS_labels[:n_nps]

    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.axhline(1.0, color='#555555', linewidth=1.1, zorder=2)

    for j, size in enumerate(sizes):
        color, ls = _H2_RATIO_COLORS[j], _H2_RATIO_LINESTYLES[j]
        ax.plot(mt.L_list_m, ratio[:, j], color=color, linestyle=ls,
                linewidth=1.8, zorder=3, label=f'NPS {size}')
        for reg, marker in _H2_RATIO_MARKERS.items():
            mask = regime[:, j] == reg
            if mask.any():
                ax.scatter(mt.L_list_m[mask], ratio[:, j][mask], marker=marker,
                           s=34, facecolor=color, edgecolor='white',
                           linewidth=0.6, zorder=4)
        ax.annotate(f'NPS {size}', xy=(mt.L_list_m[-1], ratio[-1, j]),
                    xytext=(6, 0), textcoords='offset points', va='center',
                    fontsize=9, color=color, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel('Run length (m), increasing to the right, basis of Table 17',
                  fontsize=11)
    ax.set_ylabel('Hydrogen capacity over natural-gas capacity\n'
                  '(kW/kW, dimensionless), increasing upward', fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.grid(True, which='both', linestyle='--', alpha=0.35)

    size_handles = [plt.Line2D([0], [0], color=_H2_RATIO_COLORS[j],
                               linestyle=_H2_RATIO_LINESTYLES[j], linewidth=1.8,
                               label=f'NPS {s}') for j, s in enumerate(sizes)]
    regime_handles = [plt.Line2D([0], [0], marker=m, color='#333333',
                                 linestyle='None', markersize=7,
                                 markerfacecolor='#333333', markeredgecolor='white',
                                 label=_H2_RATIO_MARKER_LABEL[k])
                      for k, m in _H2_RATIO_MARKERS.items()]
    leg1 = ax.legend(handles=size_handles, loc='lower left', fontsize=9,
                     title='Pipe size', title_fontsize=10, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=regime_handles, loc='center left', fontsize=9,
             title='Hydrogen flow regime', title_fontsize=10, frameon=True)

    fig.tight_layout()
    out_path = os.path.join(OUT, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ------------------------------------------------------- Figure 10 / CFD: CSST friction, D_eff basis

# Volume-equivalent diameter D_eff = sqrt(4*A_mean/pi) of the measured Creo profile, mm.
# Same numbers as GEOM/PROFILE in cfd/new/analyse.py; EHD 13 is the arc-arc profile the
# CFD runs on, so measured and simulated Re and f share one diameter.
D_EFF_MM = {13: 11.4922, 18: 16.1712, 23: 21.3971, 31: 28.8794}
CFD_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cfd', 'new',
                       'prod', 'analysis', 'cfd_results.csv')

FIG10_STYLE = {13: ('#1f77b4', 'o'), 18: ('#ff7f0e', 's'),
               23: ('#2ca02c', '^'), 31: ('#e377c2', 'D')}


def _parker_deff(ehd=None):
    """Manufacturer C (measured) Re and f per EHD, back-calculated on D_eff instead of
    EHD*25.4/32. Only pipe['id_mm'] changes; the back-calculation is md.pipe_arrays."""
    out = {}
    for label, pipe in md.load('parker').items():
        e = int(pipe['ehd'])
        if ehd is not None and e != ehd:
            continue
        pipe['id_mm'] = D_EFF_MM[e]
        arr = md.pipe_arrays(pipe)
        # suspect 3/4 in, 41.5 inWC row excluded, as everywhere else in the CSST analysis
        keep = parker_valid_mask(label, pipe, arr).ravel()
        Re, f = arr['Re'].ravel(), arr['f_moody'].ravel()
        m = keep & np.isfinite(Re) & np.isfinite(f)
        out[e] = (Re[m], f[m])
    return out


def fig10_csst_friction_measured():
    """Figure 10, redrawn on the D_eff basis (previous revision: EHD*25.4/32)."""
    fig, ax = plt.subplots(figsize=(8.2, 5.9))
    for e, (Re, f) in sorted(_parker_deff().items()):
        col, mk = FIG10_STYLE[e]
        ax.scatter(Re, f, s=48, marker=mk, color=col, edgecolor='white', linewidth=0.6,
                   zorder=3, label=f'Size index EHD {e}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(150, 7e4)      # same x limits as the previous (EHD*25.4/32) Figure 10
    ax.set_ylim(0.05, 1.05)    # same 1.3-decade span, shifted up to centre the D_eff points
    ax.set_xlabel('Reynolds number (dimensionless), increasing to the right', fontsize=11)
    ax.set_ylabel('Darcy friction factor back-calculated from the measured data\n'
                  '(dimensionless), increasing upward', fontsize=11)
    ax.grid(True, which='major', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.35)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    fig.tight_layout()
    _write(fig, 'fig10_csst_friction_backcalc_deff.png', dpi=200)


def fig_cfd_f_vs_re_ehd13():
    """EHD 13: Manufacturer C (measured) against the arc-arc transient CFD runs
    (k-omega SST on natural gas and hydrogen, laminar model on natural gas), all on D_eff.
    k-epsilon, sin-profile and steady runs are left out."""
    import csv
    rows = list(csv.DictReader(open(CFD_CSV)))

    def pick(keep):
        s = sorted((r for r in rows if keep(r['case'])), key=lambda r: float(r['Re']))
        return [float(r['Re']) for r in s], [float(r['f']) for r in s]

    sst_ng = pick(lambda c: 'arcarc_transient' in c and 'laminar' not in c and 'H2' not in c)
    sst_h2 = pick(lambda c: 'arcarc_H2_transient' in c)
    lam = pick(lambda c: 'arcarc_laminar_transient' in c)
    Re_pk, f_pk = _parker_deff(13)[13]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    Rl = np.logspace(np.log10(180), np.log10(2300), 50)
    ax.plot(Rl, 64 / Rl, color='0.45', ls='--', lw=1.2, zorder=1, label='Laminar, f = 64/Re')
    ax.scatter(Re_pk, f_pk, s=34, facecolor='none', edgecolor='black', lw=1.1, zorder=3,
               label='Manufacturer C (measured), EHD 13')
    ax.plot(*sst_ng, color='#2F6DB5', lw=1.6, marker='o', ms=8, mec='white', mew=1.2,
            zorder=4, label='CFD, k-Ï‰ SST transient, natural gas')
    ax.plot(*lam, color='#1F9E86', lw=1.6, marker='s', ms=7.5, mec='white', mew=1.2,
            zorder=4, label='CFD, laminar transient, natural gas')
    ax.scatter(*sst_h2, s=150, marker='D', facecolor='none', edgecolor='#D0692A', lw=2.0,
               zorder=5, label='CFD, k-Ï‰ SST transient, hydrogen')
    ax.set(xscale='log', yscale='log', xlim=(180, 2e4), ylim=(0.05, 0.7),
           xlabel='Reynolds number Re', ylabel='Darcy friction factor f')
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which='both', lw=0.4, alpha=0.35)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.95)
    fig.tight_layout()
    _write(fig, 'fig_cfd_f_vs_Re_EHD13.png', dpi=250)


# ------------------------------------------------------- Figure 1: governing documents

def fig1_governing_documents():
    """Figure 1: the four governing documents and what each governs."""
    W, H = 12.0, 7.0
    fig, ax = canvas(W, H, 100, 100 * H / W)
    YM = 100 * H / W
    TAN_HEAD = '#5A3A1A'
    label(ax, 18, YM - 2.5, 'Installation codes', size=15, weight='bold')
    label(ax, 18, YM - 5.3, 'how fuel-gas piping in buildings is sized and installed',
          size=10.5, style='italic')
    label(ax, 78, YM - 2.5, 'Product standard', size=15, weight='bold', color=TAN_HEAD)
    label(ax, 78, YM - 5.3, 'what CSST products must demonstrate to be certified and listed',
          size=10.5, style='italic', color=TAN_HEAD)

    codes = [('CSA B149.1 (Canada)', 'Natural gas and propane\ninstallation code.\n'
              'Sizing: Annex A tables and\nequation; gas multipliers\nin Table A.15.', 50.0, 34.5),
             ('IFGC (United States)', 'International Fuel Gas Code,\nmodel code.\n'
              'Sizing: Section 402\ntables and equations.', 31.0, 18.5),
             ('NFPA 54 (United States)', 'National Fuel Gas Code,\nmodel code.\n'
              'Sizing: Chapter 6 tables and\nSection 6.4 equations.', 15.0, 1.5)]
    for head, body, top, bot in codes:
        box(ax, 6, bot, 30.5, top, BLUE_FILL, BLUE_EDGE, r=0.9)
        label(ax, 18.25, top - 2.8, head, size=12, weight='bold')
        label(ax, 18.25, (top - 4.6 + bot) / 2, body, size=10.5, color='#222222', lsp=1.3)

    sx0, sx1, sy0, sy1 = 36.5, 58.5, 18.5, 40.0
    box(ax, sx0, sy0, sx1, sy1, '#F3EFE6', '#7A6A45', r=0.9)
    label(ax, (sx0 + sx1) / 2, sy1 - 3.3, 'Shared sizing basis', size=12.5, weight='bold')
    label(ax, (sx0 + sx1) / 2, (sy1 - 5.0 + sy0) / 2,
          'All three installation codes\ncarry the same underlying\nsizing equation (Polyflo),\n'
          'calibrated on natural gas\nof relative density 0.60.\nRigid pipe is sized directly\n'
          'from the code tables.', size=10.5, color='#222222', lsp=1.3)

    px0, px1, py0, py1 = 64.5, 93.0, 31.0, 50.0
    box(ax, px0, py0, px1, py1, TAN_FILL, TAN_EDGE, r=0.9)
    label(ax, (px0 + px1) / 2, py1 - 4.3, 'CSA/ANSI LC 1\n(Canada and United States)',
          size=12.5, weight='bold')
    label(ax, (px0 + px1) / 2, (py1 - 7.5 + py0) / 2,
          'Product standard for corrugated\nstainless steel tubing (CSST).\n'
          'Certification flow tests assign an\nEHD rating per product; capacities\n'
          'published in manufacturer tables.', size=10.5, color='#222222', lsp=1.3)

    hx0, hx1, hy0, hy1 = 36.5, 93.0, 1.5, 13.5
    box(ax, hx0, hy0, hx1, hy1, RED_FILL, RED_EDGE, r=0.9)
    label(ax, (hx0 + hx1) / 2, hy1 - 3.0, 'Position of 100% hydrogen', size=12.5, weight='bold')
    label(ax, (hx0 + hx1) / 2, hy0 + 3.8,
          'None of the four documents covers 100% hydrogen: the installation codes are\n'
          'calibrated to natural gas, and the product standard does not list hydrogen\n'
          'as an applicable fuel.', size=10.5, color='#222222', lsp=1.3)

    arrow(ax, (30.5, 42.0), (36.5, 36.0))
    arrow(ax, (30.5, 25.0), (36.5, 28.5))
    arrow(ax, (30.5, 8.0), (36.5, 21.0))
    arrow(ax, (58.5, 34.0), (64.5, 37.0))
    label(ax, 61.5, 30.0, 'CSST sizing\ndefers to LC 1', size=9, color='#333333')
    _write(fig, 'fig1_governing_documents.png', dpi=200)


# ------------------------------------------------------- Figure 2: Re and friction factor

def fig2_reynolds_friction():
    """Figure 2: (a) Re rises with flow through the regimes; (b) f against Re."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2))
    lam, trn, tur = '#D3E4F3', '#FAD9B5', '#D4EDCF'
    a1.axhspan(0, 2300, color=lam)
    a1.axhspan(2300, 4000, color=trn)
    a1.axhspan(4000, 8000, color=tur)
    a1.plot([0, 1], [0, 8000], color='black', lw=2.6)
    for y, t in ((2300, 'Re = 2,300'), (4000, 'Re = 4,000')):
        a1.axhline(y, color='0.45', ls=':', lw=1.1)
        a1.text(0.02, y + 80, t, fontsize=11, color='0.35', va='bottom')
    for y, t in ((1150, 'laminar'), (3150, 'transitional'), (6000, 'turbulent')):
        a1.text(0.93, y, t, fontsize=13, ha='right', va='center')
    a1.set(xlim=(0, 1), ylim=(0, 8000), xticks=[], yticks=[])
    a1.set_xlabel('Flow rate through a given pipe and gas, increasing to the right', fontsize=12)
    a1.set_ylabel('Reynolds number, increasing upward', fontsize=12)
    a1.set_title('(a) Flow rate sets the Reynolds number', fontsize=14)

    Rl = np.logspace(2, np.log10(2300), 100)
    a2.plot(Rl, 64 / Rl, color='black', lw=2.4, ls='-.', label='laminar, f = 64/Re')
    Rt = np.logspace(np.log10(4000), 7, 300)
    a2.plot(Rt, colebrook_f(Rt, mt.EPS_MM, mt.D_list_mm_steel[1]), color='#1f77b4', lw=2.4,
            label='turbulent, one roughness')
    a2.axvspan(2300, 4000, color='#FAD9B5', alpha=0.8)
    a2.text(np.sqrt(2300 * 4000), 0.0125, 'transition', rotation=90, ha='center',
            fontsize=10, color='0.35')
    a2.annotate('steep: friction falls quickly\nas flow rises', xy=(420, 0.152),
                xytext=(1.2e3, 0.24), fontsize=11,
                arrowprops=dict(arrowstyle='->', color='0.3', lw=1.0))
    a2.annotate('nearly flat: friction almost\nindependent of flow', xy=(3e5, 0.0245),
                xytext=(1.2e4, 0.0135), fontsize=11,
                arrowprops=dict(arrowstyle='->', color='0.3', lw=1.0))
    a2.set(xscale='log', yscale='log', xlim=(1e2, 1e7), ylim=(8e-3, 0.8))
    a2.set_xlabel('Reynolds number (dimensionless), increasing to the right', fontsize=12)
    a2.set_ylabel('Darcy friction factor (dimensionless),\nincreasing upward', fontsize=12)
    a2.set_title('(b) Friction behaviour changes with regime', fontsize=14)
    a2.grid(True, which='major', alpha=0.35)
    a2.legend(fontsize=11, loc='upper right')
    fig.tight_layout()
    _write(fig, 'fig2_reynolds_friction_factor.png', dpi=200)


# ------------------------------------------------------- Figure 4: gas-correction routes

def fig4_gas_correction_routes():
    """Figure 4: sqrt(0.60/S) against the B149.1 Table A.15 range and the
    NFPA 54 / IFGC S <= 0.70 route, with hydrogen marked."""
    S_H2 = mt.GASES['H2']['S']
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axvspan(0, 0.70, color='#FDEBD8', zorder=0)
    S = np.linspace(0.05, 2.2, 500)
    ax.plot(S, np.sqrt(0.60 / S), color='0.55', lw=1.8, zorder=2,
            label='physical relationship âˆš(0.60/S)')
    St = np.linspace(0.35, 2.10, 300)
    ax.plot(St, np.sqrt(0.60 / St), color=NAVY, lw=4.0, zorder=3,
            label='CSA B149.1 Table A.15 tabulated range (S = 0.35 to 2.10)')
    ax.plot([0.02, 0.70], [1, 1], color='#8A5A2A', lw=3.0, ls='--', zorder=3,
            label='NFPA 54 B.3.4 / IFGC A102.4: S â‰¤ 0.70 sized from NG tables unmodified (Ã— 1.00)')
    ax.axvline(0.60, color='0.5', ls=':', lw=1.0)
    ax.text(0.61, 0.62, 'design gas, S = 0.60,\nmultiplier 1.00', fontsize=10, color='0.35')

    m_h2, m35 = np.sqrt(0.60 / S_H2), np.sqrt(0.60 / 0.35)
    ax.plot(S_H2, m_h2, 'o', ms=13, mfc='none', mec='#7A2E0E', mew=2.2, zorder=5)
    ax.annotate(f'hydrogen, S = {S_H2}:\nthe relationship requires {m_h2:.2f}',
                xy=(S_H2, m_h2), xytext=(0.30, 2.40), fontsize=11, color='#7A2E0E',
                arrowprops=dict(arrowstyle='->', color='#7A2E0E', lw=1.0))
    ax.plot(0.35, m35, 'o', ms=10, color=NAVY, zorder=5)
    ax.plot(S_H2, m35, 's', ms=9, mfc='white', mec=NAVY, mew=1.3, zorder=5)
    ax.annotate('', xy=(S_H2 + 0.01, m35), xytext=(0.345, m35),
                arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.6, ls=':'))
    ax.annotate(f'lowest tabulated entry:\nS = 0.35, multiplier {m35:.2f}', xy=(0.35, m35),
                xytext=(0.55, 1.55), fontsize=11,
                arrowprops=dict(arrowstyle='->', color='0.3', lw=1.0))
    ax.annotate(f'B149.1 Annex A.2.6 next-higher rule\nreturns the 0.35 entry: {m35:.2f}',
                xy=(S_H2 + 0.01, m35), xytext=(0.62, 1.13), fontsize=11,
                arrowprops=dict(arrowstyle='-', color='0.3', lw=1.0))
    ax.set(xlim=(0, 2.2), ylim=(0.4, 3.2))
    ax.set_xlabel('Relative density S (dimensionless, air = 1), increasing to the right',
                  fontsize=11.5)
    ax.set_ylabel('Capacity multiplier relative to the 0.60 design gas\n'
                  '(dimensionless), increasing upward', fontsize=11.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
    fig.tight_layout()
    _write(fig, 'fig4_gas_correction_routes.png', dpi=200)


# ------------------------------------------------------- Figure 6: noise vs supply pressure

# Dipole intensity ratio I_H2/I_NG = (rho ratio)*(v ratio)^6*(c_NG/c_H2)^3 at equal
# appliance input (Â§3.4.4, Equation (3)): 3.50 (HHV) and 5.16 (LHV).
_S_R = mt.GASES['H2']['S'] / mt.GASES['NG']['S']
_C_R = np.sqrt((mt.GASES['NG']['GAMMA'] / mt.GASES['NG']['S'])
               / (mt.GASES['H2']['GAMMA'] / mt.GASES['H2']['S']))
NOISE_RATIO = {'HHV': _S_R * (37.26 / 12.1) ** 6 * _C_R ** 3,
               'LHV': _S_R * (900.0 / 274.0) ** 6 * _C_R ** 3}
P_NG_PSIG, P_ATM_PSI = 0.25, 14.696


def _noise_db(ratio, p_psig):
    """Fitting-noise level vs the NG baseline: intensity falls as P_abs^-5 at fixed
    standard-volume duty (v ~ 1/P, I ~ rho*v^6 ~ P^-5)."""
    return 10 * np.log10(ratio) - 50 * np.log10((p_psig + P_ATM_PSI) / (P_NG_PSIG + P_ATM_PSI))


def fig6_noise_vs_supply_pressure():
    """Figure 6: fitting-generated sound power for H2 against supply pressure."""
    fig, ax = plt.subplots(figsize=(10, 6.2))
    p = np.linspace(P_NG_PSIG, 8, 300)
    sty = {'HHV': dict(color='#1f77b4', ls='-'), 'LHV': dict(color='#E07B39', ls='--')}
    ax.axhline(0, color='black', lw=1.3)
    ax.text(1.15, 0.15, 'natural-gas baseline at 7 in. W.C. (0.25 psig)', fontsize=10.5,
            color='0.2')
    ax.axvline(5, color='#8F3D3D', lw=2.2)
    ax.text(5.07, 5.9, 'U.S. indoor cap, 5 psig\n(IFGC Â§402.7; NFPA 54 Â§5.5.1,\n'
            'with listed exceptions)', fontsize=10.5, color='#8F3D3D', va='top')
    for basis, r in NOISE_RATIO.items():
        s = sty[basis]
        ax.plot(p, _noise_db(r, p), lw=2.6, label=f'hydrogen, {basis} basis', **s)
        y0 = _noise_db(r, P_NG_PSIG)
        ax.plot(P_NG_PSIG, y0, 's', ms=8, color=s['color'])
        ax.text(0.4, y0, f'+{y0:.1f} dB at the NG delivery pressure', fontsize=10.5,
                color=s['color'], va='center')
        p_eq = (P_NG_PSIG + P_ATM_PSI) * r ** 0.2 - P_ATM_PSI
        ax.plot(p_eq, 0, 'o', ms=11, color=s['color'], zorder=5)
        xt, yt = ((2.4, -1.7) if basis == 'HHV' else (3.95, -2.55))
        ax.annotate(f'equalisation â‰ˆ {p_eq:.1f} psig ({basis})', xy=(p_eq, 0), xytext=(xt, yt),
                    fontsize=11, color=s['color'],
                    arrowprops=dict(arrowstyle='->', color=s['color'], lw=1.0))
    ax.set(xlim=(0, 8), ylim=(-4, 8))
    ax.set_xlabel('Hydrogen supply pressure (psig), increasing to the right', fontsize=11.5)
    ax.set_ylabel('Fitting-generated sound power relative to the\n'
                  'natural-gas baseline (dB), increasing upward', fontsize=11.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    fig.tight_layout()
    _write(fig, 'fig6_noise_vs_supply_pressure.png', dpi=200)


# ------------------------------------------------------- Figures 8, 12, 13: steel Annex A

GASES_NG_H2 = ('NG', 'H2')


def _annex_a_rows(rounded=False):
    """One row per (table, gas, NPS, length) over A.1-A.7, steel: GFE-Churchill Re and
    the Polyflo-vs-GFE difference (Q_gfe - Q_poly)/Q_poly*100. rounded=True uses the
    whole-kW capacities as published, as Table 16 does; cells whose rounded Polyflo
    capacity is zero are dropped."""
    rows = []
    for t, cfg in mt.A_TABLES.items():
        for g in GASES_NG_H2:
            r = mt.build_all(cfg, mt.GASES[g])
            if rounded:
                with np.errstate(divide='ignore', invalid='ignore'):
                    diff = (r['Q_kW_gfe'] - r['Q_kW_poly']) / r['Q_kW_poly'] * 100
            else:
                diff = r['diff_pct']
            for i in range(diff.shape[0]):
                for j, nps in enumerate(mt.NPS_labels):
                    if np.isfinite(diff[i, j]):
                        rows.append(dict(table=t, gas=g, col=nps, L=mt.L_list_m[i],
                                         re=float(r['Re_gfe'][i, j]), diff=float(diff[i, j])))
    return rows


def fig8_friction_factor_steel():
    """Figure 8: Churchill and Colebrook families for the nine steel sizes, Polyflo,
    64/Re, the agreement band and the NG/H2 operating spans over A.1-A.7."""
    rows = _annex_a_rows()
    C_CH, C_CW, C_PF, C_NG, C_H2 = '#1f77b4', '#6cb4e4', C_POLY, '#009E73', '#CC79A7'
    fig, ax = plt.subplots(figsize=(9, 7))
    Re = np.logspace(np.log10(300), 7, 500)
    for D in mt.D_list_mm_steel:
        ax.plot(Re, churchill_f(Re, mt.EPS_MM, D), color=C_CH, lw=1.3, zorder=3)
        Rt = Re[Re >= 2300]
        ax.plot(Rt, colebrook_f(Rt, mt.EPS_MM, D), color=C_CW, lw=1.8, ls=':', zorder=2)
    Rl = np.logspace(np.log10(20), np.log10(2300), 200)
    ax.plot(Rl, 64 / Rl, color='black', lw=1.6, ls='-.', zorder=4)
    Rp = np.logspace(np.log10(20), 7, 300)
    ax.plot(Rp, polyflo_f(Rp), color=C_PF, lw=2.2, ls=LS_POLY, zorder=4)
    ax.plot(mt.Re_polyflo_laminar, 64 / mt.Re_polyflo_laminar, 'o', ms=9, mfc='white',
            mec=C_PF, mew=1.8, zorder=5)
    for x, t in ((2300, 'Re 2 300'), (4000, 'Re 4 000')):
        ax.axvline(x, color='0.5', ls=':', lw=1.0, zorder=1)
        ax.text(x, 0.137, t, rotation=90, va='top', ha='right', fontsize=9, color='0.4')
    ax.axvspan(4000, 10000, color='#F2E05C', alpha=0.5, zorder=0)
    for g, y, mk, c in (('NG', 0.0105, 's', C_NG), ('H2', 0.00985, '^', C_H2)):
        pts = [r['re'] for r in rows if r['gas'] == g]
        ax.plot([min(pts), max(pts)], [y, y], color=c, lw=3.0, marker=mk, ms=9, zorder=4)
    ax.set(xscale='log', yscale='log', xlim=(20, 1e7), ylim=(0.0095, 0.145))
    ax.set_xlabel('Reynolds number (dimensionless), increasing to the right', fontsize=12)
    ax.set_ylabel('Darcy friction factor (dimensionless), increasing upward', fontsize=12)
    ax.grid(True, which='both', alpha=0.2)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(facecolor='#F2E05C', alpha=0.5, label='Agreement band, Re 4 000 to 10 000'),
         Line2D([0], [0], color=C_CH, lw=1.8, label='Churchill closure, nine steel pipe sizes'),
         Line2D([0], [0], color=C_CW, lw=1.8, ls=':', label='Colebrook-White closure, same nine sizes'),
         Line2D([0], [0], color=C_PF, lw=2.2, ls=LS_POLY,
                label='Polyflo power law used by the sizing equation'),
         Line2D([0], [0], color='black', lw=1.6, ls='-.',
                label='Laminar friction factor, 64 divided by Reynolds number'),
         Line2D([0], [0], color=C_PF, marker='o', mfc='white', ls='', ms=8,
                label=f'Polyflo meets laminar line, Re = {mt.Re_polyflo_laminar:.0f}'),
         Line2D([0], [0], color=C_NG, marker='s', ls='', ms=8,
                label='Natural gas operating range, Annex A schedules'),
         Line2D([0], [0], color=C_H2, marker='^', ls='', ms=8,
                label='Hydrogen operating range, Annex A schedules')]
    fig.legend(handles=h, loc='lower center', ncol=2, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    _write(fig, 'fig8_friction_factor.png', dpi=200)


def fig12_pooled_diff_vs_re():
    """Figure 12: Polyflo vs GFE-Churchill capacity difference against Re, all seven
    Annex A schedules pooled, per NPS, NG and H2 panels. Unrounded capacities."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    rows = _annex_a_rows()
    Y = 30
    colors = plt.cm.viridis(np.linspace(0.0, 0.92, len(mt.NPS_labels)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
    for ax, g, title in zip(axes, GASES_NG_H2, ('Natural gas', 'Hydrogen')):
        ax.axhspan(-5, 5, color='#F5E6A8', alpha=0.8, zorder=0)
        ax.axhline(5, color='0.5', ls='--', lw=0.9)
        ax.axhline(-5, color='0.5', ls='--', lw=0.9)
        ax.axvspan(4000, 10000, color='#BFDCEE', alpha=0.8, zorder=0)
        ax.axhline(0, color='0.3', lw=1.0)
        ax.axvline(mt.Re_polyflo_laminar, color='#d95f02', ls='--', lw=1.8, zorder=1)
        for j, nps in enumerate(mt.NPS_labels):
            pts = sorted((r['re'], r['diff']) for r in rows if r['gas'] == g and r['col'] == nps)
            re_a = np.array([p[0] for p in pts]); d = np.array([p[1] for p in pts])
            clip = d < -Y
            ax.plot(re_a[~clip], d[~clip], color=colors[j], lw=1.2, marker=NPS_MARKERS[j],
                    ms=4.5, zorder=3)
            if clip.any():
                ax.scatter(re_a[clip], np.full(clip.sum(), -Y), marker='v', s=45,
                           facecolor='white', edgecolor='0.2', lw=0.9, zorder=4)
        ax.set_xscale('log')
        ax.set(xlim=(12, 1e7), ylim=(-32, 30))
        ax.set_xlabel('Reynolds number (dimensionless), increasing to the right', fontsize=11.5)
        ax.grid(True, which='both', alpha=0.25)
        ax.text(0.97, 0.05, title, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    axes[0].set_ylabel('Capacity difference (percent)\nabove zero: sizing equation is conservative\n'
                       'below zero: sizing equation over-predicts capacity', fontsize=11.5)
    h = [Line2D([0], [0], color=colors[j], marker=NPS_MARKERS[j], lw=1.2, label=n)
         for j, n in enumerate(mt.NPS_labels)]
    h += [Line2D([0], [0], color='#d95f02', ls='--', lw=1.8,
                 label=f'Power law meets laminar line, Re = {mt.Re_polyflo_laminar:.0f}'),
          Patch(facecolor='#BFDCEE', label='Agreement band, Re 4 000 to 10 000'),
          Patch(facecolor='#F5E6A8', label='Within 5 percent of the flow equation'),
          Line2D([0], [0], marker='v', ls='', mfc='white', mec='0.2',
                 label='Point beyond the axis range, drawn at the edge')]
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=10, frameon=True,
               title='Nominal pipe size (inch), then plot features', title_fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    _write(fig, 'fig12_pooled_diff_vs_re.png', dpi=200)


def fig13_nonconservatism_vs_agreement_share():
    """Figure 13: mean over-prediction (mean |diff| over non-conservative cells) against
    the share of cells with Re 4000 to 10000, one point per schedule and gas. Whole-kW
    capacities as published, the same basis as Table 16."""
    rows = _annex_a_rows(rounded=True)
    sty = {'NG': ('#0072B2', 'o', 'Natural gas'), 'H2': ('#D55E00', '^', 'Hydrogen')}
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for g in GASES_NG_H2:
        col, mk, name = sty[g]
        first = True
        for t in sorted(mt.A_TABLES):
            cell = [r for r in rows if r['gas'] == g and r['table'] == t]
            x = 100 * np.mean([4000 <= r['re'] <= 10000 for r in cell])
            neg = [-r['diff'] for r in cell if r['diff'] < 0]
            y = float(np.mean(neg)) if neg else np.nan
            ax.scatter(x, y, s=110, marker=mk, color=col, zorder=3,
                       label=name if first else None)
            first = False
            dx, dy = ((0.5, 0.9) if g == 'NG' else (0.5, -1.0))
            ax.text(x + dx, y + dy, f'A.{t}', color=col, fontsize=11, fontweight='bold',
                    va='center')
    ax.set(xlim=(-1.5, 30), ylim=(-1.5, 32))
    ax.set_xlabel('Share of table cells with Reynolds number between 4 000 and 10 000 '
                  '(percent of cells)', fontsize=11.5)
    ax.set_ylabel('Mean over-prediction of capacity by the sizing equation\n'
                  '(percent, larger is worse, zero is exact agreement)', fontsize=11.5)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Gas', fontsize=11, loc='upper left')
    fig.tight_layout()
    _write(fig, 'fig13_nonconservatism_vs_agreement_share.png', dpi=200)


# ------------------------------------------------------- Figures B.1, B.2: diameter appendix

def figB1_diameter_rescale_demo():
    """Figure B.1: Manufacturer C (measured) EHD 13 at the nominal bore EHD*25.4/32 and
    at the bore that puts the laminar points on 64/Re; every point moves along slope -5."""
    pipe = next(p for p in md.load('parker').values() if int(p['ehd']) == 13)
    arr = md.pipe_arrays(pipe)
    Re, f = arr['Re'].ravel(), arr['f_moody'].ravel()
    m = np.isfinite(Re) & np.isfinite(f)
    Re, f = Re[m], f[m]
    D = float(arr['D_mm'])
    lam = Re < 2300
    gam = float(np.median((64.0 / (f[lam] * Re[lam])) ** 0.25))

    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.scatter(Re, f, s=48, marker='o', color='#1f77b4', edgecolor='white', linewidth=0.6,
               zorder=3, label=f'nominal D = {D:.2f} mm (EHDÂ·25.4/32)')
    ax.scatter(Re / gam, f * gam ** 5, s=48, marker='^', color='#D0692A', edgecolor='white',
               linewidth=0.6, zorder=4,
               label=f'rescaled D = {gam * D:.2f} mm (Î³ = {gam:.3f}, laminar-matched)')
    Rl = np.logspace(np.log10(Re.min() / gam * 0.95), np.log10(2300), 60)
    ax.plot(Rl, 64 / Rl, color='black', lw=1.8, ls='--', zorder=2, label='f = 64/Re')
    i0 = np.argmin(Re)
    g_line = np.array([0.85, 1.10])
    ax.plot(Re[i0] / g_line, f[i0] * g_line ** 5, color='0.35', lw=1.4, ls=':',
            label='direction a point moves when D changes (slope -5)')
    ax.set(xscale='log', yscale='log', xlabel='Re (-)', ylabel='f (-)')
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.95)
    fig.tight_layout()
    _write(fig, 'figB1_diameter_rescale_demo_ehd13.png', dpi=200)


def figB2_lc1_friction_backcalc():
    """Figure B.2: friction factor back-calculated from the CSA/ANSI LC 1:23 Tables 1a)
    to 1c), every EHD, on the nominal bore EHD*25.4/32 (no measured profile exists)."""
    pipes = sorted(md.load('csa').values(), key=lambda p: p['ehd'])
    pipes = [p for p in pipes if p['gas'] == 'natural_gas']
    cols = plt.cm.viridis(np.linspace(0.0, 0.92, len(pipes)))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for c, mk, p in zip(cols, itertools.cycle(markers), pipes):
        arr = md.pipe_arrays(p)
        Re, f = arr['Re'].ravel(), arr['f_moody'].ravel()
        m = np.isfinite(Re) & np.isfinite(f)
        ax.scatter(Re[m], f[m], s=44, marker=mk, color=c, edgecolor='white', linewidth=0.6,
                   zorder=3, label=f'EHD {int(p["ehd"])}')
    ax.set(xscale='log', yscale='log')
    ax.set_xlabel('Reynolds number Re (-)', fontsize=11)
    ax.set_ylabel('Darcy-Weisbach friction factor f (-)', fontsize=11)
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.95)
    fig.tight_layout()
    _write(fig, 'figB2_lc1_friction_backcalc.png', dpi=200)


# ------------------------------------------------------- Â§3.7.3: shifted Polyflo sizing law

K_SHIFT = 1.18          # clears the transition peak of every NPS 1/2-2 Churchill curve at
                        # EPS_SHIFT (exact 1.174, checks.py check 11), rounded up
EPS_SHIFT = 0.09        # mm, upper end of commercial steel
RE_TURB_SWITCH = 2e4    # the Churchill branch only applies past the transition peak


def shifted_sizing_f(Re, D_mm, k=K_SHIFT, eps_mm=EPS_SHIFT):
    """Proposed friction law for one pipe size: 64/Re below Re*, Polyflo x k above it, and
    that size's own Churchill curve wherever it lies above the shifted Polyflo at high Re."""
    f_ch = churchill_f(Re, eps_mm, D_mm)
    return np.maximum.reduce([64.0 / Re, k * polyflo_f(Re),
                              np.where(Re > RE_TURB_SWITCH, f_ch, 0.0)])


def _switch_re(D_mm, k=K_SHIFT, eps_mm=EPS_SHIFT):
    from scipy.optimize import brentq
    return brentq(lambda R: churchill_f(R, eps_mm, D_mm) - k * polyflo_f(R), RE_TURB_SWITCH, 1e9)


# Faint Churchill curves: black, one line type per size, so the proposed branches keep the colour.
FAINT_LS = ['-', '--', '-.', ':', (0, (6, 2, 1, 2, 1, 2)), (0, (10, 3))]
FAINT_C = '0.6'


def fig_shifted_polyflo(ref_eps=None, out_name='fig_shifted_polyflo_sizing_law.png'):
    """Proposed sizing law against Churchill (eps = 0.09 mm, NPS 1/2 to 2): 64/Re below Re*,
    Polyflo shifted up by k from Re*, and each size's own Churchill curve beyond the Re at
    which that curve rises above the shifted Polyflo. Faint grey curves: Churchill where the
    shifted Polyflo governs, or, with ref_eps, the full Churchill curves at that roughness."""
    Re = np.logspace(2, 7, 1200)
    Re_star = (64 / (K_SHIFT * mt.a0)) ** (1 / (1 - mt.n))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, 6))
    C_LAM = '#B2182B'
    tag = ',' if ref_eps is None else f' Îµ = {EPS_SHIFT} mm,'

    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    ax.axvspan(2300, 4000, color='0.92', zorder=0)
    lo = Re <= Re_star
    ax.plot(Re[lo], 64 / Re[lo], color=C_LAM, lw=2.8, zorder=5,
            label='Proposed, below Re*: laminar, 64/Re')
    hi = (Re >= Re_star) & (Re <= _switch_re(mt.D_list_mm_steel[5]))   # ends at the last switch
    ax.plot(Re[hi], K_SHIFT * polyflo_f(Re[hi]), color=C_SHIFT, lw=2.8, zorder=5,
            label=f'Proposed, above Re*: Polyflo Ã— {K_SHIFT}')
    ax.plot(Re, polyflo_f(Re), color=C_POLY, lw=2.2, ls=LS_POLY, zorder=4,
            label='Polyflo, as published')
    for j, (nps, D) in enumerate(zip(mt.NPS_labels[:6], mt.D_list_mm_steel[:6])):
        f_c = churchill_f(Re, EPS_SHIFT, D)
        Rx = _switch_re(D)
        after = Re >= Rx
        # Faint curves from Re* on, in a flat light grey (no alpha), so the six curves
        # stay equally light where they overlap below the transition.
        if ref_eps is None:
            m = (Re >= Re_star) & ~after
            ax.plot(Re[m], f_c[m], color=FAINT_C, lw=1.0, ls=FAINT_LS[j], zorder=2)
        else:
            m = Re >= Re_star
            ax.plot(Re[m], churchill_f(Re[m], ref_eps, D), color=FAINT_C, lw=1.0,
                    ls=FAINT_LS[j], zorder=2)
        ax.plot(Re[after], f_c[after], color=colors[j], lw=3.0, zorder=6,
                label=f'Proposed, above switch: Churchill{tag} NPS {nps}')
        ax.plot(Rx, K_SHIFT * polyflo_f(Rx), 'o', ms=6, mfc='white', mec=colors[j],
                mew=1.6, zorder=7)
    faint = ('Churchill, where the shifted Polyflo governs' if ref_eps is None
             else f'Churchill reference, Îµ = {ref_eps} mm')
    for j, nps in enumerate(mt.NPS_labels[:6]):
        ax.plot([], [], color=FAINT_C, lw=1.0, ls=FAINT_LS[j],
                label=f'{faint}, NPS {nps}')
    ax.plot([], [], 'o', ms=6, mfc='white', mec='0.3', mew=1.6, ls='',
            label='Switch point (shifted Polyflo meets that size\'s curve)')
    ax.axvline(Re_star, color='0.35', ls=':', lw=1.2)
    ax.text(Re_star * 0.94, 0.13, f'Re* â‰ˆ {Re_star:,.0f}', fontsize=10, color='0.25',
            ha='right')
    ax.text(np.sqrt(2300 * 4000), 0.19, 'transition', rotation=90, ha='center', va='top',
            fontsize=9, color='0.4')
    ax.set(xscale='log', yscale='log', xlim=(1e2, 1e7), ylim=(0.01, 0.7))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    ax.set_xlabel('Reynolds number (dimensionless), increasing to the right', fontsize=11.5)
    ax.set_ylabel('Darcy friction factor (dimensionless), increasing upward', fontsize=11.5)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.95,
              title=f'Commercial steel, Îµ = {EPS_SHIFT} mm', title_fontsize=9)
    fig.tight_layout()
    _write(fig, out_name, dpi=200)


def fig_shifted_polyflo_vs_eps045():
    """The same proposed law against the Churchill reference for new commercial steel."""
    fig_shifted_polyflo(ref_eps=mt.EPS_MM, out_name='fig_sizing_law_vs_eps045.png')


# ------------------------------------------------------- 64/Re + published Polyflo, velocity-limited

V_SCREEN = 40.0                          # m/s, upper practitioner velocity screen (Â§3.4.3)
P_ABS_MAX = mt.Patm_kPa + 69.0           # kPa, domestic ceiling of the study


def re_at_velocity(gas, D_mm, v=V_SCREEN, p_abs_kpa=P_ABS_MAX):
    """Reynolds number of a gas flowing at v in a pipe of bore D_mm at p_abs_kpa, T = Tb."""
    rho = gas['S'] * p_abs_kpa * 1000 / (mt.R_air * mt.Tb)
    return rho * v * (D_mm / 1000) / gas['mu']


def fig_laminar_polyflo_rule(eps_mm=mt.EPS_MM, re_max=2e4,
                             out_name='fig_laminar_polyflo_rule.png',
                             mark_re=None):
    """f = max(64/Re, published Polyflo) against Churchill for NPS 1/2 to 2, with the shifted
    Polyflo proposal (x K_SHIFT from Re*) for comparison. Churchill curves are faint grey over
    the whole axis.
    re_max caps the Re axis (e.g. 2e4, the practical H2 range)."""
    Re = np.logspace(2, np.log10(re_max), 1000)
    Re_x = mt.Re_polyflo_laminar
    Re_star = (64 / (K_SHIFT * mt.a0)) ** (1 / (1 - mt.n))
    C_LAM = '#B2182B'
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    ax.axvspan(2300, 4000, color='0.92', zorder=0)
    lo = Re <= Re_x
    ax.plot(Re[lo], 64 / Re[lo], color=C_LAM, lw=2.8, zorder=5,
            label='Below Re = 1,361: laminar, 64/Re')
    hi = Re >= Re_x
    ax.plot(Re[hi], polyflo_f(Re[hi]), color=C_POLY, lw=2.2, ls=LS_POLY, zorder=6,
            label='Polyflo, as published (used above Re = 1,361)')
    hs = Re >= Re_star
    ax.plot(Re[hs], K_SHIFT * polyflo_f(Re[hs]), color=C_SHIFT, lw=2.2, zorder=5,
            label=f'Shifted Polyflo, Ã— {K_SHIFT} above Re* â‰ˆ {Re_star:,.0f}')
    curves = [64 / Re[lo], polyflo_f(Re[hi]), K_SHIFT * polyflo_f(Re[hs])]
    for j, (nps, D) in enumerate(zip(mt.NPS_labels[:6], mt.D_list_mm_steel[:6])):
        f_ch = churchill_f(Re, eps_mm, D)
        ax.plot(Re, f_ch, color=FAINT_C, lw=1.0, ls=FAINT_LS[j],
                zorder=2, label=f'Churchill Îµ = {eps_mm} mm, NPS {nps}')
        curves.append(f_ch)
    ax.plot(Re_x, 64 / Re_x, 'o', ms=9, mfc='white', mec=C_POLY, mew=1.8, zorder=7,
            label='Switch from 64/Re to Polyflo, Re = 1,361')
    ax.axvline(Re_x, color='0.35', ls=':', lw=1.2)
    ax.text(Re_x * 0.95, 0.13, 'Re = 1,361', fontsize=10, color='0.25', ha='right')
    ax.text(np.sqrt(2300 * 4000), 0.19, 'transition', rotation=90, ha='center', va='top',
            fontsize=9, color='0.4')
    if mark_re is not None:
        ax.plot(mark_re, 64 / mark_re, 'o', ms=9, mfc='white', mec=C_LAM, mew=1.8, zorder=7,
                label=f'Re = {mark_re:,.0f}')
        ax.axvline(mark_re, color='0.35', ls=':', lw=1.2)
        ax.text(mark_re * 0.95, 0.09, f'Re = {mark_re:,.0f}', fontsize=10, color='0.25',
                ha='right')
    f_lo = min(np.min(c) for c in curves)
    f_hi = max(np.max(c) for c in curves)
    nice = (0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)
    ylim = (max(t for t in nice if t <= f_lo * 0.95),
            min(t for t in nice if t >= f_hi * 1.05))
    ax.set(xscale='log', yscale='log', xlim=(1e2, Re[-1]), ylim=ylim)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    yticks = [t for t in nice if ylim[0] <= t <= ylim[1]]
    ax.set_yticks(yticks)
    ax.set_xlabel('Reynolds number (dimensionless), increasing to the right', fontsize=11.5)
    ax.set_ylabel('Darcy friction factor (dimensionless), increasing upward', fontsize=11.5)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, loc='lower left' if re_max else 'upper right', framealpha=0.95)
    fig.tight_layout()
    _write(fig, out_name, dpi=200)


# ------------------------------------------------------- Â§3.7: the two groups of inputs

def fig_input_groups():
    """Figure 17: the inputs of the sensitivity analysis, in two groups side by side."""
    W, H = 11.0, 5.4
    YM = 100 * H / W
    fig, ax = canvas(W, H, 100, YM)
    groups = [
        ((1.0, 48.5), BLUE_FILL, BLUE_EDGE, 'Group 1: changes the numbers in a table',
         ['Pipe roughness', 'Heating-value basis (HHV or LHV)', 'Gas properties (viscosity)',
          'Fitting allowance factor F'],
         'Scales the capacity of each cell.\nThe flow regime of the cell, and the\n'
         'friction law that applies, stay the same.'),
        ((51.5, 99.0), TAN_FILL, TAN_EDGE, 'Group 2: changes the flow regime of a cell',
         ['Pressure schedule', 'Run length', 'Pipe size'],
         'Sets the Reynolds number of each cell,\nand with it whether the flow is laminar,\n'
         'transitional or turbulent, and whether\nPolyflo can be relied on there.')]
    top, bot = YM - 2.0, 2.0
    for (x0, x1), fill, edge, head, items, note in groups:
        box(ax, x0, bot, x1, top, fill, edge, r=4.0, lw=2.4)
        xm = (x0 + x1) / 2
        label(ax, xm, top - 4.5, head, size=14.5, weight='bold', color=edge)
        for k, it in enumerate(items):
            label(ax, xm, top - 12.0 - 5.2 * k, it, size=15, color=INK)
        label(ax, xm, bot + 9.0, note, size=13.5, color=GREY, lsp=1.35, style='italic')
    _write(fig, 'fig_3_7_input_groups.png')


# ------------------------------------------------------- Â§3.7.3: Options A and B, separately

C_LAM = '#B2182B'
RE_H2_MAX = 2e5          # highest hydrogen Re, 30 ft of NPS 2 at the Table A.5 schedule (checks.py 12)


def _friction_axes(ax, curves, re_max):
    f_lo = min(np.min(c) for c in curves)
    f_hi = max(np.max(c) for c in curves)
    nice = (0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)
    ylim = (max(t for t in nice if t <= f_lo * 0.95), min(t for t in nice if t >= f_hi * 1.05))
    ax.set(xscale='log', yscale='log', xlim=(1e2, re_max), ylim=ylim)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([t for t in nice if ylim[0] <= t <= ylim[1]])
    ax.set_xlabel('Reynolds number (dimensionless), increasing to the right', fontsize=12.5)
    ax.set_ylabel('Darcy friction factor (dimensionless), increasing upward', fontsize=12.5)
    ax.tick_params(labelsize=11.5)
    ax.grid(True, which='both', alpha=0.25)
    ax.axvspan(2300, 4000, color='0.92', zorder=0)
    ax.text(np.sqrt(2300 * 4000), ylim[1] * 0.93, 'transition', rotation=90, ha='center',
            va='top', fontsize=10.5, color='0.4')


def fig_sizing_option_a(eps_mm=mt.EPS_MM, re_max=RE_H2_MAX, out_name='fig_sizing_option_a.png'):
    """Option A: 64/Re below Re = 1,361, published Polyflo above it, against Churchill at eps_mm
    for NPS 1/2 to 2. No shifted line, so Option B can be dropped without redrawing this one."""
    Re = np.logspace(2, np.log10(re_max), 1200)
    Re_x = mt.Re_polyflo_laminar
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    lo, hi = Re <= Re_x, Re >= Re_x
    ax.plot(Re[lo], 64 / Re[lo], color=C_LAM, lw=2.8, zorder=5, label='Below Re = 1,361: laminar, 64/Re')
    ax.plot(Re[hi], polyflo_f(Re[hi]), color=C_POLY, lw=2.4, ls=LS_POLY, zorder=6,
            label='Above Re = 1,361: Polyflo, as published')
    curves = [64 / Re[lo], polyflo_f(Re[hi])]
    for j, (nps, D) in enumerate(zip(mt.NPS_labels[:6], mt.D_list_mm_steel[:6])):
        f_ch = churchill_f(Re, eps_mm, D)
        ax.plot(Re, f_ch, color=FAINT_C, lw=1.1, ls=FAINT_LS[j], zorder=2,
                label=f'Churchill, Îµ = {eps_mm:g} mm, NPS {nps}')
        curves.append(f_ch)
    ax.plot(Re_x, 64 / Re_x, 'o', ms=9, mfc='white', mec=C_POLY, mew=1.8, zorder=7,
            label='Switch from 64/Re to Polyflo, Re = 1,361')
    ax.axvline(Re_x, color='0.35', ls=':', lw=1.2)
    _friction_axes(ax, curves, re_max)
    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.95)
    fig.tight_layout()
    _write(fig, out_name, dpi=200)


def fig_sizing_option_a_eps015():
    """Option A against a smoother pipe: the transition hump remains (checks.py check 13)."""
    fig_sizing_option_a(eps_mm=0.015, out_name='fig_sizing_option_a_eps015.png')


def fig_sizing_option_b(eps_mm=EPS_SHIFT, re_max=RE_H2_MAX, out_name='fig_sizing_option_b.png'):
    """Option B: f = max(64/Re, k*Polyflo), k = 1.18, against Churchill at eps_mm for NPS 1/2 to 2.
    Open circles: where a Churchill curve rises above the shifted line within the plotted range."""
    from scipy.optimize import brentq
    Re = np.logspace(2, np.log10(re_max), 1200)
    Re_star = (64 / (K_SHIFT * mt.a0)) ** (1 / (1 - mt.n))
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    lo, hi = Re <= Re_star, Re >= Re_star
    ax.plot(Re[lo], 64 / Re[lo], color=C_LAM, lw=2.8, zorder=5, label=f'Below Re* â‰ˆ {Re_star:,.0f}: laminar, 64/Re')
    ax.plot(Re[hi], K_SHIFT * polyflo_f(Re[hi]), color=C_SHIFT, lw=2.8, zorder=6,
            label=f'Above Re*: Polyflo Ã— {K_SHIFT}')
    hp = Re >= mt.Re_polyflo_laminar
    ax.plot(Re[hp], polyflo_f(Re[hp]), color=C_POLY, lw=1.8, ls=LS_POLY, zorder=4,
            label='Polyflo, as published (for comparison)')
    curves = [64 / Re[lo], K_SHIFT * polyflo_f(Re[hi]), polyflo_f(Re[hp])]
    for j, (nps, D) in enumerate(zip(mt.NPS_labels[:6], mt.D_list_mm_steel[:6])):
        f_ch = churchill_f(Re, eps_mm, D)
        ax.plot(Re, f_ch, color=FAINT_C, lw=1.1, ls=FAINT_LS[j], zorder=2,
                label=f'Churchill, Îµ = {eps_mm:g} mm, NPS {nps}')
        curves.append(f_ch)
        g = lambda R: churchill_f(R, eps_mm, D) - K_SHIFT * polyflo_f(R)
        if g(re_max) > 0:
            Rx = brentq(g, 2e4, re_max)
            ax.plot(Rx, K_SHIFT * polyflo_f(Rx), 'o', ms=8, mfc='white', mec='0.25', mew=1.6, zorder=7)
            ax.annotate(f'NPS {nps}\nRe â‰ˆ {round(Rx, -2):,.0f}', (Rx, K_SHIFT * polyflo_f(Rx)),
                        xytext=(0, 28 + 34 * (j % 2)), textcoords='offset points', ha='center', fontsize=9.5,
                        color='0.25', arrowprops=dict(arrowstyle='-', color='0.5', lw=0.8))
    ax.axvline(Re_star, color='0.35', ls=':', lw=1.2)
    _friction_axes(ax, curves, re_max)
    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.95)
    fig.tight_layout()
    _write(fig, out_name, dpi=200)


# ------------------------------------------------------- Â§3.8: CFD geometry and velocity field

CFD_MESH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cfd', 'new',
                                         'prod', '5000', 'EHD13_5000_arcarc_fine', 'constant', 'polyMesh'))
CFD_UMEAN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cfd', 'new',
                                          'prod', '5000', 'EHD13_5000_arcarc_transient', '0.13', 'UMean'))


def _foam_list(path, kind):
    """ASCII OpenFOAM list: 'label', 'vector' or 'face' (returns flat array, counts, starts)."""
    import re
    s = open(path, 'rb').read()
    m = re.compile(rb'\n(\d+)\s*\n\(').search(s, s.find(b'}'))
    n, start = int(m.group(1)), m.end()
    b = s[start:s.find(b'\n)', start)]
    if kind == 'label':
        return np.array(b.split(), dtype=np.int64)
    b = b.replace(b'(', b' ').replace(b')', b' ')
    if kind == 'vector':
        return np.array(b.split(), dtype=np.float64).reshape(n, 3)
    a = np.array(b.split(), dtype=np.int64)
    cnt, st, i, al = np.empty(n, np.int64), np.empty(n, np.int64), 0, a.tolist()
    for k in range(n):
        cnt[k], st[k] = al[i], i + 1
        i += al[i] + 1
    return a, cnt, st


def fig_cfd_geometry_velocity_ehd13():
    """(a) EHD 13 periodic unit cell (wall from the mesh wall patch); (b) mean axial velocity at
    Re = 5,000, natural gas, transient k-omega SST, on a longitudinal section (cells within 2 degrees
    of the section plane). The transient case runs on the arc-arc fine mesh, cell for cell."""
    import matplotlib.tri as mtri
    pts = _foam_list(os.path.join(CFD_MESH, 'points'), 'vector')
    own = _foam_list(os.path.join(CFD_MESH, 'owner'), 'label')
    nei = _foam_list(os.path.join(CFD_MESH, 'neighbour'), 'label')
    a, cnt, st = _foam_list(os.path.join(CFD_MESH, 'faces'), 'face')
    U = _foam_list(CFD_UMEAN, 'vector')
    fc = np.zeros((cnt.size, 3))
    for k in np.unique(cnt):
        m = cnt == k
        fc[m] = pts[a[st[m][:, None] + np.arange(k)[None, :]]].mean(1)
    nC = own.max() + 1
    cc, nf = np.zeros((nC, 3)), np.zeros(nC)
    np.add.at(cc, own, fc)
    np.add.at(nf, own, 1)
    np.add.at(cc, nei, fc[:nei.size])
    np.add.at(nf, nei, 1)
    cc /= nf[:, None]

    bnd = open(os.path.join(CFD_MESH, 'boundary')).read()
    import re
    w = re.search(r'walls\s*\{[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);', bnd)
    nw, sw = int(w.group(1)), int(w.group(2))
    xw, rw = fc[sw:sw + nw, 0] * 1e3, np.hypot(fc[sw:sw + nw, 1], fc[sw:sw + nw, 2]) * 1e3
    edges = np.linspace(0, xw.max(), 801)
    idx = np.digitize(xw, edges)
    prof = np.array([(xw[idx == i].mean(), rw[idx == i].max()) for i in range(1, 801) if (idx == i).any()])
    xm, rm = prof[:, 0], prof[:, 1]

    ang = np.degrees(np.arctan2(cc[:, 2], cc[:, 1]))
    sel = (np.abs(ang) < 2) | (np.abs(np.abs(ang) - 180) < 2)
    x = cc[sel, 0] * 1e3
    r = np.hypot(cc[sel, 1], cc[sel, 2]) * 1e3 * np.sign(cc[sel, 1])
    u = U[sel, 0]
    tri = mtri.Triangulation(x, r)
    xc, rc = x[tri.triangles].mean(1), np.abs(r[tri.triangles].mean(1))
    tri.set_mask(rc > np.interp(xc, xm, rm) * 0.999)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 9.4), gridspec_kw=dict(height_ratios=[1.05, 1]))
    d_eff = D_EFF_MM[13]
    a1.fill_between(xm, -rm, rm, color='#EAF1FB', zorder=1)
    for sgn in (1, -1):
        a1.plot(xm, sgn * rm, color='k', lw=1.8, zorder=3)
        a1.axhline(sgn * d_eff / 2, color=C_SHIFT, ls='--', lw=1.3, zorder=2)
    a1.axhline(0, color='0.5', lw=0.8, ls='-.')
    a1.annotate('', xy=(0, -7.6), xytext=(xm.max(), -7.6),
                arrowprops=dict(arrowstyle='<->', color='0.3', lw=1.0))
    a1.text(xm.max() / 2, -8.4, f'five corrugations, pitch {xm.max() / 5:.2f} mm, periodic at both ends',
            ha='center', va='top', fontsize=10.5, color='0.25')
    a1.text(xm.max() / 2, -2.2, f'root radius {rm.min():.2f} mm\ncrest radius {rm.max():.2f} mm\n'
            r'$D_\mathrm{eff}$' + f' = {d_eff:.2f} mm (dashed)', ha='center', va='top', fontsize=10.5,
            linespacing=1.4, bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.6', lw=0.8))
    a1.annotate('flow', xy=(8.5, 1.2), xytext=(3.5, 1.2), fontsize=11, va='center',
                arrowprops=dict(arrowstyle='-|>', color='0.2', lw=1.5))
    a1.set(aspect='equal', xlim=(-0.5, 20.6), ylim=(-10.0, 7.5))
    a1.set_title('(a) Geometry of the periodic unit cell', fontsize=12, loc='left')
    a1.set_ylabel('Radius (mm)', fontsize=11)

    lev = np.linspace(min(u.min(), -1.0), u.max(), 41)
    cf = a2.tricontourf(tri, u, levels=lev, cmap='viridis')
    a2.tricontour(tri, u, levels=[0.0], colors='white', linewidths=1.0)
    for sgn in (1, -1):
        a2.plot(xm, sgn * rm, color='k', lw=1.3)
    a2.set(aspect='equal', xlim=(-0.5, 20.6), ylim=(-7.2, 7.2))
    cb = fig.colorbar(cf, ax=a2, shrink=0.9, pad=0.01, ticks=np.arange(0, u.max() + 1, 2))
    cb.set_label('Mean axial velocity (m/s)', fontsize=11)
    a2.set_title('(b) Mean axial velocity, natural gas, Re = 5,000, transient k-Ï‰ SST;\n'
                 'white line: zero axial velocity', fontsize=11, loc='left')
    a2.set_xlabel('Axial position (mm)', fontsize=11)
    a2.set_ylabel('Radius (mm)', fontsize=11)
    fig.tight_layout()
    _write(fig, 'fig_cfd_geometry_velocity_EHD13.png', dpi=250)


# ------------------------------------------------------- pipe-flow model

# Fundamental steel pipe-flow model, GFE closed by Churchill. Four BC cases:
#   1 fix P1,P2 -> solve Q per L   2 fix P1,Q -> P2 profile
#   3 fix P2,Q -> P1 profile       4 fix P1,P2,Q,L -> back-calculated f
CASE = 2
F_SOURCE = 'churchill'    # 'churchill' (assumed eps) or 'backcalc' (cases 2,3)
PLOT_PROFILE = True
PLOT_MOODY = False

NPS = '1'
F_FITTING = 1.2           # 1.2 for A.1-A.4, 1.0 for A.5-A.7
P1_g_kPa, P2_g_kPa = 14.0, 7.0
Q_kW = 254.0
L_total_m = 100.0

GAS = dict(S=0.60, mu=1.2e-5, GAMMA=1.31, b=37.258945808)
D_BY_NPS = dict(zip(mt.NPS_labels, mt.D_list_mm_steel))


def _reynolds(Q_day, D_mm, S, mu):
    rho_std = S * (mt.Pb * 1000) / (mt.R_air * mt.Tb)
    return 4 * rho_std * (Q_day / 86400) / (np.pi * (D_mm / 1000) * mu)


def _local_velocity(Q_day, P_kPa, D_mm):
    A = (np.pi / 4) * (D_mm / 1000) ** 2
    return (Q_day / 86400) * (mt.Pb / P_kPa) / A


def _regime(Re):
    return np.where(Re < 2300, 'LAM', np.where(Re < 4000, 'TRANS', 'TURB'))


# f implied by one known full-pipe point (P1, P2, Q, L together), no eps needed.
def _backcalc_f(P1, P2, Q_day, L_total, D_mm, S):
    L_eff_km = F_FITTING * L_total / 1000
    return ((1.1494e-3 * (mt.Tb / mt.Pb)) ** 2 * (P1 ** 2 - P2 ** 2)
            / (S * mt.Tf * L_eff_km) * D_mm ** 5 / Q_day ** 2)


def _table(L_arr, P_abs, Q_day, Re, f, D_mm, gas):
    V = _local_velocity(Q_day, P_abs, D_mm)
    a = np.sqrt(gas['GAMMA'] * (mt.R_air / gas['S']) * mt.Tf)
    shape = L_arr.shape
    return pd.DataFrame({
        'L (m)': np.round(L_arr, 2),
        'P (kPa,g)': np.round(P_abs - mt.Patm_kPa, 3),
        'Q (kW)': np.round(np.broadcast_to((Q_day / 24) * gas['b'] * mt.z, shape)),
        'Q (m3/day)': np.round(np.broadcast_to(Q_day, shape), 3),
        'Re': np.round(np.broadcast_to(Re, shape)).astype(int),
        'f': np.round(np.broadcast_to(f, shape), 5),
        'V (m/s)': np.round(V, 3),
        'M': np.round(V / a, 4),
        'regime': _regime(np.broadcast_to(Re, shape)),
    })


def _plot_profile(L_arr, P_abs, Q_day, D_mm, gas, title):
    P_g = P_abs - mt.Patm_kPa
    V = _local_velocity(Q_day, P_abs, D_mm)
    M = V / np.sqrt(gas['GAMMA'] * (mt.R_air / gas['S']) * mt.Tf)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(L_arr, P_g, marker='o')
    axes[0].set_ylabel('P (kPa,g)')
    axes[0].set_title(title)
    axes[1].plot(L_arr, np.abs(np.gradient(P_g, L_arr)), marker='o', color='tab:red')
    axes[1].set_ylabel('|dP/dl| (kPa/m)')
    axes[2].plot(L_arr, M, marker='o', color='tab:green')
    axes[2].axhline(0.3, color='black', linestyle='--', linewidth=1,
                    label='M=0.3 (compressibility flag)')
    axes[2].set_ylabel('Mach number')
    axes[2].set_xlabel('distance (m)')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _write(fig, f'pipe_case{CASE}_profile.png', dpi=150)


def _plot_moody(Re_data, f_data, label_data, D_mm, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    Re_lam = np.linspace(100, 2300, 200)
    ax.plot(Re_lam, 64.0 / Re_lam, color='black', lw=1.5, label='Laminar (f=64/Re)')
    Re_turb = np.logspace(np.log10(2300), np.log10(2e5), 400)
    ax.plot(Re_turb, 0.316 / Re_turb ** 0.25, color='dimgray', lw=1.5, ls='--',
            label='Blasius smooth')
    ax.plot(Re_turb, churchill_f(Re_turb, 1e-6, D_mm), color='dimgray', lw=1.5, ls=':',
            label='Churchill smooth')
    ax.plot(Re_turb, churchill_f(Re_turb, mt.EPS_MM, D_mm), color='tab:blue', lw=1.5,
            ls='-.', label=f'Churchill eps={mt.EPS_MM} mm, NPS {NPS} '
                           f'(eps/D={mt.EPS_MM / D_mm:.4f})')
    ax.axvspan(2300, 4000, color='lightyellow', alpha=0.5)
    ax.scatter(Re_data, f_data, color='tab:red', s=45, edgecolors='white', zorder=5,
               label=label_data)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Reynolds number Re')
    ax.set_ylabel('Darcy friction factor f')
    ax.set_title(title)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _write(fig, f'pipe_case{CASE}_moody.png', dpi=150)


def pipe_model():
    D_mm = D_BY_NPS[NPS]
    S, mu, b = GAS['S'], GAS['mu'], GAS['b']
    P1 = P1_g_kPa + mt.Patm_kPa
    P2 = P2_g_kPa + mt.Patm_kPa
    L_m = mt.L_list_m
    L_eff_km = F_FITTING * (L_m / 1000)
    Q_day = Q_kW / (b * mt.z) * 24

    if CASE == 1:
        f = np.full(L_eff_km.shape, 0.02)
        for _ in range(300):
            Q = (1.1494e-3 * (mt.Tb / mt.Pb)
                 * np.sqrt((P1 ** 2 - P2 ** 2) / (S * mt.Tf * L_eff_km * f)) * D_mm ** 2.5)
            Re = _reynolds(Q, D_mm, S, mu)
            f = churchill_f(Re, mt.EPS_MM, D_mm)
        P_avg = (2 / 3) * (P1 + P2 - (P1 * P2) / (P1 + P2))
        df = _table(L_m, np.full(L_m.shape, P_avg), Q, Re, f, D_mm, GAS)
        print(f"Case 1 [NPS {NPS}]: P1={P1_g_kPa:g}, P2={P2_g_kPa:g} kPa(g) fixed, Q per L\n")
        print(df.to_string(index=False))
        if PLOT_MOODY:
            _plot_moody(df['Re'].values, df['f'].values, 'Q solved per L', D_mm,
                        f'Case 1 [NPS {NPS}]: Moody diagram')
        return

    if CASE == 4:
        Re = _reynolds(Q_day, D_mm, S, mu)
        f_pred = churchill_f(Re, mt.EPS_MM, D_mm)
        f_meas = _backcalc_f(P1, P2, Q_day, L_total_m, D_mm, S)
        print(f"Case 4 [NPS {NPS}]: L={L_total_m} m, P1={P1_g_kPa:g}, P2={P2_g_kPa:g} kPa(g), "
              f"Q={Q_kW:.1f} kW")
        print(f"Re = {Re:.0f}  ({_regime(Re)})")
        print(f"f_predicted (Churchill, eps={mt.EPS_MM} mm) = {f_pred:.5f}")
        print(f"f_back-calc (from given P1,P2,Q,L)          = {f_meas:.5f}")
        print(f"%diff = {100 * (f_meas - f_pred) / f_pred:.2f}%")
        if PLOT_MOODY:
            _plot_moody(np.array([Re, Re]), np.array([f_pred, f_meas]),
                        'predicted vs back-calc', D_mm, f'Case 4 [NPS {NPS}]: Moody diagram')
        return

    Re = _reynolds(Q_day, D_mm, S, mu)
    f = (_backcalc_f(P1, P2, Q_day, L_total_m, D_mm, S) if F_SOURCE == 'backcalc'
         else churchill_f(Re, mt.EPS_MM, D_mm))
    K = (Q_day / (1.1494e-3 * (mt.Tb / mt.Pb) * D_mm ** 2.5)) ** 2 * S * mt.Tf * f
    profile = (np.sqrt(P1 ** 2 - K * L_eff_km) if CASE == 2
               else np.sqrt(P2 ** 2 + K * L_eff_km))
    anchor = f"P1={P1_g_kPa:g} kPa(g)" if CASE == 2 else f"P2={P2_g_kPa:g} kPa(g)"
    where = "from inlet" if CASE == 2 else "backward from outlet"

    df = _table(L_m, profile, Q_day, Re, f, D_mm, GAS)
    print(f"Case {CASE} [NPS {NPS}, f from {F_SOURCE}]: {anchor}, Q={Q_kW:.1f} kW fixed, "
          f"profile {where} (Re={Re:.0f}, f={f:.5f})\n")
    print(df.to_string(index=False))
    if PLOT_PROFILE:
        _plot_profile(L_m, profile, Q_day, D_mm, GAS,
                      f'Case {CASE} [NPS {NPS}]: profile {where}, Q={Q_kW:.0f} kW')
    if PLOT_MOODY:
        _plot_moody(Re, f, f'this pipe (Q={Q_kW:.0f} kW)', D_mm,
                    f'Case {CASE} [NPS {NPS}]: Moody diagram')


# ------------------------------------------------------- 30 Sep 2026: new figure options (nothing above is replaced)

L_MIN_FT_M, L_MAX_FT_M = 9.0, 76.2        # 30 to 250 ft, the domestic run range used in §3.7


def _panel_ylim():
    return 0.015, 0.7


def _option_a_panel(ax, eps_mm, re_max=RE_H2_MAX, letter='', legend_handles=None):
    Re = np.logspace(2, np.log10(re_max), 1200)
    Re_x = mt.Re_polyflo_laminar
    lo, hi = Re <= Re_x, Re >= Re_x
    ax.axvspan(2300, 4000, color='0.92', zorder=0)
    ax.plot(Re[lo], 64 / Re[lo], color=C_LAM, lw=2.6, zorder=5)
    ax.plot(Re[hi], polyflo_f(Re[hi]), color=C_POLY, lw=2.2, ls=LS_POLY, zorder=6)
    for j, D in enumerate(mt.D_list_mm_steel[:6]):
        ax.plot(Re, churchill_f(Re, eps_mm, D), color=FAINT_C, lw=1.1, ls=FAINT_LS[j], zorder=2)
    ax.plot(Re_x, 64 / Re_x, 'o', ms=8, mfc='white', mec=C_POLY, mew=1.8, zorder=7)
    ax.axvline(Re_x, color='0.35', ls=':', lw=1.1)
    _panel_style(ax, eps_mm, letter, re_max)


def _option_b_panel(ax, eps_mm, re_max=RE_H2_MAX, letter=''):
    from scipy.optimize import brentq
    Re = np.logspace(2, np.log10(re_max), 1200)
    Re_star = (64 / (K_SHIFT * mt.a0)) ** (1 / (1 - mt.n))
    lo, hi = Re <= Re_star, Re >= Re_star
    ax.axvspan(2300, 4000, color='0.92', zorder=0)
    ax.plot(Re[lo], 64 / Re[lo], color=C_LAM, lw=2.6, zorder=5)
    ax.plot(Re[hi], K_SHIFT * polyflo_f(Re[hi]), color=C_SHIFT, lw=2.6, zorder=6)
    hp = Re >= mt.Re_polyflo_laminar
    ax.plot(Re[hp], polyflo_f(Re[hp]), color=C_POLY, lw=1.6, ls=LS_POLY, zorder=4)
    for j, (nps, D) in enumerate(zip(mt.NPS_labels[:6], mt.D_list_mm_steel[:6])):
        ax.plot(Re, churchill_f(Re, eps_mm, D), color=FAINT_C, lw=1.1, ls=FAINT_LS[j], zorder=2)
        g = lambda R: churchill_f(R, eps_mm, D) - K_SHIFT * polyflo_f(R)
        if g(re_max) > 0:
            Rx = brentq(g, 2e4, re_max)
            ax.plot(Rx, K_SHIFT * polyflo_f(Rx), 'o', ms=8, mfc='white', mec='0.25', mew=1.6, zorder=8)
            ax.annotate(f'NPS {nps}: Re ≈ {round(Rx, -2):,.0f}', (Rx, K_SHIFT * polyflo_f(Rx)),
                        xytext=(-10, 26 + 26 * (j % 2)), textcoords='offset points', ha='right',
                        fontsize=9, color='0.25', arrowprops=dict(arrowstyle='-', color='0.5', lw=0.8))
    ax.axvline(Re_star, color='0.35', ls=':', lw=1.1)
    _panel_style(ax, eps_mm, letter, re_max)


def _panel_style(ax, eps_mm, letter, re_max):
    ylim = _panel_ylim()
    ax.set(xscale='log', yscale='log', xlim=(1e2, re_max), ylim=ylim)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7])
    ax.set_xlabel('Reynolds number (dimensionless)', fontsize=11.5)
    ax.set_ylabel('Darcy friction factor (dimensionless)', fontsize=11.5)
    ax.tick_params(labelsize=10.5)
    ax.grid(True, which='both', alpha=0.25)
    ax.set_title(f'({letter}) Churchill reference at ε = {eps_mm:g} mm', fontsize=12.5, loc='left')


def _three_panel(option, out_name):
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(13.5, 11.2))
    gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.75)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]
    for ax, eps, letter in zip(axes, (0.045, 0.015, 0.09), 'abc'):
        (_option_a_panel if option == 'A' else _option_b_panel)(ax, eps, letter=letter)
    Re_star = (64 / (K_SHIFT * mt.a0)) ** (1 / (1 - mt.n))
    lax = fig.add_subplot(gs[1, 0]); lax.axis('off')
    rax = fig.add_subplot(gs[1, 3]); rax.axis('off')
    if option == 'A':
        left = [Line2D([0], [0], color=C_LAM, lw=2.6, label='Below Re = 1,361:\nlaminar, 64/Re'),
                Line2D([0], [0], color=C_POLY, lw=2.2, ls=LS_POLY, label='Above Re = 1,361:\nPolyflo, as published'),
                Line2D([0], [0], color=C_POLY, marker='o', mfc='white', ls='', ms=8,
                       label='Switch, Re = 1,361')]
    else:
        left = [Line2D([0], [0], color=C_LAM, lw=2.6, label=f'Below Re* ≈ {Re_star:,.0f}:\nlaminar, 64/Re'),
                Line2D([0], [0], color=C_SHIFT, lw=2.6, label=f'Above Re*:\nPolyflo × {K_SHIFT}'),
                Line2D([0], [0], color=C_POLY, lw=1.6, ls=LS_POLY, label='Polyflo, as published\n(for comparison)'),
                Line2D([0], [0], color='0.25', marker='o', mfc='white', ls='', ms=8,
                       label='Churchill curve rises above\nthe shifted line')]
    lax.legend(handles=left, loc='center', fontsize=10.5, frameon=True, title='Sizing law',
               title_fontsize=11.5, labelspacing=1.1)
    right = [Line2D([0], [0], color=FAINT_C, lw=1.1, ls=FAINT_LS[j], label=f'NPS {n}')
             for j, n in enumerate(mt.NPS_labels[:6])]
    rax.legend(handles=right, loc='center', fontsize=10.5, frameon=True, title='Churchill curves\n(gray, one per size)',
               title_fontsize=11.5, labelspacing=0.9, handlelength=3.2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07)
    _write(fig, out_name, dpi=200)


def fig_sizing_option_a_3panel():
    """Option A against Churchill at eps = 0.045 (a), 0.015 (b) and 0.09 mm (c), NPS 1/2 to 2."""
    _three_panel('A', 'fig_sizing_option_a_3panel.png')


def fig_sizing_option_b_3panel():
    """Option B against Churchill at eps = 0.045 (a), 0.015 (b) and 0.09 mm (c), NPS 1/2 to 2."""
    _three_panel('B', 'fig_sizing_option_b_3panel.png')


def fig_colebrook_vs_churchill_transition():
    """Churchill and Colebrook-White between Re = 1,000 and 12,000 for three steel sizes at 0.045 mm.
    Colebrook-White is a turbulent correlation, drawn here from Re = 2,300 only."""
    from matplotlib.lines import Line2D
    sizes = [(0, '½'), (2, '1'), (5, '2')]
    cols = plt.cm.viridis(np.linspace(0.05, 0.75, 3))
    mk = ['o', 's', '^']
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    Re = np.logspace(3, np.log10(1.2e4), 500)
    ax.axvspan(2300, 4000, color='0.92', zorder=0)
    Rl = Re[Re <= 2300]
    ax.plot(Rl, 64 / Rl, color=C_LAM, lw=2.0, ls=':', zorder=3)
    for (j, nps), c, k in zip(sizes, cols, mk):
        D = mt.D_list_mm_steel[j]
        ax.plot(Re, churchill_f(Re, mt.EPS_MM, D), color=c, lw=2.4, zorder=4)
        Rt = Re[Re >= 2300]
        ax.plot(Rt, colebrook_f(Rt, mt.EPS_MM, D), color=c, lw=2.0, ls='--', zorder=3)
        ax.plot([2300, 2300], [churchill_f(2300, mt.EPS_MM, D), colebrook_f(2300, mt.EPS_MM, D)],
                ls='', marker=k, ms=7, mfc='white', mec=c, mew=1.6, zorder=6)
    ax.set(xscale='log', xlim=(1e3, 1.2e4), ylim=(0.02, 0.066))
    ax.set_xticks([1000, 2000, 3000, 4000, 6000, 10000])
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Reynolds number (dimensionless)', fontsize=12)
    ax.set_ylabel('Darcy friction factor (dimensionless)', fontsize=12)
    ax.grid(True, which='both', alpha=0.25)
    ax.text(np.sqrt(2300 * 4000), 0.0652, 'transition band, Re = 2,300 to 4,000', ha='center', va='top',
            fontsize=10.5, color='0.35')
    handles = [Line2D([0], [0], color='0.2', lw=2.4, label='Churchill (solid)'),
               Line2D([0], [0], color='0.2', lw=2.0, ls='--', label='Colebrook-White, drawn from Re = 2,300 (dashed)'),
               Line2D([0], [0], color=C_LAM, lw=2.0, ls=':', label='Laminar, 64/Re, below Re = 2,300'),
               ] + [Line2D([0], [0], color=c, marker=k, ls='', ms=7, mfc='white', mew=1.6,
                           label=f'NPS {nps} (marker: value at Re = 2,300)')
                    for (j, nps), c, k in zip(sizes, cols, mk)]
    ax.legend(handles=handles, loc='upper right', fontsize=9.5, framealpha=0.95,
              title='Commercial steel, ε = 0.045 mm', title_fontsize=10.5, bbox_to_anchor=(1.0, 0.96))
    fig.tight_layout()
    _write(fig, 'fig_colebrook_vs_churchill_transition.png', dpi=200)


def _dev_curve_polyflo(eps_mm, D_mm, Re):
    """Capacity of Polyflo as published against GFE-Churchill at fixed pressure drop, in %. Q ratio at
    fixed dP is Re_polyflo / Re_churchill, where Re^2 * f is the same for both."""
    from scipy.optimize import brentq
    out = []
    for r in Re:
        C = r ** 2 * polyflo_f(r)
        rr = brentq(lambda R: R ** 2 * churchill_f(R, eps_mm, D_mm) - C, 1e-3, 1e10)
        out.append((r / rr - 1) * 100)
    return np.array(out)


def fig_polyflo_deviation_bands():
    """Where Polyflo departs from GFE-Churchill (positive = more capacity than Moody friction), as four
    Reynolds-number bands, with the Re range each schedule covers for natural gas and hydrogen (30 to
    250 ft runs, NPS 1/2 to 2) underneath."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    Re = np.logspace(np.log10(150), np.log10(2e5), 260)
    sizes = [(0, '½'), (2, '1'), (5, '2')]
    cols = plt.cm.viridis(np.linspace(0.05, 0.75, 3))
    edges = [150, 1361, 2800, 5000, 2e5]
    band_fill = ['#F6D3D3', '#DDEBD7', '#FBE3BE', '#DDEBD7']
    band_txt = ['Band 1\nmore capacity than\nMoody friction',
                'Band 2\nconservative',
                'Band 3\nhump,\nup to +5.8%',
                'Band 4\nconservative for the larger sizes; smaller sizes give\nmore capacity again, up to +10% (NPS ½) at Re = 2×10⁵']
    fig = plt.figure(figsize=(12.5, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.6], hspace=0.10)
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax)
    for k in range(4):
        for a in (ax, ax2):
            a.axvspan(edges[k], edges[k + 1], color=band_fill[k], alpha=0.75, zorder=0, lw=0)
    ax.axhline(0, color='0.25', lw=1.1, zorder=2)
    for (j, nps), c, mkr in zip(sizes, cols, ['o', 's', '^']):
        D = mt.D_list_mm_steel[j]
        d = _dev_curve_polyflo(mt.EPS_MM, D, Re)
        ax.plot(Re, d, color=c, lw=2.3, zorder=4)
        ax.plot(Re[::28], d[::28], ls='', marker=mkr, ms=6, mfc='white', mec=c, mew=1.4, zorder=5)
    ax.set_yscale('symlog', linthresh=10, linscale=1.4)
    ax.set_ylim(-40, 400)
    ax.set_yticks([-30, -10, 0, 10, 20, 50, 100, 200, 400])
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:+.0f}' if v else '0'))
    ax.set_ylabel('Capacity of Polyflo relative to Moody friction (%)\n(positive: Polyflo gives more capacity)', fontsize=11.5)
    ax.set_xscale('log'); ax.set_xlim(150, 2e5)
    xm = [np.sqrt(edges[k] * edges[k + 1]) for k in range(4)]
    for k, (x, t) in enumerate(zip(xm, band_txt)):
        ax.text(x, 340 if k % 2 == 0 else 150, t, ha='center', va='top', fontsize=9.5, color='0.15', linespacing=1.25)
    for e in edges[1:-1]:
        ax.axvline(e, color='0.4', ls=':', lw=1.1)
    ax.tick_params(labelbottom=False, labelsize=10.5)
    ax.grid(True, which='both', alpha=0.22)
    # spans of operation
    ys = []
    lab = []
    for t_i, t in enumerate((1, 2, 3, 4, 5)):
        for g_i, gname in enumerate(('H2', 'NG')):
            r = mt.build_all(mt.A_TABLES[t], mt.GASES[gname])
            keep = (mt.L_list_m >= L_MIN_FT_M) & (mt.L_list_m <= L_MAX_FT_M)
            R = r['Re_gfe'][keep][:, :6]
            y = -(t_i * 3 + g_i)
            ax2.barh(y, R.max() - R.min(), left=R.min(), height=0.82, color=GAS_COLORS[gname],
                     hatch=GAS_HATCH[gname], edgecolor='white' if not GAS_HATCH[gname] else '#7a4718',
                     lw=0.6, zorder=3)
            ys.append(y); lab.append(f'A.{t}  {"hydrogen" if gname == "H2" else "natural gas"}')
    ax2.set_yticks(ys); ax2.set_yticklabels(lab, fontsize=9.5)
    ax2.set_xscale('log'); ax2.set_xlim(150, 2e5)
    ax2.set_xlabel('Reynolds number (dimensionless)', fontsize=12)
    ax2.tick_params(labelsize=10.5)
    ax2.grid(True, which='both', axis='x', alpha=0.22)
    handles = [Line2D([0], [0], color=c, lw=2.3, marker=mkr, mfc='white', ms=6, label=f'NPS {n}, ε = 0.045 mm')
               for (j, n), c, mkr in zip(sizes, cols, ['o', 's', '^'])]
    handles += [Patch(facecolor=GAS_COLORS['H2'], hatch=GAS_HATCH['H2'], edgecolor='#7a4718',
                      label='Hydrogen cells, 30 to 250 ft, NPS ½ to 2'),
                Patch(facecolor=GAS_COLORS['NG'], edgecolor='white', label='Natural gas cells, same range')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.14)
    _write(fig, 'fig_polyflo_deviation_bands.png', dpi=200)


def fig_input_groups_compact():
    """Figure 17, smaller boxes and larger type: the same two groups of inputs."""
    W, H = 12.0, 4.7
    YM = 100 * H / W
    fig, ax = canvas(W, H, 100, YM)
    groups = [
        ((1.0, 49.0), BLUE_FILL, BLUE_EDGE, 'Group 1: changes the numbers in a table',
         ['Pipe roughness', 'Heating-value basis (HHV or LHV)', 'Uncertainty in hydrogen viscosity',
          'Fitting allowance factor F'],
         'Scales the capacity of each cell. The flow regime\nand friction law of the cell stay the same.'),
        ((51.0, 99.0), TAN_FILL, TAN_EDGE, 'Group 2: changes the flow regime of a cell',
         ['Pressure schedule', 'Run length', 'Pipe size'],
         'Sets the Reynolds number of each cell, and with it\nwhether flow is laminar, transitional or turbulent.')]
    top, bot = YM - 1.5, 1.5
    for (x0, x1), fill, edge, head, items, note in groups:
        box(ax, x0, bot, x1, top, fill, edge, r=3.0, lw=2.4)
        xm = (x0 + x1) / 2
        label(ax, xm, top - 4.0, head, size=14, weight='bold', color=edge)
        for k, it in enumerate(items):
            label(ax, xm, top - 10.5 - 4.2 * k, it, size=15, color=INK)
        label(ax, xm, bot + 6.0, note, size=13, color=GREY, lsp=1.3, style='italic')
    _write(fig, 'fig_3_7_input_groups_compact.png')


FIGURES = {
    'three_ratios': fig_three_ratios,
    'energy_chain': fig_energy_chain,
    'capacity_ratio': fig_capacity_ratio,
    'capacity_ratio_accessible': fig_capacity_ratio_accessible,
    'reynolds_spans': fig_reynolds_spans,
    'reynolds_spans_accessible': fig_reynolds_spans_accessible,
    'fig14_csd': fig14_csd,
    'fig14_csd_2x2': fig14_csd_2x2,
    'fig14_csd_all_ehd': fig14_csd_all_ehd,
    'fig15_capacity_vs_ehd': fig15_capacity_vs_ehd,
    'fig15_capacity_vs_ehd_c_only': fig15_capacity_vs_ehd_c_only,
    'fig16_h2_ratio_vs_length': fig16_h2_ratio_vs_length,
    'fig10_csst_friction': fig10_csst_friction_measured,
    'cfd_f_vs_re': fig_cfd_f_vs_re_ehd13,
    'fig1_documents': fig1_governing_documents,
    'fig2_reynolds_friction': fig2_reynolds_friction,
    'fig4_gas_correction': fig4_gas_correction_routes,
    'fig6_noise': fig6_noise_vs_supply_pressure,
    'fig8_friction_factor': fig8_friction_factor_steel,
    'fig12_pooled_diff': fig12_pooled_diff_vs_re,
    'fig13_nonconservatism': fig13_nonconservatism_vs_agreement_share,
    'figB1_rescale': figB1_diameter_rescale_demo,
    'figB2_lc1': figB2_lc1_friction_backcalc,
    'shifted_polyflo': fig_shifted_polyflo,
    'shifted_polyflo_vs_eps045': fig_shifted_polyflo_vs_eps045,
    'laminar_polyflo_rule_re2e4': lambda: fig_laminar_polyflo_rule(
        re_max=2e4, out_name='fig_laminar_polyflo_rule_re2e4.png'),
    'laminar_polyflo_rule_re2e4_re1120': lambda: fig_laminar_polyflo_rule(
        re_max=2e4, out_name='fig_laminar_polyflo_rule_re2e4_re1120.png', mark_re=1120),
    'input_groups': fig_input_groups,
    'sizing_option_a': fig_sizing_option_a,
    'sizing_option_a_eps015': fig_sizing_option_a_eps015,
    'sizing_option_b': fig_sizing_option_b,
    'cfd_geometry_velocity': fig_cfd_geometry_velocity_ehd13,
    'pipe': pipe_model,
    'sizing_option_a_3panel': fig_sizing_option_a_3panel,
    'sizing_option_b_3panel': fig_sizing_option_b_3panel,
    'colebrook_vs_churchill_transition': fig_colebrook_vs_churchill_transition,
    'polyflo_deviation_bands': fig_polyflo_deviation_bands,
    'input_groups_compact': fig_input_groups_compact,
}


if __name__ == '__main__':
    matplotlib.use('Agg')
    pd.set_option('display.width', 160)
    names = [a for a in sys.argv[1:] if a in FIGURES] or list(FIGURES)
    for name in names:
        FIGURES[name]()
