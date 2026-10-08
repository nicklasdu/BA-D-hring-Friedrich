"""
Pyomo Heat Pump Optimisation - Sensitivitaetsanalyse ueber 8 EFH-Varianten

Fuehrt die Volljahres-Optimierung fuer alle acht Gebaeude aus der
Sensitivitaets-Tabelle durch (efh1-efh8), mit je eigener Flaeche,
spezifischer Waermelast, Auslegungslast bei -12 C, installierter WP-Leistung
und Vorlauftemperatur:

  Gebaeude  Flaeche  Spez.Waermelast  Last@-12C  Installierte WP  Vorlauf
  efh1      150 m2   40 W/m2          6 kW       5 kW             35 C
  efh2      150 m2   40 W/m2          6 kW       5 kW             55 C
  efh3      150 m2   40 W/m2          6 kW       7 kW             35 C
  efh4      150 m2   40 W/m2          6 kW       7 kW             55 C
  efh5      150 m2   60 W/m2          9 kW       8 kW             35 C
  efh6      150 m2   60 W/m2          9 kW       8 kW             55 C
  efh7      150 m2   60 W/m2          9 kW      10 kW             35 C
  efh8      150 m2   60 W/m2          9 kW      10 kW             55 C

Modellierungsentscheidungen (siehe Chat-Verlauf):
- Waermebedarf je Gebaeude wird NICHT aus der vorhandenen heating_output_w-
  Spalte skaliert (die gehoert zu einem einzelnen Referenzhaus), sondern ueber
  eine vereinfachte lineare Heizkurve aus der Aussentemperatur UND der
  gebaeudeeigenen Auslegungslast bei -12 C erzeugt (space_demand_kw()).
  Heizgrenztemperatur HEIZGRENZE_C=15 C ist eine Standardannahme und sollte im
  Methodikkapitel explizit benannt werden.
- COP haengt ueber Carnot + zurueckgerechneten Guetegrad vom Vorlauf ab
  (get_cop(), siehe vorherige Herleitung: COP_CURVE gilt fuer VL_BASE=35 C).
- Tankgroesse TANK_MAX_KWH ist fuer alle acht Gebaeude gleich gehalten (nicht
  Teil der Tabelle); falls gewuenscht laesst sich das leicht in BUILDINGS
  ergaenzen und in solve_building() verwenden.
- model.deficit[t]: Schlupfvariable fuer nicht gedeckten Bedarf (kWh), stark
  bestraft in der Zielfunktion, haelt das Modell bei unterdimensionierten
  Kombinationen loesbar statt infeasible.
"""

import os
import csv
import json
import pyomo.environ as pyo
from pyomo_windows.solvers import SolverManager
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime

# Arbeitsverzeichnis auf den Ordner dieser Datei setzen, damit Input-CSVs,
# PNGs und sensitivity_results.json IMMER im selben Ordner wie das Skript
# landen, unabhaengig davon, mit welchem Working Directory die IDE startet.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE     = "hamburg_heating_electricity_2025.csv"
WARMSTART_FILE = "simulation_results.csv"

MOD_MIN        = 0.30
TANK_START_PCT = 0.50
TANK_LOSS_PCT  = 0.01
MIN_RUNTIME_H  = 2
TANK_MAX_KWH   = 40.6       # 1000L - fuer alle Gebaeude gleich, siehe Hinweis oben
HEIZGRENZE_C   = 15.0        # Heizgrenztemperatur der vereinfachten Heizkurve
NORM_TEMP_C    = -12.0       # Bezugspunkt der "Last bei -12 Grad"

VL_BASE       = 35           # C, Vorlauf fuer den COP_CURVE unten kalibriert ist
COND_APPROACH = 5            # K, Kondensator-Graedigkeit
EVAP_APPROACH = 5            # K, Verdampfer-Graedigkeit (Luft-WP)

DEFICIT_PENALTY_EUR_PER_KWH = 1000.0

SOLVER_TIME_LIMIT_S = 3600    # je Gebaeude; in Tests reichte CBC ~15-50s bis "optimal"
MIP_GAP              = 0.02

BUILDINGS = [
    {"name": "efh1", "flaeche": 150, "spez_last": 40, "last_minus12": 6, "hp_kw": 5,  "vorlauf": 35},
    {"name": "efh2", "flaeche": 150, "spez_last": 40, "last_minus12": 6, "hp_kw": 5,  "vorlauf": 55},
    {"name": "efh3", "flaeche": 150, "spez_last": 40, "last_minus12": 6, "hp_kw": 7,  "vorlauf": 35},
    {"name": "efh4", "flaeche": 150, "spez_last": 40, "last_minus12": 6, "hp_kw": 7,  "vorlauf": 55},
    {"name": "efh5", "flaeche": 150, "spez_last": 60, "last_minus12": 9, "hp_kw": 8,  "vorlauf": 35},
    {"name": "efh6", "flaeche": 150, "spez_last": 60, "last_minus12": 9, "hp_kw": 8,  "vorlauf": 55},
    {"name": "efh7", "flaeche": 150, "spez_last": 60, "last_minus12": 9, "hp_kw": 10, "vorlauf": 35},
    {"name": "efh8", "flaeche": 150, "spez_last": 60, "last_minus12": 9, "hp_kw": 10, "vorlauf": 55},
]

DHW_SHOWER_PER_PEAK = (2500 * 0.50) / 365 / 2
DHW_BASE_KW         = (2500 * 0.50) / (365 * 16)

def dhw_demand_kw(hour):
    d = 0.0
    if hour in (7, 19): d += DHW_SHOWER_PER_PEAK
    if 6 <= hour <= 21: d += DHW_BASE_KW
    return d

def space_demand_kw(t_amb, last_minus12_kw):
    """Vereinfachte lineare Heizkurve: 0 kW an der Heizgrenze, last_minus12_kw bei -12 C."""
    return last_minus12_kw * max(0.0, (HEIZGRENZE_C - t_amb) / (HEIZGRENZE_C - NORM_TEMP_C))

# COP_CURVE gilt fuer VL_BASE=35 C (siehe Herleitung im Chat: der daraus
# zurueckgerechnete Guetegrad liegt konstant im plausiblen Bereich 0.35-0.50).
COP_CURVE = [(-15,1.8),(-10,2.2),(-7,2.5),(-5,2.7),(0,3.1),(2,3.3),
             (5,3.7),(7,4.0),(10,4.4),(12,4.7),(15,5.0),(20,5.5)]

def _cop_base_interp(temp_c):
    if temp_c <= COP_CURVE[0][0]:  return COP_CURVE[0][1]
    if temp_c >= COP_CURVE[-1][0]: return COP_CURVE[-1][1]
    for i in range(len(COP_CURVE)-1):
        t0,c0 = COP_CURVE[i]; t1,c1 = COP_CURVE[i+1]
        if t0 <= temp_c <= t1:
            return c0 + (temp_c-t0)/(t1-t0)*(c1-c0)
    return 3.0

def carnot_cop(t_amb_c, vl_c):
    t_cond = vl_c + COND_APPROACH + 273.15
    t_evap = t_amb_c - EVAP_APPROACH + 273.15
    return t_cond / (t_cond - t_evap)

def get_cop(temp_c, vl_c=VL_BASE):
    """COP bei Aussentemperatur temp_c und Vorlauf vl_c, siehe Herleitung im Chat
    (Carnot-COP skaliert mit dem aus COP_CURVE zurueckgerechneten Guetegrad;
    vgl. VDI 4650 Blatt 1 sowie Ruhnau/Hirth/Praktiknjo 2019, Scientific Data 6, 189)."""
    guetegrad = _cop_base_interp(temp_c) / carnot_cop(temp_c, VL_BASE)
    return guetegrad * carnot_cop(temp_c, vl_c)

def compute_tank(u_schedule, demands, hp_kw, tank_start, tank_max, loss_pct):
    tanks = []
    tank  = tank_start
    for t in range(len(u_schedule)):
        tank = tank*(1-loss_pct) - demands[t] + hp_kw*u_schedule[t]
        tank = max(0.0, min(tank, tank_max))
        tanks.append(tank)
    return tanks

def solve_building(b, demands, prices, elec_per_h, ws_x, n_hours):
    hp_kw = b["hp_kw"]
    label = f"{b['name']} ({hp_kw:.0f}kW, VL{b['vorlauf']})"
    T          = n_hours
    tank_start = TANK_MAX_KWH * TANK_START_PCT
    ws_u       = ws_x[:]
    ws_tank    = compute_tank(ws_u, demands, hp_kw, tank_start, TANK_MAX_KWH, TANK_LOSS_PCT)

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T-1)
    model.demand   = pyo.Param(model.T, initialize=dict(enumerate(demands)))
    model.price    = pyo.Param(model.T, initialize=dict(enumerate(prices)))
    model.elec_kwh = pyo.Param(model.T, initialize=dict(enumerate(elec_per_h)))

    model.x = pyo.Var(model.T, domain=pyo.Binary)
    model.u = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, TANK_MAX_KWH))
    model.deficit = pyo.Var(model.T, domain=pyo.NonNegativeReals)   # ungedeckter Bedarf [kWh]

    for t in range(T):
        model.x[t].set_value(ws_x[t])
        model.u[t].set_value(ws_u[t])
        model.s[t].set_value(max(0.0, ws_tank[t]))
        model.deficit[t].set_value(0.0)

    model.obj = pyo.Objective(
        expr=sum(model.u[t]*model.elec_kwh[t]*model.price[t]/1000 for t in model.T)
             + DEFICIT_PENALTY_EUR_PER_KWH * sum(model.deficit[t] for t in model.T),
        sense=pyo.minimize)

    def tank_balance_rule(m, t):
        prev = tank_start if t == 0 else m.s[t-1]
        return m.s[t] == prev*(1-TANK_LOSS_PCT) - m.demand[t] + hp_kw*m.u[t] + m.deficit[t]
    model.tank_balance = pyo.Constraint(model.T, rule=tank_balance_rule)

    model.mod_upper = pyo.Constraint(model.T, rule=lambda m, t: m.u[t] <= m.x[t])
    model.mod_lower = pyo.Constraint(model.T, rule=lambda m, t: m.u[t] >= MOD_MIN * m.x[t])

    def min_on_rule(m, t):
        if t < 1 or t + MIN_RUNTIME_H > T:
            return pyo.Constraint.Skip
        return sum(m.x[t+k] for k in range(MIN_RUNTIME_H)) >= \
               MIN_RUNTIME_H*(m.x[t]-m.x[t-1])
    model.min_on = pyo.Constraint(model.T, rule=min_on_rule)

    solver_manager = SolverManager()
    solver = solver_manager.get_solver("cbc")
    solver.options["seconds"] = SOLVER_TIME_LIMIT_S
    solver.options["ratio"]   = MIP_GAP
    solver.options["heurist"] = "on"

    print(f"\n  Solving {label} (bis zu {SOLVER_TIME_LIMIT_S/60:.0f} min, "
          f"MIP-Gap {MIP_GAP*100:.0f}%, {T} Stunden)...")
    result = solver.solve(model, tee=False, warmstart=True)
    status = str(result.solver.termination_condition)

    try:
        raw_x = [pyo.value(model.x[t]) for t in model.T]
        raw_u = [pyo.value(model.u[t]) for t in model.T]
        raw_deficit = [pyo.value(model.deficit[t]) for t in model.T]
        x_vals = [round(v) for v in raw_x] if all(abs(v-round(v)) < 0.01 for v in raw_x) else ws_x[:]
        u_vals = [min(1.0, max(0.0, v)) for v in raw_u]
        deficit_vals = [max(0.0, v) for v in raw_deficit]
    except Exception:
        x_vals, u_vals, deficit_vals = ws_x[:], ws_u[:], [0.0]*T
        status += " (extraction failed)"

    s_vals = compute_tank(u_vals, demands, hp_kw, tank_start, TANK_MAX_KWH, TANK_LOSS_PCT)
    total_cost = sum(u_vals[t]*elec_per_h[t]*prices[t]/1000 for t in range(T))
    total_elec = sum(u_vals[t]*elec_per_h[t] for t in range(T))
    hp_hours   = sum(x_vals)
    avg_paid   = total_cost/total_elec*1000 if total_elec > 0 else 0
    cycles     = sum(1 for t in range(1,T) if x_vals[t]==1 and x_vals[t-1]==0)
    total_deficit = sum(deficit_vals)
    deficit_hours = sum(1 for v in deficit_vals if v > 1e-4)

    print(f"  Status: {status}")
    print(f"  HP hours: {hp_hours}, Cost: {total_cost:.2f} EUR, Avg price: {avg_paid:.1f} EUR/MWh, "
          f"Cycles: {cycles}, Defizit: {total_deficit:.2f} kWh in {deficit_hours} Std.")

    return {
        "name": b["name"], "hp_kw": hp_kw, "vorlauf": b["vorlauf"], "last_minus12": b["last_minus12"],
        "u_vals": u_vals, "s_vals": s_vals, "deficit_vals": deficit_vals,
        "hp_hours": hp_hours, "total_elec": total_elec, "total_cost": total_cost,
        "avg_paid": avg_paid, "cycles": cycles, "total_deficit": total_deficit,
        "deficit_hours": deficit_hours, "status": status,
    }

# ================================================================
# LOAD DATA
# ================================================================
print("Loading data...")
with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))
T = len(all_rows)

valid_prices = [float(r["electricity_price_eur_per_mwh"])
                for r in all_rows if r["electricity_price_eur_per_mwh"] != ""]
avg_market_price = sum(valid_prices)/len(valid_prices)

datetimes, temps, prices, hours = [], [], [], []
for r in all_rows:
    h   = int(r["datetime"][11:13])
    tmp = float(r["temperature_outdoor_c"])
    p   = float(r["electricity_price_eur_per_mwh"]) if r["electricity_price_eur_per_mwh"] != "" else avg_market_price
    datetimes.append(r["datetime"]); temps.append(tmp); prices.append(p); hours.append(h)

with open(WARMSTART_FILE, newline="", encoding="utf-8") as f:
    ws_rows_all = list(csv.DictReader(f))
ws_x_raw = [int(row["hp_running"]) for row in ws_rows_all]
if len(ws_x_raw) < T:
    reps = (T // len(ws_x_raw)) + 1
    ws_x_raw = (ws_x_raw * reps)[:T]
ws_x_base = ws_x_raw[:T]

# ================================================================
# SOLVE - ein Lauf je Gebaeude aus der Tabelle
# ================================================================
results = {}
for b in BUILDINGS:
    demands    = [space_demand_kw(t, b["last_minus12"]) + dhw_demand_kw(h) for t, h in zip(temps, hours)]
    cops       = [get_cop(t, b["vorlauf"]) for t in temps]
    elec_per_h = [b["hp_kw"]/c for c in cops]
    results[b["name"]] = solve_building(b, demands, prices, elec_per_h, ws_x_base, T)
    results[b["name"]]["demands"] = demands  # fuer Beispieltag-Plots o.ae. aufheben

# ================================================================
# PRINT COMPARISON TABLE
# ================================================================
print(f"\n{'='*100}")
print(f"{'Gebaeude':10} {'WP kW':>6} {'Vorlauf':>7} {'HP h':>6} {'Elec kWh':>10} "
      f"{'Cost EUR':>10} {'Avg EUR/MWh':>12} {'Zyklen':>7} {'Defizit kWh':>12} {'Defizit h':>10}")
print(f"{'-'*100}")
for b in BUILDINGS:
    r = results[b["name"]]
    print(f"{r['name']:10} {r['hp_kw']:>6} {r['vorlauf']:>7} {r['hp_hours']:>6} {r['total_elec']:>10.1f} "
          f"{r['total_cost']:>10.2f} {r['avg_paid']:>12.2f} {r['cycles']:>7} "
          f"{r['total_deficit']:>12.2f} {r['deficit_hours']:>10}")
print(f"\nMarket average price: {avg_market_price:.2f} EUR/MWh")

# ================================================================
# PLOT: Kennzahlenvergleich ueber alle 8 Gebaeude
# ================================================================
names  = [b["name"] for b in BUILDINGS]
labels = [f"{n}\n{results[n]['hp_kw']:.0f}kW/VL{results[n]['vorlauf']}" for n in names]
colors = ["seagreen" if results[n]["vorlauf"] == 35 else "tomato" for n in names]

def add_values(ax, bars, fmt="{:.0f}"):
    top = max((b.get_height() for b in bars), default=1)
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01*max(top,1e-6), fmt.format(v),
                 ha="center", va="bottom", fontsize=7)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.suptitle(f"Sensitivitätsanalyse efh1-efh8 ({T}h)", fontsize=12, fontweight="bold")

ax = axes[0]
vals = [results[n]["total_cost"] for n in names]
bars = ax.bar(labels, vals, color=colors, alpha=0.85); add_values(ax, bars, "{:.0f}")
ax.set_ylabel("EUR"); ax.set_title("Stromkosten"); ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(axis="x", labelsize=7)

ax = axes[1]
vals = [results[n]["total_elec"] for n in names]
bars = ax.bar(labels, vals, color=colors, alpha=0.85); add_values(ax, bars, "{:.0f}")
ax.set_ylabel("kWh"); ax.set_title("Stromverbrauch"); ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(axis="x", labelsize=7)

ax = axes[2]
vals = [results[n]["avg_paid"] for n in names]
bars = ax.bar(labels, vals, color=colors, alpha=0.85); add_values(ax, bars, "{:.1f}")
ax.axhline(avg_market_price, color="navy", linestyle="--", label=f"Marktdurchschnitt ({avg_market_price:.0f})")
ax.set_ylabel("EUR/MWh"); ax.set_title("Ø bezahlter Preis"); ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y"); ax.tick_params(axis="x", labelsize=7)

ax = axes[3]
vals = [results[n]["total_deficit"] for n in names]
bars = ax.bar(labels, vals, color=colors, alpha=0.85); add_values(ax, bars, "{:.2f}")
ax.set_ylabel("kWh"); ax.set_title("Defizit (ungedeckter Bedarf)")
ax.grid(True, alpha=0.3, axis="y"); ax.tick_params(axis="x", labelsize=7)

fig.legend(handles=[mpatches.Patch(color="seagreen", label="VL35 (FBH)"),
                     mpatches.Patch(color="tomato", label="VL55 (Heizkörper)")],
           loc="upper right", fontsize=8, bbox_to_anchor=(0.99, 0.98))
plt.tight_layout()
plt.savefig("efh1_8_sensitivity_summary.png", dpi=150)
plt.show()
print("Saved: efh1_8_sensitivity_summary.png")

# ================================================================
# PLOT: Wann tritt ein Bedarfsdefizit auf? (je Gebaeude, nur falls > 0)
# ================================================================
dts = [datetime.strptime(d, "%Y-%m-%dT%H:%M") for d in datetimes]
buildings_with_deficit = [n for n in names if results[n]["total_deficit"] > 1e-3]

if buildings_with_deficit:
    fig, axes = plt.subplots(len(buildings_with_deficit)+1, 1,
                              figsize=(14, 2.2*(len(buildings_with_deficit)+1)), sharex=True)
    axes[0].plot(dts, temps, color="steelblue", linewidth=0.4)
    axes[0].axhline(-12, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Außentemp.\n[°C]", fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("Wann kann der Wärmebedarf nicht gedeckt werden?", fontsize=11, fontweight="bold")
    for i, n in enumerate(buildings_with_deficit):
        r = results[n]
        ax = axes[i+1]
        ax.fill_between(dts, r["deficit_vals"], step="post", color="tomato", alpha=0.6)
        ax.set_ylabel(f"{n}\nDefizit [kWh/h]", fontsize=7)
        ax.text(0.01, 0.85, f"{r['total_deficit']:.1f} kWh in {r['deficit_hours']} Std.",
                transform=ax.transAxes, fontsize=7, va="top")
        ax.grid(True, alpha=0.15, axis="x")
    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("Datum")
    plt.tight_layout()
    plt.savefig("efh1_8_deficit_timing.png", dpi=150)
    plt.show()
    print("Saved: efh1_8_deficit_timing.png")
else:
    print("\nHinweis: In keinem der acht Gebaeude trat ein Defizit auf "
          "(mit 1000L-Tank auf den vorliegenden Jahresdaten) - keine Defizit-Grafik erzeugt.")


# ================================================================
# EXPORT: Ergebnisse als JSON fuer efh_sensitivity_plots.py
# ================================================================
with open("sensitivity_results.json", "w") as f:
    json.dump({
        "datetimes": datetimes, "temps": temps,
        "results": {n: {k: v for k, v in results[n].items() if k != "demands"} for n in names},
    }, f)
print("Saved: sensitivity_results.json (Grundlage fuer efh_sensitivity_plots.py)")

print("\nDone!")