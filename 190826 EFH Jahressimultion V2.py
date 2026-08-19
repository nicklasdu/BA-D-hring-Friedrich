"""
Pyomo Heat Pump Optimisation - EFH, Volljahr, mit Modulation (30-100%)

Erweiterung ggue. der 2-Wochen-Version:
- N_HOURS_REQUESTED auf 8760h (Volljahr) gesetzt statt 336h (2 Wochen).
  Tatsaechlich verwendet wird min(N_HOURS_REQUESTED, verfuegbare Datenzeilen).
- Solver-Zeitlimit auf SOLVER_TIME_LIMIT_S (Default 3600s = 1h) je Tankgroesse
  erhoeht; MIP-Gap (MIP_GAP) auf 2% gelockert, da ein Volljahres-MILP mit
  ~8760 Binaervariablen deutlich schwerer zu loesen ist als der 2-Wochen-Fall.
- RUN_ALL_TANK_SIZES=False als Default: es wird nur EINE Tankgroesse geloest,
  um zunaechst die Machbarkeit/Laufzeit zu testen, bevor alle vier (bis zu
  4x SOLVER_TIME_LIMIT_S) gerechnet werden. Auf True setzen fuer alle vier.
- Warmstart-Laenge wird geprueft und bei Bedarf per Wiederholung aufgefuellt,
  statt bei einer zu kurzen simulation_results.csv mit IndexError abzubrechen.
- Modulation (u(t) in {0} u [MOD_MIN,1]) wie in der 2-Wochen-Version.

WICHTIG: Auch mit diesen Anpassungen ist nicht garantiert, dass CBC das
Volljahresproblem innerhalb des Zeitlimits bis zur Optimalitaet loest. Der
Solver liefert nach Ablauf von SOLVER_TIME_LIMIT_S in der Regel trotzdem die
beste bis dahin gefundene Loesung zurueck (kein Abbruch ohne Ergebnis) -
`status` zeigt dann z.B. "maxTimeLimit" statt "optimal" an.
"""

import csv
import numpy as np
import pyomo.environ as pyo
from pyomo_windows.solvers import SolverManager
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

INPUT_FILE     = "hamburg_heating_electricity_2025.csv"
WARMSTART_FILE = "simulation_results.csv"

HP_OUTPUT_KW   = 7.0
MOD_MIN        = 0.30      # minimaler Modulationsgrad (30% der Nennleistung)
TANK_START_PCT = 0.50      # start at 50% full
TANK_LOSS_PCT  = 0.01      # 1% loss per hour
MIN_RUNTIME_H  = 2

# ---- Volljahr-Konfiguration ----
N_HOURS_REQUESTED   = 8760     # Ziel: Volljahr. Wird ggf. auf verfuegbare Daten begrenzt.
SOLVER_TIME_LIMIT_S = 3600     # 1h pro Tankgroesse (statt 300s im 2-Wochen-Fall)
MIP_GAP              = 0.02    # 2% statt 1% - realistischer fuer ein grosses MILP
RUN_ALL_TANK_SIZES   = False   # False = nur EINE Tankgroesse (Machbarkeitstest zuerst)
SINGLE_TANK_INDEX    = 2       # Index in TANK_SIZES, falls RUN_ALL_TANK_SIZES=False (2 = 1000L)

# Tank sizes to compare (kWh and approximate litres)
TANK_SIZES = [
    (20.3,  500,  "500L (20 kWh)"),
    (30.45, 750,  "750L (30 kWh)"),
    (40.6,  1000, "1000L (40 kWh)"),
    (50.75, 1250, "1250L (50 kWh)"),
]

DHW_SHOWER_PER_PEAK = (2500 * 0.50) / 365 / 2
DHW_BASE_KW         = (2500 * 0.50) / (365 * 16)

def dhw_demand_kw(hour):
    d = 0.0
    if hour in (7, 19): d += DHW_SHOWER_PER_PEAK
    if 6 <= hour <= 21: d += DHW_BASE_KW
    return d

COP_CURVE = [(-15,1.8),(-10,2.2),(-7,2.5),(-5,2.7),(0,3.1),(2,3.3),
             (5,3.7),(7,4.0),(10,4.4),(12,4.7),(15,5.0),(20,5.5)]

def get_cop(temp_c):
    if temp_c <= COP_CURVE[0][0]:  return COP_CURVE[0][1]
    if temp_c >= COP_CURVE[-1][0]: return COP_CURVE[-1][1]
    for i in range(len(COP_CURVE)-1):
        t0,c0 = COP_CURVE[i]; t1,c1 = COP_CURVE[i+1]
        if t0 <= temp_c <= t1:
            return c0 + (temp_c-t0)/(t1-t0)*(c1-c0)
    return 3.0

def compute_tank(u_schedule, demands, tank_start, tank_max, loss_pct):
    tanks = []
    tank  = tank_start
    for t in range(len(u_schedule)):
        tank = tank*(1-loss_pct) - demands[t] + HP_OUTPUT_KW*u_schedule[t]
        tank = max(0.0, min(tank, tank_max))
        tanks.append(tank)
    return tanks

def solve_tank(tank_max_kwh, demands, prices, elec_per_h, cops,
               ws_x, n_hours, label):
    T          = n_hours
    tank_start = tank_max_kwh * TANK_START_PCT
    ws_u       = ws_x[:]
    ws_tank    = compute_tank(ws_u, demands, tank_start, tank_max_kwh, TANK_LOSS_PCT)

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T-1)
    model.demand   = pyo.Param(model.T, initialize=dict(enumerate(demands)))
    model.price    = pyo.Param(model.T, initialize=dict(enumerate(prices)))
    model.elec_kwh = pyo.Param(model.T, initialize=dict(enumerate(elec_per_h)))

    model.x = pyo.Var(model.T, domain=pyo.Binary)
    model.u = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals,
                      bounds=(0, tank_max_kwh))

    for t in range(T):
        model.x[t].set_value(ws_x[t])
        model.u[t].set_value(ws_u[t])
        model.s[t].set_value(max(0.0, ws_tank[t]))

    model.obj = pyo.Objective(
        expr=sum(model.u[t]*model.elec_kwh[t]*model.price[t]/1000
                 for t in model.T),
        sense=pyo.minimize)

    def tank_balance_rule(m, t):
        prev = tank_start if t == 0 else m.s[t-1]
        return m.s[t] == prev*(1-TANK_LOSS_PCT) - m.demand[t] + HP_OUTPUT_KW*m.u[t]
    model.tank_balance = pyo.Constraint(model.T, rule=tank_balance_rule)

    def mod_upper_rule(m, t):
        return m.u[t] <= m.x[t]
    model.mod_upper = pyo.Constraint(model.T, rule=mod_upper_rule)

    def mod_lower_rule(m, t):
        return m.u[t] >= MOD_MIN * m.x[t]
    model.mod_lower = pyo.Constraint(model.T, rule=mod_lower_rule)

    def min_on_rule(m, t):
        if t < 1 or t + MIN_RUNTIME_H > T:
            return pyo.Constraint.Skip
        return sum(m.x[t+k] for k in range(MIN_RUNTIME_H)) >= \
               MIN_RUNTIME_H*(m.x[t]-m.x[t-1])
    model.min_on = pyo.Constraint(model.T, rule=min_on_rule)

    # Keine explizite Mindeststillstands-Nebenbedingung (bei stuendlicher
    # Aufloesung fuer F_min=1h ohnehin wirkungslos, siehe Methodik-Dokument).

    solver_manager = SolverManager()
    solver = solver_manager.get_solver("cbc")
    solver.options["seconds"] = SOLVER_TIME_LIMIT_S
    solver.options["ratio"]   = MIP_GAP
    solver.options["heurist"] = "on"

    print(f"\n  Solving {label} (bis zu {SOLVER_TIME_LIMIT_S/60:.0f} min, "
          f"MIP-Gap {MIP_GAP*100:.0f}%, {T} Stunden, {2*T} Binaer-/Kopplungsvariablen)...")
    result = solver.solve(model, tee=False, warmstart=True)
    status = str(result.solver.termination_condition)

    try:
        raw_x = [pyo.value(model.x[t]) for t in model.T]
        raw_u = [pyo.value(model.u[t]) for t in model.T]
        if all(abs(v - round(v)) < 0.01 for v in raw_x):
            x_vals = [round(v) for v in raw_x]
        else:
            x_vals = ws_x[:]
            status += " (non-integer, using warmstart)"
        u_vals = [min(1.0, max(0.0, v)) for v in raw_u]
    except Exception:
        x_vals = ws_x[:]
        u_vals = ws_u[:]
        status += " (extraction failed)"

    s_vals = compute_tank(u_vals, demands, tank_start, tank_max_kwh, TANK_LOSS_PCT)

    total_cost = sum(u_vals[t]*elec_per_h[t]*prices[t]/1000 for t in range(T))
    total_elec = sum(u_vals[t]*elec_per_h[t] for t in range(T))
    hp_hours   = sum(x_vals)
    avg_paid   = total_cost/total_elec*1000 if total_elec > 0 else 0
    cycles     = sum(1 for t in range(1,T) if x_vals[t]==1 and x_vals[t-1]==0)
    min_tank   = min(s_vals)
    on_hours   = [u for u, x in zip(u_vals, x_vals) if x == 1]
    avg_mod    = sum(on_hours)/len(on_hours) if on_hours else 0.0

    print(f"  Status: {status}")
    print(f"  HP hours: {hp_hours}, Cost: {total_cost:.2f} EUR, "
          f"Avg price: {avg_paid:.1f} EUR/MWh, Cycles: {cycles}, "
          f"Ø Modulation wenn an: {avg_mod*100:.0f}%")

    return {
        "x_vals": x_vals, "u_vals": u_vals, "s_vals": s_vals, "hp_hours": hp_hours,
        "total_elec": total_elec, "total_cost": total_cost, "avg_paid": avg_paid,
        "cycles": cycles, "min_tank": min_tank, "avg_mod": avg_mod,
        "status": status, "tank_max": tank_max_kwh,
    }

# ================================================================
# LOAD DATA
# ================================================================
print("Loading data...")
with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))

N_HOURS = min(N_HOURS_REQUESTED, len(all_rows))
if N_HOURS < N_HOURS_REQUESTED:
    print(f"WARNUNG: {INPUT_FILE} enthaelt nur {len(all_rows)} Zeilen, "
          f"nicht die angeforderten {N_HOURS_REQUESTED}. Verwende N_HOURS={N_HOURS}.")
rows = all_rows[:N_HOURS]
T    = N_HOURS

valid_prices     = [float(r["electricity_price_eur_per_mwh"])
                    for r in all_rows if r["electricity_price_eur_per_mwh"] != ""]
avg_market_price = sum(valid_prices)/len(valid_prices)

datetimes,temps,demands,prices,cops,elec_per_h = [],[],[],[],[],[]
for r in rows:
    dt  = r["datetime"]
    h   = int(dt[11:13])
    tmp = float(r["temperature_outdoor_c"])
    dem = float(r["heating_output_w"])/1000 + dhw_demand_kw(h)
    p   = float(r["electricity_price_eur_per_mwh"]) if r["electricity_price_eur_per_mwh"] != "" else avg_market_price
    cop = get_cop(tmp)
    datetimes.append(dt); temps.append(tmp); demands.append(dem)
    prices.append(p); cops.append(cop); elec_per_h.append(HP_OUTPUT_KW/cop)

# Warmstart laden und auf T Stunden auffuellen, falls die Datei kuerzer ist
# (z.B. wenn bislang nur ein 2-Wochen-Warmstart existiert). Ohne dieses
# Auffuellen wuerde ws_x[t] fuer t >= len(ws_x) mit IndexError abbrechen.
with open(WARMSTART_FILE, newline="", encoding="utf-8") as f:
    ws_rows_all = list(csv.DictReader(f))
ws_x_raw = [int(row["hp_running"]) for row in ws_rows_all]
if len(ws_x_raw) < T:
    print(f"WARNUNG: {WARMSTART_FILE} hat nur {len(ws_x_raw)} Zeilen fuer T={T} "
          f"Stunden. Fuelle den Warmstart durch Wiederholung des Musters auf "
          f"(nur Startloesung fuer den Solver, keine Auswirkung auf Korrektheit).")
    reps = (T // len(ws_x_raw)) + 1
    ws_x_raw = (ws_x_raw * reps)[:T]
ws_x = ws_x_raw[:T]

# ================================================================
# SOLVE (Machbarkeitstest: 1 Tankgroesse, oder alle 4 falls RUN_ALL_TANK_SIZES=True)
# ================================================================
TANK_SIZES_ACTIVE = TANK_SIZES if RUN_ALL_TANK_SIZES else [TANK_SIZES[SINGLE_TANK_INDEX]]

worst_case_min = len(TANK_SIZES_ACTIVE) * SOLVER_TIME_LIMIT_S / 60
print(f"\n{len(TANK_SIZES_ACTIVE)} Tankgroesse(n) ausgewaehlt "
      f"({', '.join(l for _,_,l in TANK_SIZES_ACTIVE)}), "
      f"Zeitbudget bis zu {worst_case_min:.0f} min insgesamt.")
if not RUN_ALL_TANK_SIZES:
    print("Hinweis: RUN_ALL_TANK_SIZES=True setzen, um alle vier Tankgroessen "
          "zu rechnen, sobald die Machbarkeit fuer eine Groesse bestaetigt ist.")

results = {}
for tank_kwh, tank_litres, label in TANK_SIZES_ACTIVE:
    results[label] = solve_tank(
        tank_kwh, demands, prices, elec_per_h, cops,
        ws_x, N_HOURS, label)

# ================================================================
# PRINT COMPARISON TABLE
# ================================================================
print(f"\n{'='*95}")
print(f"{'Tank':22} {'HP h':>6} {'Elec kWh':>10} {'Cost EUR':>10} "
      f"{'Avg EUR/MWh':>12} {'Cycles':>7} {'Min tank':>9} {'Ø Mod.':>8}")
print(f"{'-'*95}")
for tank_kwh, tank_litres, label in TANK_SIZES_ACTIVE:
    r = results[label]
    print(f"{label:22} {r['hp_hours']:>6} {r['total_elec']:>10.1f} "
          f"{r['total_cost']:>10.2f} {r['avg_paid']:>12.2f} "
          f"{r['cycles']:>7} {r['min_tank']:>9.3f} {r['avg_mod']*100:>7.0f}%")
print(f"\nMarket average price: {avg_market_price:.2f} EUR/MWh")

# ================================================================
# PLOTS (Achsenformat passt sich automatisch an 2 Wochen oder Volljahr an)
# ================================================================
dts    = [datetime.strptime(d, "%Y-%m-%dT%H:%M") for d in datetimes]
labels = [label for _, _, label in TANK_SIZES_ACTIVE]
colors = ["steelblue", "seagreen", "darkorange", "tomato"][:len(TANK_SIZES_ACTIVE)]

def format_date_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

# --- Plot 1: Tank level for all active tank sizes ---
fig, axes = plt.subplots(len(TANK_SIZES_ACTIVE), 1, figsize=(14, 3*len(TANK_SIZES_ACTIVE)),
                          sharex=True, squeeze=False)
axes = axes[:, 0]
fig.suptitle(f"Buffer Tank Level – {N_HOURS}h – mit Modulation", fontsize=12, fontweight="bold")
for i, (tank_kwh, tank_litres, label) in enumerate(TANK_SIZES_ACTIVE):
    r   = results[label]
    ax  = axes[i]
    col = colors[i]
    tank_pct = [v/tank_kwh*100 for v in r["s_vals"]]
    ax.plot(dts, tank_pct, color=col, linewidth=0.8)
    ax.fill_between(dts, tank_pct, alpha=0.15, color=col)
    ax.set_ylabel(f"{label}\n[% full]", fontsize=8)
    ax.set_ylim(-5, 115)
    ax.grid(True, alpha=0.25)
    ax.text(0.01, 0.9, f"Cost: {r['total_cost']:.2f} EUR | HP: {r['hp_hours']}h | "
            f"Cycles: {r['cycles']} | Ø Mod: {r['avg_mod']*100:.0f}%",
            transform=ax.transAxes, fontsize=8, color=col, va="top")
format_date_axis(axes[-1])
axes[-1].set_xlabel("Datum")
plt.tight_layout()
plt.savefig("efh_year_tank_levels.png", dpi=150)
plt.show()
print("Saved: efh_year_tank_levels.png")

# --- Plot 2: Modulationsgrad-Schedule vs. Preis (vektorisiert, auch fuer Volljahr schnell) ---
price_sorted = sorted(valid_prices)
cheap_thresh = price_sorted[int(len(price_sorted)*0.40)]
cheap_mask   = [p <= cheap_thresh for p in prices]
exp_mask     = [not c for c in cheap_mask]

fig, axes = plt.subplots(len(TANK_SIZES_ACTIVE)+1, 1, figsize=(14, 2.2*(len(TANK_SIZES_ACTIVE)+1)),
                          sharex=True, squeeze=False)
axes = axes[:, 0]
fig.suptitle("HP Modulationsgrad je Stunde vs Electricity Price", fontsize=11, fontweight="bold")

axes[0].plot(dts, prices, color="gray", linewidth=0.5)
axes[0].axhline(cheap_thresh, color="green", linestyle="--", linewidth=1)
axes[0].set_ylabel("Price\n[EUR/MWh]", fontsize=8)
axes[0].grid(True, alpha=0.25)

for i, (tank_kwh, tank_litres, label) in enumerate(TANK_SIZES_ACTIVE):
    r  = results[label]
    ax = axes[i+1]
    ax.fill_between(dts, 0, r["u_vals"], where=cheap_mask, step="post",
                     color=colors[i], alpha=0.7, linewidth=0)
    ax.fill_between(dts, 0, r["u_vals"], where=exp_mask, step="post",
                     color="tomato", alpha=0.7, linewidth=0)
    ax.set_ylabel(f"{label[:5]}", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.3, 1.0])
    ax.set_yticklabels(["30%", "100%"], fontsize=6)
    ax.grid(True, alpha=0.15, axis="x")

format_date_axis(axes[-1])
axes[-1].set_xlabel("Datum")
plt.tight_layout()
plt.savefig("efh_year_modulation_schedule.png", dpi=150)
plt.show()
print("Saved: efh_year_modulation_schedule.png")

best_label  = TANK_SIZES_ACTIVE[-1][2]
best_result = results[best_label]
tank_max    = best_result["tank_max"]

# ================================================================
# PLOT 3: Kennzahlenvergleich nach Tankgroesse
# ================================================================
short_labels = [f"{l}L" for _, l, _ in TANK_SIZES_ACTIVE]
costs = [results[lbl]["total_cost"] for _, _, lbl in TANK_SIZES_ACTIVE]
avgp  = [results[lbl]["avg_paid"]   for _, _, lbl in TANK_SIZES_ACTIVE]
mint  = [results[lbl]["min_tank"]   for _, _, lbl in TANK_SIZES_ACTIVE]
cyc   = [results[lbl]["cycles"]     for _, _, lbl in TANK_SIZES_ACTIVE]

def add_values(ax, bars, fmt="{:.1f}"):
    top = max(b.get_height() for b in bars)
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01 * top,
                 fmt.format(v), ha="center", va="bottom", fontsize=9)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"EFH - Kennzahlenvergleich nach Tankgroesse ({N_HOURS}h)", fontsize=12, fontweight="bold")

ax = axes[0, 0]
bars = ax.bar(short_labels, costs, color=colors, alpha=0.85)
add_values(ax, bars, "{:.2f}")
ax.set_ylabel("EUR"); ax.set_title("Gesamtkosten"); ax.grid(True, alpha=0.3, axis="y")

ax = axes[0, 1]
bars = ax.bar(short_labels, avgp, color=colors, alpha=0.85)
ax.axhline(avg_market_price, color="navy", linestyle="--",
           label=f"Marktdurchschnitt ({avg_market_price:.0f})")
add_values(ax, bars, "{:.1f}")
ax.set_ylabel("EUR/MWh"); ax.set_title("Ø bezahlter Preis")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 0]
bars = ax.bar(short_labels, mint, color=colors, alpha=0.85)
add_values(ax, bars, "{:.2f}")
ax.set_ylabel("kWh"); ax.set_title("Minimaler Tankstand"); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 1]
bars = ax.bar(short_labels, cyc, color=colors, alpha=0.85)
add_values(ax, bars, "{:.0f}")
ax.set_ylabel("Zyklen"); ax.set_title("Start/Stopp-Zyklen"); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("efh_year_summary_compare.png", dpi=150)
plt.show()
print("Saved: efh_year_summary_compare.png")

# ================================================================
# PLOT 4: Aussentemperatur und COP ueber den Betrachtungszeitraum
# ================================================================
fig, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(dts, temps, color="steelblue", linewidth=0.6, label="Außentemperatur")
ax1.set_ylabel("Temperatur [°C]", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax2 = ax1.twinx()
ax2.plot(dts, cops, color="darkorange", linewidth=0.6, label="COP")
ax2.set_ylabel("COP", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax1.set_title("Außentemperatur und resultierender COP", fontsize=11, fontweight="bold")
format_date_axis(ax1)
ax1.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("efh_year_cop_temp.png", dpi=150)
plt.show()
print("Saved: efh_year_cop_temp.png")

# ================================================================
# PLOT 5: Dauerlinie des Bedarfs vs. Modulationsbereich der WP
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5))
sorted_demand = sorted(demands, reverse=True)
ax.plot(range(N_HOURS), sorted_demand, color="steelblue", linewidth=1.2, label="Bedarf (sortiert)")
ax.fill_between(range(N_HOURS), HP_OUTPUT_KW * MOD_MIN, HP_OUTPUT_KW,
                alpha=0.08, color="seagreen", label="Modulationsbereich (30-100%)")
ax.axhline(HP_OUTPUT_KW, color="gray", linestyle=":", linewidth=1)
ax.text(N_HOURS * 0.99, HP_OUTPUT_KW + 0.15, f"{HP_OUTPUT_KW:.1f} kW (100%)",
        fontsize=7, ha="right", color="gray")
ax.axhline(HP_OUTPUT_KW * MOD_MIN, color="gray", linestyle=":", linewidth=1)
ax.text(N_HOURS * 0.99, HP_OUTPUT_KW * MOD_MIN + 0.15, f"{HP_OUTPUT_KW*MOD_MIN:.1f} kW (30%)",
        fontsize=7, ha="right", color="gray")
ax.set_xlabel("Stunden (nach Bedarf absteigend sortiert)")
ax.set_ylabel("Wärmebedarf [kW]")
ax.set_title("Dauerlinie des Bedarfs vs. Modulationsbereich der Wärmepumpe",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("efh_year_load_duration.png", dpi=150)
plt.show()
print("Saved: efh_year_load_duration.png")

# ================================================================
# PLOT 6: Kalender-Heatmap - Modulationsgrad je Stunde
# ================================================================
n_days = N_HOURS // 24
heat_data = np.zeros((24, n_days))
for d in range(n_days):
    for h in range(24):
        t = d * 24 + h
        heat_data[h, d] = best_result["u_vals"][t] * 100

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heat_data, aspect="auto", cmap="YlOrRd", origin="lower",
               extent=[0, n_days, 0, 24], vmin=0, vmax=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Modulationsgrad [%]")
ax.set_xlabel("Tag")
ax.set_ylabel("Stunde des Tages")
ax.set_title(f"EFH Modulationsgrad über {n_days} Tage - {best_label}", fontsize=12, fontweight="bold")
if n_days > 40:
    tick_days = [d for d in range(n_days) if dts[d * 24].day == 1]
    ax.set_xticks([d + 0.5 for d in tick_days])
    ax.set_xticklabels([dts[d * 24].strftime("%b") for d in tick_days], fontsize=8)
else:
    ax.set_xticks([d + 0.5 for d in range(n_days)])
    ax.set_xticklabels([dts[d * 24].strftime("%d.%m") for d in range(n_days)],
                        rotation=45, fontsize=7)
ax.set_yticks(range(0, 24, 3))
plt.tight_layout()
plt.savefig("efh_year_heatmap.png", dpi=150)
plt.show()
print("Saved: efh_year_heatmap.png")

# ================================================================
# PLOT 7-11: Extremer Winter-Tag + repraesentativer Tag je Jahreszeit
# "Repraesentativ" = Tag der jeweiligen Saison, dessen Tagesmitteltemperatur
# am naechsten an der Saison-Durchschnittstemperatur liegt (kein Extremwert).
# ================================================================
day_avg_temps = [sum(temps[d * 24:(d + 1) * 24]) / 24 for d in range(n_days)]
extreme_winter_idx = day_avg_temps.index(min(day_avg_temps))

season_of_month = {12: "Winter", 1: "Winter", 2: "Winter",
                    3: "Fruehling", 4: "Fruehling", 5: "Fruehling",
                    6: "Sommer", 7: "Sommer", 8: "Sommer",
                    9: "Herbst", 10: "Herbst", 11: "Herbst"}
day_season = [season_of_month[dts[d * 24].month] for d in range(n_days)]

representative_days = {}
for season in ["Winter", "Fruehling", "Sommer", "Herbst"]:
    idxs = [d for d in range(n_days) if day_season[d] == season]
    if not idxs:
        print(f"Hinweis: Keine Tage fuer Saison {season} im geloesten Zeitraum "
              f"({N_HOURS}h) vorhanden - wird uebersprungen.")
        continue
    season_mean = sum(day_avg_temps[d] for d in idxs) / len(idxs)
    representative_days[season] = min(idxs, key=lambda d: abs(day_avg_temps[d] - season_mean))

def plot_example_day_efh(day_idx, title_suffix, filename):
    start, end = day_idx * 24, day_idx * 24 + 24
    day_dts = dts[start:end]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                              gridspec_kw={"height_ratios": [0.8, 0.8, 1.3, 1]})
    fig.suptitle(f"Beispieltag: {title_suffix} ({day_dts[0].strftime('%d.%m.%Y')}, "
                 f"Ø {day_avg_temps[day_idx]:.1f}°C)", fontsize=12, fontweight="bold")

    axes[0].plot(day_dts, temps[start:end], color="steelblue", marker="o", markersize=3)
    axes[0].set_ylabel("Außentemp.\n[°C]", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(day_dts, prices[start:end], color="gray", marker="o", markersize=3)
    axes[1].set_ylabel("Preis\n[EUR/MWh]", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    power = [HP_OUTPUT_KW * u for u in best_result["u_vals"][start:end]]
    axes[2].fill_between(day_dts, power, step="post", color="seagreen", alpha=0.6, label="WP-Leistung")
    axes[2].step(day_dts, demands[start:end], color="black", linewidth=1.5,
                 linestyle="--", where="post", label="Bedarf")
    axes[2].axhline(HP_OUTPUT_KW, color="gray", linestyle=":", linewidth=0.8)
    axes[2].axhline(HP_OUTPUT_KW * MOD_MIN, color="gray", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel("Leistung\n[kW]", fontsize=8)
    axes[2].legend(fontsize=7, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    tank_pct = [v / tank_max * 100 for v in best_result["s_vals"][start:end]]
    axes[3].plot(day_dts, tank_pct, color="steelblue", linewidth=1.5)
    axes[3].fill_between(day_dts, tank_pct, alpha=0.15, color="steelblue")
    axes[3].set_ylabel("Tank\n[% voll]", fontsize=8)
    axes[3].set_ylim(-5, 115)
    axes[3].set_xlabel("Uhrzeit")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")

plot_example_day_efh(extreme_winter_idx, "Extremer Winter-Tag (kältester Tag im Zeitraum)",
                      "efh_example_extreme_winter.png")
for season, idx in representative_days.items():
    plot_example_day_efh(idx, f"Repräsentativer {season}-Tag", f"efh_example_{season.lower()}.png")

print("\nDone!")