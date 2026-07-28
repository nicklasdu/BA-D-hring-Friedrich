"""
Pyomo Kaskaden-Waermepumpenoptimierung - MFH "Hamburger Neubau mit Sozialwohnungen"
10 WE, 600 m² Wohnflaeche, Norm-Heizlast ~24 kW (40 W/m², KfW55-Niveau)
Kaskade: 12 + 12 + 6 kW, monovalent, freie Zuordnung der Module durch den Solver
Tank-Groessen-Sensitivitaet analog zum Einzelhaus-Skript, auf MFH-Massstab skaliert
"""

import csv
import pyomo.environ as pyo
from pyomo_windows.solvers import SolverManager
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

INPUT_FILE = "hamburg_heating_electricity_2025.csv"

# ================================================================
# GEBAEUDE / KASKADE
# ================================================================
N_UNITS             = 10
LIVING_AREA_M2      = 600
SPEC_LOAD_W_PER_M2  = 40                                   # KfW55-Niveau (oberes Band)
DESIGN_LOAD_KW      = LIVING_AREA_M2 * SPEC_LOAD_W_PER_M2 / 1000   # = 24.0 kW

# Asymmetrische Kaskade, monovalent (kein bivalenter Backup fuer die Heizlast).
# Gleiche COP-Kennlinie fuer alle Module angenommen (baugleiche Kaeltekreis-
# Technologie unterschiedlicher Nennleistung, z.B. Viessmann Vitocal 250-A /
# Mitsubishi Ecodan PUZ-Kaskadenreihe). Die beiden 12-kW-Module allein (24 kW)
# decken die volle Auslegungslast -> n-1-Redundanz bei Ausfall eines Moduls.
MODULES = [
    ("HP_A_12kW", 12.0),
    ("HP_B_12kW", 12.0),
    ("HP_C_6kW",   6.0),
]
MODULE_NAMES = [m for m, _ in MODULES]
CAPACITIES   = {m: c for m, c in MODULES}

TANK_START_PCT = 0.50
TANK_LOSS_PCT  = 0.01     # 1 %/h, wie im Einzelhaus-Skript (Annahme beibehalten)
MIN_RUNTIME_H  = 2        # je Modul (Verdichterschutz)
MIN_OFF_H      = 1        # je Modul
N_HOURS        = 336      # 2 Wochen

# Tankgroessen: Einzelhaus-Skript testete 500-1250 L bei 7 kW HP-Leistung
# (Speicherzeit bei Volllast ca. 2.9-7.25 h). Bei 30 kW installierter Kaskaden-
# leistung skaliert das linear auf ca. 2150-5375 L -> gerundet 2000-5000 L.
KWH_PER_LITRE = 20.3 / 500   # 0.0406 kWh/L, wie im Einzelhaus-Skript (30-65°C nutzbar)
TANK_SIZES_L  = [2000, 3000, 4000, 5000]
TANK_SIZES = [
    (round(l * KWH_PER_LITRE, 1), l, f"{l} L ({round(l*KWH_PER_LITRE)} kWh)")
    for l in TANK_SIZES_L
]

# ================================================================
# WARMWASSER - linear mit Anzahl WE skaliert.
# Begruendung: Die DIN-4708-Gleichzeitigkeitsfaktoren gelten fuer Spitzenlast
# auf Minuten-/Sekundenbasis (Auslegung Durchlauferhitzer/Rohrnetz). Auf der
# hier verwendeten Stundenaufloesung verteilt sich der Duschbedarf mehrerer
# Wohnungen ohnehin ueber die Stunde, eine zusaetzliche Diversitaets-Reduktion
# ist auf diesem Zeitraster nicht sachgerecht begruendbar.
# ================================================================
DHW_SHOWER_PER_PEAK = N_UNITS * (2500 * 0.50) / 365 / 2
DHW_BASE_KW         = N_UNITS * (2500 * 0.50) / (365 * 16)

def dhw_demand_kw(hour):
    d = 0.0
    if hour in (7, 19):
        d += DHW_SHOWER_PER_PEAK
    if 6 <= hour <= 21:
        d += DHW_BASE_KW
    return d

# ================================================================
# COP-KENNLINIE (unveraendert aus dem Einzelhaus-Skript uebernommen,
# gilt fuer alle drei Module)
# ================================================================
COP_CURVE = [(-15, 1.8), (-10, 2.2), (-7, 2.5), (-5, 2.7), (0, 3.1), (2, 3.3),
             (5, 3.7), (7, 4.0), (10, 4.4), (12, 4.7), (15, 5.0), (20, 5.5)]

def get_cop(temp_c):
    if temp_c <= COP_CURVE[0][0]:
        return COP_CURVE[0][1]
    if temp_c >= COP_CURVE[-1][0]:
        return COP_CURVE[-1][1]
    for i in range(len(COP_CURVE) - 1):
        t0, c0 = COP_CURVE[i]
        t1, c1 = COP_CURVE[i + 1]
        if t0 <= temp_c <= t1:
            return c0 + (temp_c - t0) / (t1 - t0) * (c1 - c0)
    return 3.0


def compute_tank(x_by_module, demands, tank_start, tank_max, loss_pct, T):
    """x_by_module: dict[module_name] -> list[0/1] length T"""
    tanks = []
    tank = tank_start
    for t in range(T):
        hp_out = sum(CAPACITIES[m] * x_by_module[m][t] for m in MODULE_NAMES)
        tank = tank * (1 - loss_pct) - demands[t] + hp_out
        tank = max(0.0, min(tank, tank_max))
        tanks.append(tank)
    return tanks


def build_warmstart(demands, prices, tank_max, T, cheap_thresh):
    """Heuristischer MIP-Start: laedt den Speicher in guenstigen Stunden,
    schaltet im Notfall (Tank nahe leer) das kleinste Modul zwingend zu.
    Dient nur als Startloesung fuer den Solver, keine Optimalitaetsgarantie."""
    ordered = sorted(MODULE_NAMES, key=lambda m: CAPACITIES[m])  # klein -> gross
    x = {m: [0] * T for m in MODULE_NAMES}
    tank = tank_max * TANK_START_PCT
    for t in range(T):
        low   = tank < tank_max * 0.30
        cheap = prices[t] <= cheap_thresh
        on = []
        if low or cheap:
            running_cap = 0.0
            for m in ordered:
                on.append(m)
                running_cap += CAPACITIES[m]
                if running_cap >= DESIGN_LOAD_KW:
                    break
        for m in MODULE_NAMES:
            x[m][t] = 1 if m in on else 0
        hp_out = sum(CAPACITIES[m] * x[m][t] for m in MODULE_NAMES)
        tank = max(0.0, min(tank_max, tank * (1 - TANK_LOSS_PCT) - demands[t] + hp_out))
        if tank <= 0.01 and not on:
            x[ordered[0]][t] = 1  # Notfall: kleinstes Modul erzwingen
    return x


def solve_cascade(tank_max_kwh, demands, prices, cops, T, label):
    tank_start = tank_max_kwh * TANK_START_PCT
    price_sorted = sorted(prices)
    cheap_thresh = price_sorted[int(len(price_sorted) * 0.40)]

    ws_x    = build_warmstart(demands, prices, tank_max_kwh, T, cheap_thresh)
    ws_tank = compute_tank(ws_x, demands, tank_start, tank_max_kwh, TANK_LOSS_PCT, T)

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T - 1)
    model.M = pyo.Set(initialize=MODULE_NAMES)

    model.demand = pyo.Param(model.T, initialize=dict(enumerate(demands)))
    model.price  = pyo.Param(model.T, initialize=dict(enumerate(prices)))
    model.cop    = pyo.Param(model.T, initialize=dict(enumerate(cops)))
    model.cap    = pyo.Param(model.M, initialize=CAPACITIES)

    model.x = pyo.Var(model.M, model.T, domain=pyo.Binary)
    model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, tank_max_kwh))

    for m in MODULE_NAMES:
        for t in range(T):
            model.x[m, t].set_value(ws_x[m][t])
    for t in range(T):
        model.s[t].set_value(max(0.0, ws_tank[t]))

    model.obj = pyo.Objective(
        expr=sum(model.x[m, t] * model.cap[m] / model.cop[t] * model.price[t] / 1000
                  for m in model.M for t in model.T),
        sense=pyo.minimize)

    def tank_balance_rule(mdl, t):
        prev = tank_start if t == 0 else mdl.s[t - 1]
        hp_out = sum(mdl.cap[m] * mdl.x[m, t] for m in mdl.M)
        return mdl.s[t] == prev * (1 - TANK_LOSS_PCT) - mdl.demand[t] + hp_out
    model.tank_balance = pyo.Constraint(model.T, rule=tank_balance_rule)

    def min_on_rule(mdl, m, t):
        if t < 1 or t + MIN_RUNTIME_H > T:
            return pyo.Constraint.Skip
        return sum(mdl.x[m, t + k] for k in range(MIN_RUNTIME_H)) >= \
               MIN_RUNTIME_H * (mdl.x[m, t] - mdl.x[m, t - 1])
    model.min_on = pyo.Constraint(model.M, model.T, rule=min_on_rule)

    def min_off_rule(mdl, m, t):
        if t < 1 or t + 1 >= T:
            return pyo.Constraint.Skip
        return mdl.x[m, t - 1] - mdl.x[m, t] + mdl.x[m, t + 1] <= 1
    model.min_off = pyo.Constraint(model.M, model.T, rule=min_off_rule)

    solver_manager = SolverManager()
    solver = solver_manager.get_solver("cbc")
    solver.options["seconds"] = 300
    solver.options["ratio"]   = 0.01
    solver.options["heurist"] = "on"

    print(f"\n  Solving {label}...")
    result = solver.solve(model, tee=False, warmstart=True)
    status = str(result.solver.termination_condition)

    x_vals = {}
    try:
        for m in MODULE_NAMES:
            raw = [pyo.value(model.x[m, t]) for t in model.T]
            if all(abs(v - round(v)) < 0.01 for v in raw):
                x_vals[m] = [round(v) for v in raw]
            else:
                x_vals[m] = ws_x[m][:]
                status += " (non-integer, using warmstart)"
    except Exception:
        x_vals = ws_x
        status += " (extraction failed)"

    s_vals = compute_tank(x_vals, demands, tank_start, tank_max_kwh, TANK_LOSS_PCT, T)

    total_cost = sum(x_vals[m][t] * CAPACITIES[m] / cops[t] * prices[t] / 1000
                      for m in MODULE_NAMES for t in range(T))
    total_elec = sum(x_vals[m][t] * CAPACITIES[m] / cops[t]
                      for m in MODULE_NAMES for t in range(T))
    hp_hours_per_module = {m: sum(x_vals[m]) for m in MODULE_NAMES}
    cycles_per_module = {
        m: sum(1 for t in range(1, T) if x_vals[m][t] == 1 and x_vals[m][t - 1] == 0)
        for m in MODULE_NAMES
    }
    avg_paid = total_cost / total_elec * 1000 if total_elec > 0 else 0
    min_tank = min(s_vals)

    print(f"  Status: {status}")
    print(f"  Module-Betriebsstunden: {hp_hours_per_module}")
    print(f"  Kosten: {total_cost:.2f} EUR, Ø-Preis: {avg_paid:.1f} EUR/MWh, "
          f"Zyklen: {cycles_per_module}")

    return {
        "x_vals": x_vals, "s_vals": s_vals, "hp_hours": hp_hours_per_module,
        "total_elec": total_elec, "total_cost": total_cost, "avg_paid": avg_paid,
        "cycles": cycles_per_module, "min_tank": min_tank, "status": status,
        "tank_max": tank_max_kwh,
    }


# ================================================================
# DATEN LADEN
# ================================================================
print("Loading data...")
with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))
rows = all_rows[:N_HOURS]
T = N_HOURS

valid_prices = [float(r["electricity_price_eur_per_mwh"])
                 for r in all_rows if r["electricity_price_eur_per_mwh"] != ""]
avg_market_price = sum(valid_prices) / len(valid_prices)

# Skalierungsfaktor: Heizlastprofil stammt aus dem Einzelhaus-Datensatz und wird
# proportional auf die MFH-Auslegungslast (24 kW) skaliert. Faktor wird aus den
# Daten berechnet, nicht hart codiert.
household_peak_kw = max(float(r["heating_output_w"]) for r in all_rows) / 1000
scale_factor = DESIGN_LOAD_KW / household_peak_kw
print(f"Skalierungsfaktor Heizlast: {scale_factor:.2f} "
      f"(Haushalts-Peak {household_peak_kw:.2f} kW -> MFH-Auslegungslast {DESIGN_LOAD_KW:.1f} kW)")

datetimes, temps, demands, prices, cops = [], [], [], [], []
for r in rows:
    dt  = r["datetime"]
    h   = int(dt[11:13])
    tmp = float(r["temperature_outdoor_c"])
    dem = float(r["heating_output_w"]) / 1000 * scale_factor + dhw_demand_kw(h)
    p   = float(r["electricity_price_eur_per_mwh"]) if r["electricity_price_eur_per_mwh"] != "" else avg_market_price
    datetimes.append(dt)
    temps.append(tmp)
    demands.append(dem)
    prices.append(p)
    cops.append(get_cop(tmp))

# ================================================================
# ALLE TANKGROESSEN LOESEN
# ================================================================
print(f"\nSolving {len(TANK_SIZES)} tank configurations (up to 5 min each)...")
results = {}
for tank_kwh, tank_litres, label in TANK_SIZES:
    results[label] = solve_cascade(tank_kwh, demands, prices, cops, N_HOURS, label)

# ================================================================
# VERGLEICHSTABELLE
# ================================================================
print(f"\n{'='*95}")
print(f"{'Tank':22} {'Kosten EUR':>10} {'Ø EUR/MWh':>10} {'Min.Tank':>9}  Betriebsstunden je Modul")
print(f"{'-'*95}")
for tank_kwh, tank_litres, label in TANK_SIZES:
    r = results[label]
    hrs = ", ".join(f"{m}:{h}" for m, h in r["hp_hours"].items())
    print(f"{label:22} {r['total_cost']:>10.2f} {r['avg_paid']:>10.2f} {r['min_tank']:>9.2f}  {hrs}")
print(f"\nMarktdurchschnittspreis: {avg_market_price:.2f} EUR/MWh")
print(f"MFH-Auslegungslast: {DESIGN_LOAD_KW:.1f} kW | Kaskade installiert: "
      f"{sum(CAPACITIES.values()):.0f} kW ({', '.join(f'{m}={c}kW' for m,c in MODULES)})")

# ================================================================
# PLOT: Tanklevel + Modul-Dispatch fuer die groesste Tankvariante
# ================================================================
dts = [datetime.strptime(d, "%Y-%m-%dT%H:%M") for d in datetimes]
best_label = TANK_SIZES[-1][2]
r = results[best_label]
tank_max = r["tank_max"]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                          gridspec_kw={"height_ratios": [1, 1.4, 1]})
fig.suptitle(f"Kaskaden-Fahrweise - {best_label}", fontsize=12, fontweight="bold")

axes[0].plot(dts, prices, color="gray", linewidth=1)
axes[0].set_ylabel("Preis\n[EUR/MWh]", fontsize=8)
axes[0].grid(True, alpha=0.25)

tank_pct = [v / tank_max * 100 for v in r["s_vals"]]
axes[1].plot(dts, tank_pct, color="steelblue", linewidth=1.3)
axes[1].fill_between(dts, tank_pct, alpha=0.15, color="steelblue")
axes[1].axhline(100, color="gray", linestyle=":", linewidth=1)
axes[1].set_ylabel("Tank [% voll]", fontsize=8)
axes[1].set_ylim(-5, 115)
axes[1].grid(True, alpha=0.25)

colors = {"HP_A_12kW": "seagreen", "HP_B_12kW": "darkorange", "HP_C_6kW": "tomato"}
y0 = 0
for m in MODULE_NAMES:
    for t in range(N_HOURS - 1):
        if r["x_vals"][m][t] == 1:
            axes[2].axvspan(dts[t], dts[t + 1], ymin=y0/3, ymax=(y0+1)/3,
                              color=colors[m], alpha=0.8)
    y0 += 1
axes[2].set_yticks([0.17, 0.5, 0.83])
axes[2].set_yticklabels(MODULE_NAMES, fontsize=7)
axes[2].set_xlabel("Datum")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
axes[2].xaxis.set_major_locator(mdates.DayLocator())
plt.tight_layout()
plt.savefig("cascade_dispatch.png", dpi=150)
plt.show()
print("Saved: cascade_dispatch.png")

print("\nDone!")