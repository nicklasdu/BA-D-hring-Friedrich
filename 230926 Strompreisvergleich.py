"""
Normaltarif-Simulation fuer die zwei Beispielgebaeude (EFH und MFH)

Abgeleitet aus dem Tarifvergleichsskript (Dynamisch vs. Normaltarif, siehe
Chat) - der komplette Dynamiktarif-Teil (Tibber-Modell, Spotpreis-Einlesen,
Mittelwert-Angleichung) wurde entfernt, hier interessiert ausschliesslich
der Normaltarif (Festpreis) fuer beide Gebaeude. Da ein Festpreis per
Definition nicht vom realen Markt abhaengt, wird die Preisspalte der CSV
gar nicht mehr eingelesen.

Aenderungen ggue. dem Tarifvergleichsskript (siehe Chat fuer die volle
Diskussion):
- WARMSTART_ENABLED auf True gesetzt. War dort False (offenbar ein Debug-
  Rest zusammen mit SOLVER_TEE=True) - fuer einen echten Ergebnislauf sollte
  der Warmstart wie ueberall sonst im Projekt aktiv sein.
- SOLVER_TEE auf False gesetzt (war True) - sonst haette CBC bei bis zu 2x
  60 Minuten Laufzeit die volle Konsolenausgabe erzeugt.
- cheap_thresh (fuer die Warmstart-Heuristik, sowohl MFH als auch EFH) wird
  jetzt aus der tatsaechlich geloesten Preisreihe berechnet statt (bei EFH)
  aus den Spotpreisen - solve_cascade() fuer MFH machte das im Original
  bereits richtig, efh_build_warmstart() nicht (siehe Chat, "Bug C"). Bei
  einem echten Festpreis ist cheap_thresh == prices[t] fuer jede Stunde,
  das "cheap"-Kriterium ist dadurch fuer jede Stunde erfuellt.
- resume_file/save_schedule/load_schedule aus solve_cascade() entfernt -
  die dienten nur dazu, einen Fahrplan zwischen mehreren Tarif-Laeufen
  wiederzuverwenden; bei nur einem Lauf pro Gebaeude nicht mehr noetig.

Alles andere (Bedarfsmodell je Gebaeude - inkl. der bekannten Differenz,
dass EFH weiterhin die vorbereitete heating_output_w-Spalte statt einer
eigenen Heizkurve nutzt, und dass das EFH-Warmwasserprofil noch die alte
50/50-Aufteilung ohne Nachtlast ist, siehe Chat -, COP, Tank, Kaskade,
Deficit/Spill nur bei MFH) ist unveraendert aus dem Tarifvergleichsskript
uebernommen.
"""

import os
import csv
import pyomo.environ as pyo
from pyomo.common.tempfiles import TempfileManager
from pyomo_windows.solvers import SolverManager
import matplotlib.pyplot as plt
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
TempfileManager.tempdir = os.getcwd()

INPUT_FILE = "hamburg_heating_electricity_2025.csv"

# ================================================================
# GEBAEUDE / KASKADE (MFH) - unveraendert aus dem Tarifvergleichsskript
# ================================================================
N_UNITS            = 10
LIVING_AREA_M2     = 600
SPEC_LOAD_W_PER_M2 = 40
DESIGN_LOAD_KW     = LIVING_AREA_M2 * SPEC_LOAD_W_PER_M2 / 1000   # = 24.0 kW

MODULES = [
    ("HP_A_12kW", 12.0),
    ("HP_B_12kW", 12.0),
    ("HP_C_6kW",   6.0),
]
MODULE_NAMES = [m for m, _ in MODULES]
CAPACITIES   = {m: c for m, c in MODULES}

TANK_START_PCT = 0.50
TANK_LOSS_PCT  = 0.01
MIN_RUNTIME_H  = 2
MIN_OFF_H      = 1
MOD_MIN        = 0.30
BALANCE_PENALTY_EUR_PER_H = 0.01

DEFICIT_PENALTY_EUR_PER_KWH = 1000.0
SPILL_PENALTY_EUR_PER_KWH   = 1000.0
WARMSTART_ENABLED = True    # NEU: war False im Tarifvergleichsskript (Debug-Rest)
SOLVER_TEE        = False   # NEU: war True im Tarifvergleichsskript (Debug-Rest)

VL_BASE       = 35
COND_APPROACH = 5
EVAP_APPROACH = 5

HEIZGRENZE_C = 15.0
NORM_TEMP_C  = -12.0

N_HOURS_REQUESTED   = 8760
SOLVER_TIME_LIMIT_S = 3600
MIP_GAP             = 0.02

KWH_PER_LITRE = 20.3 / 500
TANK_LITRES   = 3000
TANK_KWH      = round(TANK_LITRES * KWH_PER_LITRE, 1)

# ---- Warmwasser MFH (10/90-Profil mit Nachtlast, siehe Chat) ----
DHW_PEAK_SHARE    = 0.10
NIGHT_FACTOR      = 0.5
DAY_HOURS_COUNT   = 16
NIGHT_HOURS_COUNT = 8

DAILY_TW_KWH_PER_UNIT = 2500 / 365
DHW_SHOWER_PER_PEAK   = N_UNITS * DHW_PEAK_SHARE * DAILY_TW_KWH_PER_UNIT / 2
_DHW_BASE_SHARE = 1 - DHW_PEAK_SHARE
_DHW_X_DAY      = _DHW_BASE_SHARE * DAILY_TW_KWH_PER_UNIT / (DAY_HOURS_COUNT + NIGHT_HOURS_COUNT * NIGHT_FACTOR)
DHW_BASE_DAY_KW   = N_UNITS * _DHW_X_DAY
DHW_BASE_NIGHT_KW = N_UNITS * _DHW_X_DAY * NIGHT_FACTOR

def dhw_demand_kw(hour):
    d = 0.0
    if hour in (7, 19):
        d += DHW_SHOWER_PER_PEAK
    if 6 <= hour <= 21:
        d += DHW_BASE_DAY_KW
    else:
        d += DHW_BASE_NIGHT_KW
    return d

def space_demand_kw(t_amb, last_minus12_kw=DESIGN_LOAD_KW):
    """Vereinfachte lineare Heizkurve: 0 kW an der Heizgrenze, last_minus12_kw
    bei -12 C."""
    return last_minus12_kw * max(0.0, (HEIZGRENZE_C - t_amb) / (HEIZGRENZE_C - NORM_TEMP_C))

COP_CURVE = [(-15, 1.8), (-10, 2.2), (-7, 2.5), (-5, 2.7), (0, 3.1), (2, 3.3),
             (5, 3.7), (7, 4.0), (10, 4.4), (12, 4.7), (15, 5.0), (20, 5.5)]

def _cop_base_interp(temp_c):
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

def carnot_cop(t_amb_c, vl_c):
    t_cond = vl_c + COND_APPROACH + 273.15
    t_evap = t_amb_c - EVAP_APPROACH + 273.15
    return t_cond / (t_cond - t_evap)

def get_cop(temp_c, vl_c=VL_BASE):
    guetegrad = _cop_base_interp(temp_c) / carnot_cop(temp_c, VL_BASE)
    return guetegrad * carnot_cop(temp_c, vl_c)


def compute_tank(u_by_module, demands, tank_start, tank_max, loss_pct, T):
    tanks = []
    tank = tank_start
    for t in range(T):
        hp_out = sum(CAPACITIES[m] * u_by_module[m][t] for m in MODULE_NAMES)
        tank = tank * (1 - loss_pct) - demands[t] + hp_out
        tank = max(0.0, min(tank, tank_max))
        tanks.append(tank)
    return tanks


def compute_tank_with_slack(u_by_module, demands, tank_start, tank_max, loss_pct, T):
    tanks, deficits, spills = [], [], []
    tank = tank_start
    for t in range(T):
        hp_out = sum(CAPACITIES[m] * u_by_module[m][t] for m in MODULE_NAMES)
        raw = tank * (1 - loss_pct) - demands[t] + hp_out
        if raw < 0:
            deficit, spill, tank = -raw, 0.0, 0.0
        elif raw > tank_max:
            deficit, spill, tank = 0.0, raw - tank_max, tank_max
        else:
            deficit, spill, tank = 0.0, 0.0, raw
        tanks.append(tank)
        deficits.append(deficit)
        spills.append(spill)
    return tanks, deficits, spills


def build_warmstart(demands, prices, tank_max, T, cheap_thresh):
    hours_used = {m: 0 for m in MODULE_NAMES}
    hours_since_on = {m: None for m in MODULE_NAMES}
    x = {m: [0] * T for m in MODULE_NAMES}
    tank = tank_max * TANK_START_PCT
    for t in range(T):
        low   = tank < tank_max * 0.30
        cheap = prices[t] <= cheap_thresh
        ordered_t = sorted(MODULE_NAMES, key=lambda m: (CAPACITIES[m], hours_used[m]))

        on = [m for m in MODULE_NAMES
              if hours_since_on[m] is not None and hours_since_on[m] + 1 < MIN_RUNTIME_H]
        running_cap = sum(CAPACITIES[m] for m in on)

        if low or cheap:
            reserve = CAPACITIES[ordered_t[0]] if low else 0.0
            target = demands[t] + reserve
            max_allowed = tank_max - tank * (1 - TANK_LOSS_PCT) + demands[t]
            for m in ordered_t:
                if m in on:
                    continue
                if running_cap >= target:
                    break
                if running_cap + CAPACITIES[m] > max_allowed + 1e-9:
                    break
                future_tank = tank * (1 - TANK_LOSS_PCT) - demands[t] + running_cap + CAPACITIES[m]
                future_ok = True
                for k in range(1, MIN_RUNTIME_H):
                    if t + k >= T:
                        break
                    future_tank = future_tank * (1 - TANK_LOSS_PCT) - demands[t + k] + running_cap + CAPACITIES[m]
                    if future_tank > tank_max + 1e-9:
                        future_ok = False
                        break
                    future_tank = max(0.0, future_tank)
                if not future_ok:
                    continue
                on.append(m)
                running_cap += CAPACITIES[m]

        for m in MODULE_NAMES:
            if m in on:
                x[m][t] = 1
                hours_used[m] += 1
                hours_since_on[m] = 0 if hours_since_on[m] is None else hours_since_on[m] + 1
            else:
                x[m][t] = 0
                hours_since_on[m] = None
        hp_out = sum(CAPACITIES[m] * x[m][t] for m in MODULE_NAMES)
        tank = max(0.0, min(tank_max, tank * (1 - TANK_LOSS_PCT) - demands[t] + hp_out))
        if tank <= 0.01 and not on:
            emergency = min(MODULE_NAMES, key=lambda m: (CAPACITIES[m], hours_used[m]))
            x[emergency][t] = 1
            hours_used[emergency] += 1
            hours_since_on[emergency] = 0
    return x


def solve_cascade(tank_max_kwh, demands, prices, cops, T, label):
    tank_start = tank_max_kwh * TANK_START_PCT
    price_sorted = sorted(prices)
    cheap_thresh = price_sorted[int(len(price_sorted) * 0.40)]

    ws_x = build_warmstart(demands, prices, tank_max_kwh, T, cheap_thresh)
    ws_u = {m: [float(v) for v in ws_x[m]] for m in MODULE_NAMES}
    ws_tank = compute_tank(ws_u, demands, tank_start, tank_max_kwh, TANK_LOSS_PCT, T)

    cap_groups = {}
    for m in MODULE_NAMES:
        cap_groups.setdefault(CAPACITIES[m], []).append(m)
    symmetric_pairs = [(grp[i], grp[j]) for grp in cap_groups.values()
                       for i in range(len(grp)) for j in range(i + 1, len(grp))]

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T - 1)
    model.M = pyo.Set(initialize=MODULE_NAMES)

    model.demand = pyo.Param(model.T, initialize=dict(enumerate(demands)))
    model.price  = pyo.Param(model.T, initialize=dict(enumerate(prices)))
    model.cop    = pyo.Param(model.T, initialize=dict(enumerate(cops)))
    model.cap    = pyo.Param(model.M, initialize=CAPACITIES)

    model.x = pyo.Var(model.M, model.T, domain=pyo.Binary)
    model.u = pyo.Var(model.M, model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, tank_max_kwh))
    model.deficit = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.spill   = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    for m in MODULE_NAMES:
        for t in range(T):
            model.x[m, t].set_value(ws_x[m][t])
            model.u[m, t].set_value(ws_u[m][t])
    for t in range(T):
        model.s[t].set_value(max(0.0, ws_tank[t]))
        model.deficit[t].set_value(0.0)
        model.spill[t].set_value(0.0)

    if symmetric_pairs:
        model.P = pyo.RangeSet(0, len(symmetric_pairs) - 1)
        model.imbalance = pyo.Var(model.P, domain=pyo.NonNegativeReals)

        def imbalance_pos_rule(mdl, p):
            m1, m2 = symmetric_pairs[p]
            return mdl.imbalance[p] >= sum(mdl.x[m1, t] for t in mdl.T) - sum(mdl.x[m2, t] for t in mdl.T)
        def imbalance_neg_rule(mdl, p):
            m1, m2 = symmetric_pairs[p]
            return mdl.imbalance[p] >= sum(mdl.x[m2, t] for t in mdl.T) - sum(mdl.x[m1, t] for t in mdl.T)
        model.imbalance_pos = pyo.Constraint(model.P, rule=imbalance_pos_rule)
        model.imbalance_neg = pyo.Constraint(model.P, rule=imbalance_neg_rule)
        balance_term = BALANCE_PENALTY_EUR_PER_H * sum(model.imbalance[p] for p in model.P)
    else:
        balance_term = 0

    model.obj = pyo.Objective(
        expr=sum(model.u[m, t] * model.cap[m] / model.cop[t] * model.price[t] / 1000
                  for m in model.M for t in model.T) + balance_term
             + DEFICIT_PENALTY_EUR_PER_KWH * sum(model.deficit[t] for t in model.T)
             + SPILL_PENALTY_EUR_PER_KWH * sum(model.spill[t] for t in model.T),
        sense=pyo.minimize)

    def tank_balance_rule(mdl, t):
        prev = tank_start if t == 0 else mdl.s[t - 1]
        hp_out = sum(mdl.cap[m] * mdl.u[m, t] for m in mdl.M)
        return mdl.s[t] == (prev * (1 - TANK_LOSS_PCT) - mdl.demand[t] + hp_out
                             + mdl.deficit[t] - mdl.spill[t])
    model.tank_balance = pyo.Constraint(model.T, rule=tank_balance_rule)

    def mod_upper_rule(mdl, m, t):
        return mdl.u[m, t] <= mdl.x[m, t]
    def mod_lower_rule(mdl, m, t):
        return mdl.u[m, t] >= MOD_MIN * mdl.x[m, t]
    model.mod_upper = pyo.Constraint(model.M, model.T, rule=mod_upper_rule)
    model.mod_lower = pyo.Constraint(model.M, model.T, rule=mod_lower_rule)

    def min_on_rule(mdl, m, t):
        if t < 1 or t + MIN_RUNTIME_H > T:
            return pyo.Constraint.Skip
        return sum(mdl.x[m, t + k] for k in range(MIN_RUNTIME_H)) >= \
               MIN_RUNTIME_H * (mdl.x[m, t] - mdl.x[m, t - 1])
    model.min_on = pyo.Constraint(model.M, model.T, rule=min_on_rule)

    solver_manager = SolverManager()
    solver = solver_manager.get_solver("cbc")
    solver.options["seconds"] = SOLVER_TIME_LIMIT_S
    solver.options["ratio"]   = MIP_GAP
    solver.options["heurist"] = "on"

    print(f"\n  Solving {label} (bis zu {SOLVER_TIME_LIMIT_S/60:.0f} min, "
          f"MIP-Gap {MIP_GAP*100:.0f}%, {T} Stunden, "
          f"{len(MODULE_NAMES)*T} Binaervariablen)...")
    result = solver.solve(model, tee=SOLVER_TEE, warmstart=WARMSTART_ENABLED)
    status = str(result.solver.termination_condition)

    x_vals, u_vals, extraction_ok = {}, {}, True
    try:
        for m in MODULE_NAMES:
            raw_x = [pyo.value(model.x[m, t]) for t in model.T]
            raw_u = [pyo.value(model.u[m, t]) for t in model.T]
            if all(abs(v - round(v)) < 0.01 for v in raw_x):
                x_vals[m] = [round(v) for v in raw_x]
                u_vals[m] = [min(1.0, max(0.0, v)) for v in raw_u]
            else:
                x_vals[m] = ws_x[m][:]
                u_vals[m] = ws_u[m][:]
                status += " (non-integer, using warmstart)"
                extraction_ok = False
    except Exception:
        x_vals, u_vals = ws_x, ws_u
        status += " (extraction failed)"
        extraction_ok = False

    if extraction_ok:
        try:
            s_vals = [pyo.value(model.s[t]) for t in model.T]
            deficit_vals = [max(0.0, pyo.value(model.deficit[t])) for t in model.T]
            spill_vals   = [max(0.0, pyo.value(model.spill[t])) for t in model.T]
        except Exception:
            s_vals, deficit_vals, spill_vals = compute_tank_with_slack(
                u_vals, demands, tank_start, tank_max_kwh, TANK_LOSS_PCT, T)
    else:
        s_vals, deficit_vals, spill_vals = compute_tank_with_slack(
            u_vals, demands, tank_start, tank_max_kwh, TANK_LOSS_PCT, T)

    total_deficit = sum(deficit_vals)
    total_spill = sum(spill_vals)
    deficit_hours = sum(1 for v in deficit_vals if v > 1e-4)
    spill_hours = sum(1 for v in spill_vals if v > 1e-4)

    total_cost = sum(u_vals[m][t] * CAPACITIES[m] / cops[t] * prices[t] / 1000
                      for m in MODULE_NAMES for t in range(T))
    total_elec = sum(u_vals[m][t] * CAPACITIES[m] / cops[t]
                      for m in MODULE_NAMES for t in range(T))
    hp_hours_per_module = {m: sum(x_vals[m]) for m in MODULE_NAMES}
    cycles_per_module = {
        m: sum(1 for t in range(1, T) if x_vals[m][t] == 1 and x_vals[m][t - 1] == 0)
        for m in MODULE_NAMES
    }
    avg_mod_per_module = {
        m: (sum(u_vals[m]) / sum(x_vals[m]) * 100) if sum(x_vals[m]) > 0 else 0.0
        for m in MODULE_NAMES
    }
    avg_paid = total_cost / total_elec * 1000 if total_elec > 0 else 0
    min_tank = min(s_vals)

    print(f"  Status: {status}")
    print(f"  Betriebsstunden je Modul: {hp_hours_per_module}")
    print(f"  Ø-Modulation waehrend Laufzeit: "
          f"{ {m: f'{v:.0f}%' for m, v in avg_mod_per_module.items()} }")
    print(f"  Kosten: {total_cost:.2f} EUR, Ø-Preis: {avg_paid:.1f} EUR/MWh, "
          f"Zyklen: {cycles_per_module}")
    print(f"  Deficit: {total_deficit:.2f} kWh in {deficit_hours} Std. | "
          f"Spill: {total_spill:.2f} kWh in {spill_hours} Std.")

    return {
        "x_vals": x_vals, "u_vals": u_vals, "s_vals": s_vals, "hp_hours": hp_hours_per_module,
        "avg_mod": avg_mod_per_module, "cycles": cycles_per_module,
        "total_elec": total_elec, "total_cost": total_cost, "avg_paid": avg_paid,
        "min_tank": min_tank, "status": status, "tank_max": tank_max_kwh,
        "total_deficit": total_deficit, "total_spill": total_spill,
        "deficit_hours": deficit_hours, "spill_hours": spill_hours,
    }

# ================================================================
# EFH-TEIL - unveraendert aus dem Tarifvergleichsskript uebernommen
# ================================================================
HP_OUTPUT_KW    = 7.0
EFH_TANK_LITRES = 1000
EFH_TANK_KWH    = 40.6

EFH_DHW_SHOWER_PER_PEAK = (2500 * 0.50) / 365 / 2
EFH_DHW_BASE_KW         = (2500 * 0.50) / (365 * 16)

def efh_dhw_demand_kw(hour):
    d = 0.0
    if hour in (7, 19):
        d += EFH_DHW_SHOWER_PER_PEAK
    if 6 <= hour <= 21:
        d += EFH_DHW_BASE_KW
    return d

EFH_COP_CURVE = [(-15, 1.8), (-10, 2.2), (-7, 2.5), (-5, 2.7), (0, 3.1), (2, 3.3),
                 (5, 3.7), (7, 4.0), (10, 4.4), (12, 4.7), (15, 5.0), (20, 5.5)]

def efh_get_cop(temp_c):
    if temp_c <= EFH_COP_CURVE[0][0]:
        return EFH_COP_CURVE[0][1]
    if temp_c >= EFH_COP_CURVE[-1][0]:
        return EFH_COP_CURVE[-1][1]
    for i in range(len(EFH_COP_CURVE) - 1):
        t0, c0 = EFH_COP_CURVE[i]
        t1, c1 = EFH_COP_CURVE[i + 1]
        if t0 <= temp_c <= t1:
            return c0 + (temp_c - t0) / (t1 - t0) * (c1 - c0)
    return 3.0

def efh_compute_tank(u_schedule, demands, tank_start, tank_max, loss_pct):
    tanks = []
    tank = tank_start
    for t in range(len(u_schedule)):
        tank = tank * (1 - loss_pct) - demands[t] + HP_OUTPUT_KW * u_schedule[t]
        tank = max(0.0, min(tank, tank_max))
        tanks.append(tank)
    return tanks

def efh_build_warmstart(demands, prices, tank_max, T, cheap_thresh):
    x = [0] * T
    tank = tank_max * TANK_START_PCT
    hours_since_on = None
    for t in range(T):
        low = tank < tank_max * 0.30
        cheap = prices[t] <= cheap_thresh
        forced = hours_since_on is not None and hours_since_on + 1 < MIN_RUNTIME_H
        max_allowed = tank_max - tank * (1 - TANK_LOSS_PCT) + demands[t]
        can_run = HP_OUTPUT_KW <= max_allowed + 1e-9
        on = forced or (can_run and (low or cheap))
        if on:
            x[t] = 1
            hours_since_on = 0 if hours_since_on is None else hours_since_on + 1
        else:
            x[t] = 0
            hours_since_on = None
        hp_out = HP_OUTPUT_KW * x[t]
        tank = max(0.0, min(tank_max, tank * (1 - TANK_LOSS_PCT) - demands[t] + hp_out))
    return x

def solve_efh(tank_max_kwh, demands, prices, elec_per_h, cops, n_hours, label):
    T = n_hours
    tank_start = tank_max_kwh * TANK_START_PCT

    # NEU ggue. Tarifvergleichsskript (siehe Chat, "Bug C"): cheap_thresh wird
    # jetzt aus derselben Preisreihe berechnet, die auch geloest wird.
    price_sorted = sorted(prices)
    cheap_thresh = price_sorted[int(len(price_sorted) * 0.40)]
    ws_x = efh_build_warmstart(demands, prices, tank_max_kwh, T, cheap_thresh)
    ws_u = ws_x[:]
    ws_tank = efh_compute_tank(ws_u, demands, tank_start, tank_max_kwh, TANK_LOSS_PCT)

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T - 1)
    model.demand = pyo.Param(model.T, initialize=dict(enumerate(demands)))
    model.price = pyo.Param(model.T, initialize=dict(enumerate(prices)))
    model.elec_kwh = pyo.Param(model.T, initialize=dict(enumerate(elec_per_h)))

    model.x = pyo.Var(model.T, domain=pyo.Binary)
    model.u = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, tank_max_kwh))

    for t in range(T):
        model.x[t].set_value(ws_x[t])
        model.u[t].set_value(ws_u[t])
        model.s[t].set_value(max(0.0, ws_tank[t]))

    model.obj = pyo.Objective(
        expr=sum(model.u[t] * model.elec_kwh[t] * model.price[t] / 1000 for t in model.T),
        sense=pyo.minimize)

    def tank_balance_rule(m, t):
        prev = tank_start if t == 0 else m.s[t - 1]
        return m.s[t] == prev * (1 - TANK_LOSS_PCT) - m.demand[t] + HP_OUTPUT_KW * m.u[t]
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
        return sum(m.x[t + k] for k in range(MIN_RUNTIME_H)) >= \
               MIN_RUNTIME_H * (m.x[t] - m.x[t - 1])
    model.min_on = pyo.Constraint(model.T, rule=min_on_rule)

    solver_manager = SolverManager()
    solver = solver_manager.get_solver("cbc")
    solver.options["seconds"] = SOLVER_TIME_LIMIT_S
    solver.options["ratio"] = MIP_GAP
    solver.options["heurist"] = "on"

    print(f"\n  Solving {label} (bis zu {SOLVER_TIME_LIMIT_S/60:.0f} min, "
          f"MIP-Gap {MIP_GAP*100:.0f}%, {T} Stunden)...")
    result = solver.solve(model, tee=SOLVER_TEE, warmstart=WARMSTART_ENABLED)
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
        x_vals, u_vals = ws_x[:], ws_u[:]
        status += " (extraction failed)"

    s_vals = efh_compute_tank(u_vals, demands, tank_start, tank_max_kwh, TANK_LOSS_PCT)

    total_cost = sum(u_vals[t] * elec_per_h[t] * prices[t] / 1000 for t in range(T))
    total_elec = sum(u_vals[t] * elec_per_h[t] for t in range(T))
    hp_hours = sum(x_vals)
    avg_paid = total_cost / total_elec * 1000 if total_elec > 0 else 0
    min_tank = min(s_vals)

    print(f"  Status: {status}")
    print(f"  HP hours: {hp_hours}, Kosten: {total_cost:.2f} EUR, "
          f"Ø-Preis: {avg_paid:.1f} EUR/MWh")

    return {
        "x_vals": x_vals, "u_vals": u_vals, "s_vals": s_vals, "hp_hours": hp_hours,
        "total_elec": total_elec, "total_cost": total_cost, "avg_paid": avg_paid,
        "min_tank": min_tank, "status": status, "tank_max": tank_max_kwh,
    }

# ================================================================
# DATEN LADEN - keine Preisspalte mehr noetig (Normaltarif ist konstant)
# ================================================================
print("Loading data...")
with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))

N_HOURS = min(N_HOURS_REQUESTED, len(all_rows))
if N_HOURS < N_HOURS_REQUESTED:
    print(f"WARNUNG: {INPUT_FILE} enthaelt nur {len(all_rows)} Zeilen, "
          f"nicht die angeforderten {N_HOURS_REQUESTED}. Verwende N_HOURS={N_HOURS}.")
rows = all_rows[:N_HOURS]

datetimes, temps, demands_mfh, demands_efh = [], [], [], []
for r in rows:
    dt  = r["datetime"]
    h   = int(dt[11:13])
    tmp = float(r["temperature_outdoor_c"])
    demands_mfh.append(space_demand_kw(tmp) + dhw_demand_kw(h))
    demands_efh.append(float(r["heating_output_w"]) / 1000 + efh_dhw_demand_kw(h))
    datetimes.append(dt); temps.append(tmp)

cops = [get_cop(t, VL_BASE) for t in temps]
cops_efh = [efh_get_cop(t) for t in temps]
elec_per_h_efh = [HP_OUTPUT_KW / c for c in cops_efh]
dts = [datetime.strptime(d, "%Y-%m-%dT%H:%M") for d in datetimes]

# ================================================================
# NORMALTARIF - konstanter Festpreis fuer alle 8760 Stunden
# (Marktbeispiel dedizierter WP-Tarife 2026, siehe Chat/Docstring des
# Tarifvergleichsskripts fuer die Quellenangabe)
# ================================================================
NORMALTARIF_CT_PRO_KWH         = 24.0
NORMALTARIF_GRUNDPREIS_EUR_MON = 12.0

flat_price_eur_mwh = [NORMALTARIF_CT_PRO_KWH * 10.0] * N_HOURS

# ================================================================
# BEIDE GEBAEUDE LOESEN
# ================================================================
print(f"\n=== MFH, Normaltarif ({NORMALTARIF_CT_PRO_KWH:.1f} ct/kWh) ===")
result_mfh = solve_cascade(TANK_KWH, demands_mfh, flat_price_eur_mwh, cops, N_HOURS, "MFH Normaltarif")

print(f"\n=== EFH, Normaltarif ({NORMALTARIF_CT_PRO_KWH:.1f} ct/kWh) ===")
result_efh = solve_efh(EFH_TANK_KWH, demands_efh, flat_price_eur_mwh, elec_per_h_efh, cops_efh,
                        N_HOURS, "EFH Normaltarif")

# ================================================================
# JAHRESKOSTEN inkl. Grundpreis
# ================================================================
gp_jahr = 12 * NORMALTARIF_GRUNDPREIS_EUR_MON
summary = {}
for name, r in [("MFH", result_mfh), ("EFH", result_efh)]:
    gesamt = r["total_cost"] + gp_jahr
    avg_ct = r["total_cost"] / r["total_elec"] * 100 if r["total_elec"] > 0 else 0
    summary[name] = {"strom": r["total_cost"], "grundpreis": gp_jahr,
                      "gesamt": gesamt, "avg_ct": avg_ct, "elec_kwh": r["total_elec"]}

print(f"\n{'='*75}")
print(f"{'Gebaeude':10} {'Stromkosten EUR':>16} {'Grundpreis EUR':>15} {'Gesamt EUR':>12} {'Ø ct/kWh':>10}")
print(f"{'-'*75}")
for name in ["MFH", "EFH"]:
    s = summary[name]
    print(f"{name:10} {s['strom']:>16.2f} {s['grundpreis']:>15.2f} {s['gesamt']:>12.2f} {s['avg_ct']:>10.2f}")

# Diagnose: laeuft die Anlage bevorzugt bei milder Witterung? Unter einem
# Festpreis ist das die mathematisch optimale Loesung des Kaskadenmodells,
# ohne eigene Heuristik (siehe Docstring des Tarifvergleichsskripts).
on_temps_mfh = [temps[t] for t in range(N_HOURS) if any(result_mfh["x_vals"][m][t] for m in MODULE_NAMES)]
if on_temps_mfh:
    print(f"\nMFH: Ø Aussentemp. waehrend Betrieb = {sum(on_temps_mfh)/len(on_temps_mfh):.2f} °C "
          f"(Jahresmittel: {sum(temps)/len(temps):.2f} °C)")
on_temps_efh = [temps[t] for t in range(N_HOURS) if result_efh["x_vals"][t]]
if on_temps_efh:
    print(f"EFH: Ø Aussentemp. waehrend Betrieb = {sum(on_temps_efh)/len(on_temps_efh):.2f} °C "
          f"(Jahresmittel: {sum(temps)/len(temps):.2f} °C)")

# ================================================================
# PLOT: Jahreskosten beider Gebaeude
# ================================================================
fig, ax = plt.subplots(figsize=(6, 5))
names = ["MFH", "EFH"]
strom_vals = [summary[n]["strom"] for n in names]
gp_vals = [summary[n]["grundpreis"] for n in names]
ax.bar(names, strom_vals, label="Stromkosten", color="steelblue")
ax.bar(names, gp_vals, bottom=strom_vals, label="Grundpreis", color="lightgray")
for i, n in enumerate(names):
    ax.text(i, summary[n]["gesamt"] + max(strom_vals) * 0.01,
            f"{summary[n]['gesamt']:.0f} €", ha="center", fontweight="bold")
ax.set_ylabel("EUR/Jahr")
ax.set_title(f"Normaltarif ({NORMALTARIF_CT_PRO_KWH:.0f} ct/kWh + {NORMALTARIF_GRUNDPREIS_EUR_MON:.0f} EUR/Monat)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("normaltarif_kosten.png", dpi=150)
plt.show()
print("Saved: normaltarif_kosten.png")

print("\nDone!")