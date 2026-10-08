"""
Pyomo Kaskaden-Waermepumpenoptimierung - MFH "Hamburger Neubau mit Sozialwohnungen"
10 WE, 600 m² Wohnflaeche, Norm-Heizlast ~24 kW (40 W/m², KfW55-Niveau)
Kaskade: 12 + 12 + 6 kW, monovalent, freie Zuordnung der Module durch den Solver
Tank-Groessen-Sensitivitaet analog zum Einzelhaus-Skript, auf MFH-Massstab skaliert

Aenderungen ggue. der Vorversion:
- COP haengt jetzt zusaetzlich vom Vorlauf ab (Carnot-COP skaliert mit einem aus
  COP_CURVE zurueckgerechneten Guetegrad), identische Herleitung wie im
  EFH-Sensitivitaetsskript (siehe get_cop()). COP_CURVE gilt weiterhin fuer alle
  drei Module und ist jetzt explizit fuer VL_BASE=35 C kalibriert.
- VL_SCENARIOS = [35, 55]: jede aktive Tankgroesse wird jetzt fuer BEIDE
  Vorlauftemperaturen geloest (SCENARIOS kombiniert Tankgroesse x Vorlauf).
- Die vorher leere Sektion "DATEN LADEN" fehlte im Originalskript komplett
  (demands/prices/temps/datetimes/avg_market_price wurden nirgends erzeugt,
  das Skript waere so nicht lauffaehig gewesen) und wurde ergaenzt, inkl. einer
  vereinfachten linearen Heizkurve space_demand_kw() analog zum
  EFH-Sensitivitaetsskript (Heizgrenze 15 C, Bezugspunkt -12 C = DESIGN_LOAD_KW).
- Die "tiefen" Detail-Plots (Dispatch-Zoom, Beispieltage, Heatmap, Stacked-Area,
  Dauerlinie) laufen weiterhin nur fuer EIN Referenzszenario (VL_BASE=35,
  erste aktive Tankgroesse), um die Zahl der erzeugten Dateien nicht zu
  verdoppeln; die Vergleichs-Plots (Kennzahlen, Dispatch-Uebersicht, COP/Temp)
  zeigen beide Vorlauf-Szenarien.

NEU in dieser Version (V9, siehe Chat): Teillastfaehigkeit (Modulation) fuer
alle drei Kaskaden-Module, analog zum EFH-Sensitivitaetsskript. Vorherige
Version konnte jedes Modul nur 0/1 bei voller Nennleistung fahren; zusammen
mit MIN_RUNTIME_H=2 fuehrte das bei Volljahresdaten dazu, dass CBC in einer
Stunde Rechenzeit keine einzige Ganzzahlloesung verbessern konnte (LP-Bound
und gefundene Loesung lagen um den Faktor ~700 auseinander) und regelmaessig
auf Deficit/Spill (siehe V8) zurueckgreifen musste, weil ein Modul im
Volllastbetrieb den Bedarf in Schwachlaststunden zwangslaeufig ueber- bzw.
in Extremstunden unterschreiten konnte. Jedes Modul kann jetzt zwischen
MOD_MIN=30% und 100% seiner Nennleistung stufenlos fahren (model.u je Modul,
gebunden an model.x fuer die Mindestlaufzeit) - siehe solve_cascade().
Deficit/Spill bleiben als Sicherheitsnetz erhalten, sollten mit Modulation
aber nur noch in Extremfaellen (wenn ueberhaupt) groesser als 0 sein.
"""

import os
import csv
import numpy as np
import pyomo.environ as pyo
from pyomo.common.tempfiles import TempfileManager
from pyomo_windows.solvers import SolverManager
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Arbeitsverzeichnis auf den Ordner dieser Datei setzen (fehlte bisher in
# diesem Skript, siehe Hinweis in den anderen Skripten) - liest/schreibt
# garantiert im selben Ordner wie das Skript liegt.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# WICHTIG (Windows-Bugfix): Pyomo legt die temporaere Warmstart-Datei fuer CBC
# standardmaessig im System-Temp-Ordner ab. Liegt der auf einem anderen
# Laufwerk als das Arbeitsverzeichnis, entfernt Pyomo den Laufwerksbuchstaben
# NICHT aus dem Dateipfad (CBC kommt damit nachweislich nicht klar, siehe
# https://github.com/coin-or/Cbc/issues/32) und die Warmstart-Datei wird von
# CBC still ignoriert - das Log zeigt dann keinen '-mipstart'-Eintrag mehr,
# CBC startet komplett kalt und findet bei grossen Kaskaden-Modellen oft gar
# keine ganzzahlige Loesung mehr (Status 'intermediateNonInteger' statt einer
# echten Loesung). Indem die Temp-Datei explizit in denselben Ordner wie das
# Skript gelegt wird (siehe os.chdir oben), landet sie garantiert auf
# demselben Laufwerk, und Pyomos eigener Workaround greift zuverlaessig.
TempfileManager.tempdir = os.getcwd()

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
# NEU (V9, siehe Chat): Teillast-Untergrenze je Modul, identischer Wert und
# identische Modellierung wie MOD_MIN im EFH-Sensitivitaetsskript (model.u
# zwischen MOD_MIN*x und x). Gilt einheitlich fuer alle drei Kaskaden-Module.
MOD_MIN = 0.30
BALANCE_PENALTY_EUR_PER_H = 0.01   # kleine Strafe je Stunde Laufzeitdifferenz
                                     # zwischen baugleichen Modulen (Wechselbetrieb, siehe solve_cascade())

# Schlupfvariablen fuer die Tankbilanz (seit V8, siehe Chat), analog zu
# model.deficit im EFH-Sensitivitaetsskript. In V8 (nur 0/1-Volllastbetrieb je
# Modul) waren diese noch regelmaessig aktiv, weil ein zwangslaeufig volles
# Modul den Bedarf in Schwachlaststunden ueberschreiten und den Tank
# ueberlaufen lassen konnte. Mit der Teillastfaehigkeit (MOD_MIN, siehe oben)
# ist die kleinste erzwungene Ueberschussmenge pro Modul jetzt viel kleiner
# (z.B. 0.30*6kW=1.8 kW statt 6 kW beim kleinsten Modul), daher sollten
# deficit/spill jetzt nur noch als Sicherheitsnetz fuer echte Extremfaelle
# dienen (z.B. wenn Bedarf kurzzeitig ueber die installierten 30 kW steigt)
# und im Normalfall nahe 0 bleiben. Beide bleiben trotzdem stark bestraft,
# damit sie nur im echten Notfall greifen (siehe Methodik-Dokument, Abschnitt
# Kaskadenmodell / Limitationen).
DEFICIT_PENALTY_EUR_PER_KWH = 1000.0
SPILL_PENALTY_EUR_PER_KWH   = 1000.0
WARMSTART_ENABLED = False   # auf False setzen, um CBC ganz ohne Startloesung
                             # testen zu lassen (Fehlersuche, siehe Chat)
SOLVER_TEE        = True  # auf True setzen fuer die rohe CBC-Konsolenausgabe

# ---- Vorlauftemperatur-Szenarien (NEU, analog zum EFH-Sensitivitaetsskript) ----
VL_SCENARIOS  = [35, 55]    # C: Fussbodenheizung vs. Heizkoerper
VL_BASE       = 35          # C, Vorlauf, fuer den COP_CURVE unten kalibriert ist
COND_APPROACH = 5           # K, Kondensator-Graedigkeit
EVAP_APPROACH = 5           # K, Verdampfer-Graedigkeit (Luft-WP)

# ---- Vereinfachte lineare Heizkurve fuer die (fehlende) Bedarfsherleitung ----
HEIZGRENZE_C = 15.0          # C, oberhalb keine Heizlast
NORM_TEMP_C  = -12.0         # C, Bezugspunkt fuer DESIGN_LOAD_KW

# ---- Volljahr-Konfiguration (analog zur EFH-Version) ----
N_HOURS_REQUESTED   = 8760     # Ziel: Volljahr. Wird ggf. auf verfuegbare Daten begrenzt.
SOLVER_TIME_LIMIT_S = 3600     # 1h pro Szenario (statt 300s im 2-Wochen-Fall)
MIP_GAP              = 0.02    # 2% statt 1% - mit 3 Modulen x 8760h noch groesser als beim EFH
RUN_ALL_TANK_SIZES   = True   # False = nur EINE Tankgroesse (Machbarkeitstest zuerst)
SINGLE_TANK_INDEX    = 1       # Index in TANK_SIZES, falls RUN_ALL_TANK_SIZES=False (1 = 3000L)

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
# Begruendung Skalierung: Die DIN-4708-Gleichzeitigkeitsfaktoren gelten fuer
# Spitzenlast auf Minuten-/Sekundenbasis (Auslegung Durchlauferhitzer/Rohrnetz).
# Auf der hier verwendeten Stundenaufloesung verteilt sich der Duschbedarf
# mehrerer Wohnungen ohnehin ueber die Stunde, eine zusaetzliche
# Diversitaets-Reduktion ist auf diesem Zeitraster nicht sachgerecht begruendbar.
#
# Tagesprofil-Anpassung (NEU): Die urspruengliche 50%/50%-Aufteilung
# (Duschspitze/Grundlast, Grundlast nur 6-21 Uhr) war fuer einen einzelnen
# Haushalt gedacht und fuehrte bei linearer Skalierung auf N_UNITS=10 zu einer
# Spitze von >19 kW zu den Duschzeiten - auch im Hochsommer ganz ohne
# Heizbedarf, was dort taeglich zwei Module allein fuers Duschen erzwang
# (184 von 2208 Sommerstunden > 18 kW, siehe Analyse). Realistischer fuer ein
# Gebaeude mit zeitlich gestreuten Duschzeiten: nur 10 % des Tagesbedarfs als
# Spitze, die restlichen 90 % als Grundlast ueber 24 h verteilt, mit einer
# Nachtabsenkung (22-6 Uhr) auf die Haelfte der Tagesrate. Das
# Jahresgesamtbudget (10 x 2500 kWh/a) bleibt unveraendert, nur die Tagesform
# wird geglaettet.
# ================================================================
DHW_PEAK_SHARE    = 0.10   # Anteil des Tagesbedarfs in den beiden Spitzenstunden (vorher 0.50)
NIGHT_FACTOR      = 0.5    # Nachtgrundlast als Anteil der Taggrundlast
DAY_HOURS_COUNT   = 16     # 6:00-21:00
NIGHT_HOURS_COUNT = 8      # 22:00-5:00

DAILY_TW_KWH_PER_UNIT = 2500 / 365   # taeglicher TWW-Bedarf pro Wohneinheit [kWh/d]
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
    bei -12 C. Identische Herleitung wie im EFH-Sensitivitaetsskript."""
    return last_minus12_kw * max(0.0, (HEIZGRENZE_C - t_amb) / (HEIZGRENZE_C - NORM_TEMP_C))

# ================================================================
# COP-KENNLINIE (Stuetzpunkte unveraendert aus dem Einzelhaus-Skript
# uebernommen, gilt fuer alle drei Module). Gilt fuer VL_BASE=35 C; fuer
# andere Vorlauftemperaturen wird sie ueber Carnot + Guetegrad umgerechnet,
# siehe get_cop().
# ================================================================
COP_CURVE = [(-15, 1.8), (-10, 2.2), (-7, 2.5), (-5, 2.7), (0, 3.1), (2, 3.3),
             (5, 3.7), (7, 4.0), (10, 4.4), (12, 4.7), (15, 5.0), (20, 5.5)]

def _cop_base_interp(temp_c):
    """Bisherige get_cop()-Logik: lineare Interpolation der Stuetzpunkte, gilt
    fuer VL_BASE."""
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
    """Ideale (Carnot-)Leistungszahl zwischen Verdampfer- und Kondensatortemperatur."""
    t_cond = vl_c + COND_APPROACH + 273.15
    t_evap = t_amb_c - EVAP_APPROACH + 273.15
    return t_cond / (t_cond - t_evap)

def get_cop(temp_c, vl_c=VL_BASE):
    """COP bei Aussentemperatur temp_c und Vorlauf vl_c. Guetegrad wird aus der
    kalibrierten COP_CURVE (VL_BASE) zurueckgerechnet und auf den gewuenschten
    Vorlauf uebertragen. Identisch zur Herleitung im EFH-Sensitivitaetsskript
    (vgl. Methodik-Dokument, Abschnitt "Vorlauftemperaturabhaengigkeit des COP")."""
    guetegrad = _cop_base_interp(temp_c) / carnot_cop(temp_c, VL_BASE)
    return guetegrad * carnot_cop(temp_c, vl_c)


def compute_tank(u_by_module, demands, tank_start, tank_max, loss_pct, T):
    """u_by_module: dict[module_name] -> list[float in [0,1]] length T
    (Modulationsgrad je Modul; 0/1-Werte wie bisher sind ein Spezialfall)."""
    tanks = []
    tank = tank_start
    for t in range(T):
        hp_out = sum(CAPACITIES[m] * u_by_module[m][t] for m in MODULE_NAMES)
        tank = tank * (1 - loss_pct) - demands[t] + hp_out
        tank = max(0.0, min(tank, tank_max))
        tanks.append(tank)
    return tanks


def compute_tank_with_slack(u_by_module, demands, tank_start, tank_max, loss_pct, T):
    """Wie compute_tank(), berechnet aber zusaetzlich explizit, wieviel
    Deficit/Spill der gegebene Fahrplan (u_by_module) gebraucht haette, statt
    das stillschweigend wegzuclippen. Wird als Fallback verwendet, wenn CBC
    keine Ganzzahlloesung liefert (Bugfix, siehe Chat: vorher wurden Deficit/
    Spill in diesem Fall faelschlich als 0 ausgewiesen, selbst wenn der
    verwendete Warmstart-Fahrplan sie eigentlich gebraucht haette)."""
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
    """Heuristischer MIP-Start: laedt den Speicher in guenstigen Stunden,
    schaltet im Notfall (Tank nahe leer) das kleinste Modul zwingend zu.
    Dient nur als Startloesung fuer den Solver, keine Optimalitaetsgarantie.

    Wechselbetrieb: bei baugleichen Modulen (gleiche Kapazitaet, z.B. den
    beiden 12-kW-Einheiten) wird nicht immer dasselbe Modul zuerst gewaehlt,
    sondern das bis dahin am wenigsten genutzte.

    Mindestlaufzeit (NEU, Bugfix): ein Modul, das einschaltet, wird jetzt
    zwingend fuer mindestens MIN_RUNTIME_H Stunden weiterlaufen gelassen,
    auch wenn Preis/Tankstand in der Zwischenzeit ein Abschalten nahelegen
    wuerden. Ohne das verletzten in einem Test 70,2% aller Einschaltvorgaenge
    die Mindestlaufzeit-Nebenbedingung des Modells (min_on_rule) - CBC konnte
    aus einem so inkonsistenten Fahrplan keine gueltige Startloesung bauen
    ('mipstart values could not be used to build a solution', siehe
    Chat-Verlauf), selbst nachdem der Ueberlauf-Fehler bereits behoben war."""
    hours_used = {m: 0 for m in MODULE_NAMES}
    hours_since_on = {m: None for m in MODULE_NAMES}  # None = aus; sonst Anzahl bereits gelaufener Stunden
    x = {m: [0] * T for m in MODULE_NAMES}
    tank = tank_max * TANK_START_PCT
    for t in range(T):
        low   = tank < tank_max * 0.30
        cheap = prices[t] <= cheap_thresh
        ordered_t = sorted(MODULE_NAMES, key=lambda m: (CAPACITIES[m], hours_used[m]))

        # Module, die ihre Mindestlaufzeit noch nicht erreicht haben, MUESSEN
        # weiterlaufen - unabhaengig von Preis/Tankstand.
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
                    break  # naechst-groessere Module passen erst recht nicht mehr
                # Vorausschau (NEU, Bugfix): ein NEU gestartetes Modul laeuft
                # zwingend MIN_RUNTIME_H Stunden weiter (siehe oben). Reicht
                # der Tank auch fuer die folgende(n) Zwangsstunde(n) noch,
                # bei bekanntem zukuenftigem Bedarf und sonst unveraenderter
                # Fahrweise? Ohne diese Pruefung blieben vereinzelt (bei einem
                # Test: 33 von 8760 Stunden) Ueberlauf-Faelle uebrig, die CBC
                # nicht mehr reparieren konnte ('mipstart values could not be
                # used to build a solution').
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
            x[emergency][t] = 1  # Notfall: kleinstes, am wenigsten genutztes Modul erzwingen
            hours_used[emergency] += 1
            hours_since_on[emergency] = 0
    return x


def save_schedule(x_vals, T, path):
    """Speichert einen geloesten (oder auch nur warmgestarteten) Fahrplan als
    CSV, damit er als Startloesung fuer einen spaeteren, laenger laufenden
    Versuch wiederverwendet werden kann (siehe load_schedule())."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t"] + MODULE_NAMES)
        for t in range(T):
            w.writerow([t] + [x_vals[m][t] for m in MODULE_NAMES])


def load_schedule(path, T):
    """Laedt einen zuvor mit save_schedule() gespeicherten Fahrplan. Gibt None
    zurueck, wenn die Datei fehlt oder nicht zur aktuellen Laenge T passt."""
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != T:
        print(f"  Hinweis: {path} hat {len(rows)} Zeilen, erwartet {T} - "
              f"wird ignoriert (vermutlich anderer Zeitraum).")
        return None
    x = {m: [int(r[m]) for r in rows] for m in MODULE_NAMES}
    return x


def solve_cascade(tank_max_kwh, demands, prices, cops, T, label, resume_file=None):
    tank_start = tank_max_kwh * TANK_START_PCT
    price_sorted = sorted(prices)
    cheap_thresh = price_sorted[int(len(price_sorted) * 0.40)]

    ws_x = load_schedule(resume_file, T) if resume_file else None
    if ws_x is not None:
        print(f"  Verwende gespeicherten Fahrplan aus {resume_file} als Startloesung.")
    else:
        ws_x = build_warmstart(demands, prices, tank_max_kwh, T, cheap_thresh)
    # NEU (V9): build_warmstart() liefert weiterhin nur 0/1 (kennt keine
    # Modulation). Als Start-Modulationsgrad wird "voll an, wenn x=1" verwendet
    # (ws_u = ws_x als Float) - eine einfache, aber zulaessige Anfangsloesung
    # (erfuellt u<=x und u>=MOD_MIN*x trivial), von der aus der Solver die
    # tatsaechliche Teillast optimiert.
    ws_u = {m: [float(v) for v in ws_x[m]] for m in MODULE_NAMES}
    ws_tank = compute_tank(ws_u, demands, tank_start, tank_max_kwh, TANK_LOSS_PCT, T)

    # Symmetrische Modulgruppen (gleiche Kapazitaet, z.B. die beiden 12-kW-
    # Module) fuer den Wechselbetriebs-Ausgleich in der Zielfunktion.
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
    # NEU (V9): Modulationsgrad je Modul und Stunde, 0-1 (Anteil der Nennleistung).
    # Gebunden an model.x ueber mod_lower/mod_upper weiter unten - identische
    # Modellierung wie model.u im EFH-Sensitivitaetsskript, hier je Modul.
    model.u = pyo.Var(model.M, model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, tank_max_kwh))
    # Schlupfvariablen der Tankbilanz (seit V8), siehe Kommentar bei
    # DEFICIT_PENALTY_EUR_PER_KWH/SPILL_PENALTY_EUR_PER_KWH weiter oben.
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

    # NEU: Wechselbetriebs-Ausgleich fuer baugleiche Module. imbalance[p] >=
    # |Betriebsstunden Modul1 - Betriebsstunden Modul2| je symmetrischem Paar,
    # mit kleiner Strafe in der Zielfunktion (BALANCE_PENALTY_EUR_PER_H). Das
    # bricht die im Methodik-Dokument beschriebene Symmetrie zugunsten
    # gleichmaessigen Verschleisses, ohne die eigentliche Kostenoptimierung
    # spuerbar zu verzerren (bei z.B. 1000h Ungleichgewicht nur ~10 EUR
    # Strafe gegenueber Jahreskosten in der Groessenordnung 2000-3000 EUR).
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

    # NEU (V9): Elektrischer Verbrauch (und damit Kosten) skaliert jetzt mit dem
    # Modulationsgrad u statt mit dem reinen An/Aus-Zustand x - ein Modul bei
    # 50% Teillast verbraucht auch nur ~50% des Volllast-Stroms (gleicher COP
    # angenommen, unabhaengig vom Lastpunkt - identische Vereinfachung wie im
    # EFH-Sensitivitaetsskript).
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

    # NEU (V9): Modulationsgrenzen je Modul - aus (x=0) erzwingt u=0, an (x=1)
    # erlaubt u zwischen MOD_MIN und 1. Mindestlaufzeit (min_on_rule) bezieht
    # sich weiterhin auf x (An/Aus), nicht auf die Lasthoehe.
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

    # Keine explizite Mindeststillstands-Nebenbedingung: bei stuendlicher
    # Aufloesung ist eine Mindeststillstandszeit von MIN_OFF_H = 1h automatisch
    # erfuellt (ein Modul kann fruehestens eine Stunde nach dem Abschalten
    # wieder anspringen, das gibt das Zeitraster ohnehin vor). Die zuvor hier
    # stehende Bedingung x[t-1]-x[t]+x[t+1]<=1 erzwang unbeabsichtigt 2h statt
    # 1h Mindeststillstand und wurde daher entfernt (siehe Methodik-Dokument,
    # Abschnitt Kaskadenmodell).

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

    # Falls der Solver abgebrochen wurde: berichten, was ueberhaupt an Schranken
    # bekannt ist, statt "wie weit vom Optimum" unbeantwortet zu lassen.
    try:
        lb = result.problem.lower_bound
        ub = result.problem.upper_bound
        if lb is not None and ub not in (None, 0):
            gap_pct = abs(ub - lb) / abs(ub) * 100
            print(f"  Bekannte Schranken: unten={lb:.2f}, oben={ub:.2f} -> Gap ~{gap_pct:.1f}% "
                  f"(nur ein Anhaltswert, keine exakte Garantie bei abgebrochenem Solve).")
        else:
            print("  Keine verwertbaren Schranken vom Solver erhalten (Status "
                  f"'{status}') - Gap zum Optimum unbekannt.")
    except Exception:
        print("  Schranken konnten nicht ausgelesen werden.")

    # NEU (V9): x (An/Aus) UND u (Modulationsgrad) werden jetzt gemeinsam
    # extrahiert. Wenn x nicht (nahezu) ganzzahlig ist - der Solver also keine
    # brauchbare Ganzzahlloesung geliefert hat - fallen BEIDE, x und u,
    # gemeinsam auf den Warmstart-Fahrplan zurueck (ein "geliehenes" u zu einem
    # frisch extrahierten x waere inkonsistent).
    x_vals = {}
    u_vals = {}
    extraction_ok = True
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
        x_vals = ws_x
        u_vals = ws_u
        status += " (extraction failed)"
        extraction_ok = False

    if resume_file:
        save_schedule(x_vals, T, resume_file)
        print(f"  Fahrplan gespeichert in {resume_file} (Basis fuer einen spaeteren, "
              f"laenger laufenden Versuch).")

    # s_vals aus den tatsaechlich geloesten Modellvariablen (inkl. deficit/
    # spill) statt aus compute_tank(u_vals, ...) neu berechnet - compute_tank()
    # clippt Tankwerte stillschweigend auf [0, tank_max] und wuerde damit einen
    # etwaigen deficit-/spill-Einsatz unsichtbar machen. Nur wenn die Extraktion
    # fehlschlug, wird auf compute_tank_with_slack() mit dem Warmstart-Fahrplan
    # zurueckgefallen - die berechnet deficit/spill fuer diesen Fallback-Fahrplan
    # jetzt explizit mit (Bugfix ggue. V8, siehe Chat: dort wurden im
    # Fallback-Fall faelschlich immer 0 ausgewiesen).
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

    # NEU (V9): Kosten/Stromverbrauch jetzt ueber u (Modulationsgrad) statt x.
    total_cost = sum(u_vals[m][t] * CAPACITIES[m] / cops[t] * prices[t] / 1000
                      for m in MODULE_NAMES for t in range(T))
    total_elec = sum(u_vals[m][t] * CAPACITIES[m] / cops[t]
                      for m in MODULE_NAMES for t in range(T))
    hp_hours_per_module = {m: sum(x_vals[m]) for m in MODULE_NAMES}
    cycles_per_module = {
        m: sum(1 for t in range(1, T) if x_vals[m][t] == 1 and x_vals[m][t - 1] == 0)
        for m in MODULE_NAMES
    }
    # NEU (V9): mittlerer Modulationsgrad waehrend der Laufzeit je Modul - zeigt,
    # wie oft die Teillastfaehigkeit tatsaechlich genutzt wird (100% = Modul
    # laeuft nie im Teillastbetrieb, MOD_MIN*100% = immer am unteren Anschlag).
    avg_mod_per_module = {
        m: (sum(u_vals[m]) / sum(x_vals[m]) * 100) if sum(x_vals[m]) > 0 else 0.0
        for m in MODULE_NAMES
    }
    avg_paid = total_cost / total_elec * 1000 if total_elec > 0 else 0
    min_tank = min(s_vals)

    print(f"  Status: {status}")
    print(f"  Module-Betriebsstunden: {hp_hours_per_module}")
    print(f"  Ø-Modulation waehrend Laufzeit: "
          f"{ {m: f'{v:.0f}%' for m, v in avg_mod_per_module.items()} }")
    print(f"  Kosten: {total_cost:.2f} EUR, Ø-Preis: {avg_paid:.1f} EUR/MWh, "
          f"Zyklen: {cycles_per_module}")
    print(f"  Deficit: {total_deficit:.2f} kWh in {deficit_hours} Std. | "
          f"Spill: {total_spill:.2f} kWh in {spill_hours} Std. "
          f"(sollten mit Modulation nahe 0 sein - siehe Kommentar bei DEFICIT_PENALTY_EUR_PER_KWH)")

    return {
        "x_vals": x_vals, "u_vals": u_vals, "s_vals": s_vals, "hp_hours": hp_hours_per_module,
        "avg_mod": avg_mod_per_module,
        "total_elec": total_elec, "total_cost": total_cost, "avg_paid": avg_paid,
        "cycles": cycles_per_module, "min_tank": min_tank, "status": status,
        "tank_max": tank_max_kwh,
        "total_deficit": total_deficit, "total_spill": total_spill,
        "deficit_hours": deficit_hours, "spill_hours": spill_hours,
    }


# ================================================================
# DATEN LADEN (fehlte im Originalskript komplett - ergaenzt)
# ================================================================
print("Loading data...")
with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))

N_HOURS = min(N_HOURS_REQUESTED, len(all_rows))
if N_HOURS < N_HOURS_REQUESTED:
    print(f"WARNUNG: {INPUT_FILE} enthaelt nur {len(all_rows)} Zeilen, "
          f"nicht die angeforderten {N_HOURS_REQUESTED}. Verwende N_HOURS={N_HOURS}.")
rows = all_rows[:N_HOURS]

valid_prices = [float(r["electricity_price_eur_per_mwh"])
                for r in all_rows if r["electricity_price_eur_per_mwh"] != ""]
avg_market_price = sum(valid_prices) / len(valid_prices)

datetimes, temps, demands, prices = [], [], [], []
for r in rows:
    dt  = r["datetime"]
    h   = int(dt[11:13])
    tmp = float(r["temperature_outdoor_c"])
    dem = space_demand_kw(tmp) + dhw_demand_kw(h)
    p   = float(r["electricity_price_eur_per_mwh"]) if r["electricity_price_eur_per_mwh"] != "" else avg_market_price
    datetimes.append(dt); temps.append(tmp); demands.append(dem); prices.append(p)

# COP/Strombedarf jetzt je Vorlauf-Szenario (get_cop haengt jetzt von vl ab)
cops_by_vl = {vl: [get_cop(t, vl) for t in temps] for vl in VL_SCENARIOS}

# ================================================================
# SZENARIEN AUSWAEHLEN UND LOESEN - Tankgroesse(n) x Vorlauf-Szenario(en)
# (Machbarkeitstest: 1 Tankgroesse per Default, jetzt IMMER beide Vorlaeufe)
# ================================================================
TANK_SIZES_ACTIVE = TANK_SIZES if RUN_ALL_TANK_SIZES else [TANK_SIZES[SINGLE_TANK_INDEX]]

SCENARIOS = []
for tank_kwh, tank_litres, tank_label in TANK_SIZES_ACTIVE:
    for vl in VL_SCENARIOS:
        SCENARIOS.append({
            "tank_kwh": tank_kwh, "tank_litres": tank_litres, "vl": vl,
            "label": f"{tank_label} | VL{vl}",
        })

worst_case_min = len(SCENARIOS) * SOLVER_TIME_LIMIT_S / 60
print(f"\n{len(SCENARIOS)} Szenario(en) ausgewaehlt "
      f"({', '.join(s['label'] for s in SCENARIOS)}), "
      f"Zeitbudget bis zu {worst_case_min:.0f} min insgesamt.")
if not RUN_ALL_TANK_SIZES:
    print("Hinweis: RUN_ALL_TANK_SIZES=True setzen, um alle vier Tankgroessen "
          "(je Vorlauf) zu rechnen, sobald die Machbarkeit bestaetigt ist.")

results = {}
for sc in SCENARIOS:
    safe_name = sc["label"].replace(" ", "").replace("|", "_").replace("(", "").replace(")", "")
    resume_file = f"schedule_{safe_name}.csv"
    results[sc["label"]] = solve_cascade(
        sc["tank_kwh"], demands, prices, cops_by_vl[sc["vl"]], N_HOURS, sc["label"],
        resume_file=resume_file)

# ================================================================
# VERGLEICHSTABELLE
# ================================================================
print(f"\n{'='*130}")
print(f"{'Szenario':26} {'Kosten EUR':>10} {'Ø EUR/MWh':>10} {'Min.Tank':>9} "
      f"{'Deficit kWh':>12} {'Spill kWh':>10}  Betriebsstunden je Modul")
print(f"{'-'*130}")
for sc in SCENARIOS:
    r = results[sc["label"]]
    hrs = ", ".join(f"{m}:{h}" for m, h in r["hp_hours"].items())
    print(f"{sc['label']:26} {r['total_cost']:>10.2f} {r['avg_paid']:>10.2f} {r['min_tank']:>9.2f} "
          f"{r['total_deficit']:>12.2f} {r['total_spill']:>10.2f}  {hrs}")
print(f"\nMarktdurchschnittspreis: {avg_market_price:.2f} EUR/MWh")
print("Hinweis: Deficit/Spill > 0 bedeutet, dass selbst mit Teillastbetrieb "
      "(MOD_MIN..100% je Modul) der Bedarf in einzelnen Stunden nicht exakt getroffen "
      "werden konnte (z.B. Bedarf kurzzeitig ausserhalb von [MOD_MIN*kleinstes Modul, "
      "installierte 30 kW]) - siehe Kommentar bei DEFICIT_PENALTY_EUR_PER_KWH weiter oben im Skript.")
print(f"MFH-Auslegungslast: {DESIGN_LOAD_KW:.1f} kW | Kaskade installiert: "
      f"{sum(CAPACITIES.values()):.0f} kW ({', '.join(f'{m}={c}kW' for m,c in MODULES)})")

dts = [datetime.strptime(d, "%Y-%m-%dT%H:%M") for d in datetimes]
colors = {"HP_A_12kW": "seagreen", "HP_B_12kW": "darkorange", "HP_C_6kW": "tomato"}

def format_date_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

# ================================================================
# PLOT 1: Tanklevel + Modul-Dispatch, EIN Plot je Vorlauf-Szenario
# ================================================================
for sc in SCENARIOS:
    r = results[sc["label"]]
    tank_max = sc["tank_kwh"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1.4, 1]})
    fig.suptitle(f"Kaskaden-Fahrweise - {sc['label']} ({N_HOURS}h)", fontsize=12, fontweight="bold")

    axes[0].plot(dts, prices, color="gray", linewidth=0.6)
    axes[0].set_ylabel("Preis\n[EUR/MWh]", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    tank_pct = [v / tank_max * 100 for v in r["s_vals"]]
    axes[1].plot(dts, tank_pct, color="steelblue", linewidth=0.6)
    axes[1].fill_between(dts, tank_pct, alpha=0.15, color="steelblue")
    axes[1].axhline(100, color="gray", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Tank [% voll]", fontsize=8)
    axes[1].set_ylim(-5, 115)
    axes[1].grid(True, alpha=0.25)

    # NEU (V9): zeigt jetzt die tatsaechliche modulierte Leistung je Modul
    # (CAPACITIES[m]*u_vals) statt nur An/Aus, da Zwischenwerte jetzt moeglich sind.
    bottom = [0.0] * N_HOURS
    for m in MODULE_NAMES:
        level = [bottom[t] + r["u_vals"][m][t] * CAPACITIES[m] for t in range(N_HOURS)]
        axes[2].fill_between(dts, bottom, level, step="post", color=colors[m], alpha=0.8, label=m)
        bottom = level
    axes[2].set_ylabel("Leistung\n[kW]", fontsize=8)
    axes[2].legend(fontsize=7, ncol=3, loc="upper right")
    axes[2].set_xlabel("Datum")
    format_date_axis(axes[2])
    plt.tight_layout()
    fname = f"cascade_dispatch_VL{sc['vl']}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

# ================================================================
# PLOT 2: Tank-Fuellstand - Vergleich ALLER Szenarien (Tankgroesse x Vorlauf)
# ================================================================
colors_tank = ["steelblue", "seagreen", "darkorange", "tomato",
               "purple", "brown", "olive", "gray"][:len(SCENARIOS)]
fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(14, 3 * len(SCENARIOS)),
                          sharex=True, squeeze=False)
axes = axes[:, 0]
fig.suptitle("Pufferspeicher-Fuellstand - Vergleich der Szenarien",
             fontsize=12, fontweight="bold")
for i, sc in enumerate(SCENARIOS):
    ri = results[sc["label"]]
    ax = axes[i]
    tank_pct = [v / sc["tank_kwh"] * 100 for v in ri["s_vals"]]
    ax.plot(dts, tank_pct, color=colors_tank[i], linewidth=0.6)
    ax.fill_between(dts, tank_pct, alpha=0.15, color=colors_tank[i])
    ax.axhline(100, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0, color="tomato", linestyle="--", linewidth=0.8)
    ax.set_ylabel(f"{sc['label']}\n[% voll]", fontsize=8)
    ax.set_ylim(-5, 115)
    ax.grid(True, alpha=0.25)
format_date_axis(axes[-1])
axes[-1].set_xlabel("Datum")
plt.tight_layout()
plt.savefig("cascade_tank_levels_compare.png", dpi=150)
plt.show()
print("Saved: cascade_tank_levels_compare.png")

# ================================================================
# PLOT 3: Kennzahlenvergleich nach Szenario (Kosten, Preis, Min-Tank, Zyklen)
# ================================================================
short_labels = [sc["label"] for sc in SCENARIOS]
bar_colors   = ["seagreen" if sc["vl"] == 35 else "tomato" for sc in SCENARIOS]
costs  = [results[sc["label"]]["total_cost"] for sc in SCENARIOS]
avgp   = [results[sc["label"]]["avg_paid"]   for sc in SCENARIOS]
mint   = [results[sc["label"]]["min_tank"]   for sc in SCENARIOS]
totcyc = [sum(results[sc["label"]]["cycles"].values()) for sc in SCENARIOS]

def add_values(ax, bars, fmt="{:.1f}"):
    top = max((b.get_height() for b in bars), default=1)
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01 * max(top, 1e-6),
                 fmt.format(v), ha="center", va="bottom", fontsize=8)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Kaskade - Kennzahlenvergleich nach Szenario ({N_HOURS}h)",
             fontsize=12, fontweight="bold")

ax = axes[0, 0]
bars = ax.bar(short_labels, costs, color=bar_colors, alpha=0.85)
add_values(ax, bars, "{:.2f}")
ax.set_ylabel("EUR"); ax.set_title("Gesamtkosten"); ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(axis="x", rotation=20, labelsize=8)

ax = axes[0, 1]
bars = ax.bar(short_labels, avgp, color=bar_colors, alpha=0.85)
ax.axhline(avg_market_price, color="navy", linestyle="--",
           label=f"Marktdurchschnitt ({avg_market_price:.0f})")
add_values(ax, bars, "{:.1f}")
ax.set_ylabel("EUR/MWh"); ax.set_title("Ø bezahlter Preis")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(axis="x", rotation=20, labelsize=8)

ax = axes[1, 0]
bars = ax.bar(short_labels, mint, color=bar_colors, alpha=0.85)
add_values(ax, bars, "{:.2f}")
ax.set_ylabel("kWh"); ax.set_title("Minimaler Tankstand"); ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(axis="x", rotation=20, labelsize=8)

ax = axes[1, 1]
bars = ax.bar(short_labels, totcyc, color=bar_colors, alpha=0.85)
add_values(ax, bars, "{:.0f}")
ax.set_ylabel("Zyklen")
ax.set_title("Start/Stopp-Zyklen (alle Module)"); ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(axis="x", rotation=20, labelsize=8)

plt.tight_layout()
plt.savefig("cascade_summary_compare.png", dpi=150)
plt.show()
print("Saved: cascade_summary_compare.png")

# ================================================================
# PLOT 4: Aussentemperatur und COP je Vorlauf-Szenario
# ================================================================
fig, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(dts, temps, color="steelblue", linewidth=0.6, label="Außentemperatur")
ax1.set_ylabel("Temperatur [°C]", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax2 = ax1.twinx()
for i, vl in enumerate(VL_SCENARIOS):
    ax2.plot(dts, cops_by_vl[vl], linewidth=0.6, color=["seagreen", "tomato"][i % 2],
             label=f"COP (VL{vl})")
ax2.set_ylabel("COP")
ax1.set_title("Außentemperatur und resultierender COP je Vorlauf-Szenario", fontsize=11, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
format_date_axis(ax1)
ax1.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("cascade_cop_temp.png", dpi=150)
plt.show()
print("Saved: cascade_cop_temp.png")

# ================================================================
# Ab hier: Detail-Plots nur fuer EIN Referenzszenario (VL_BASE=35, erste
# aktive Tankgroesse), um die Zahl der Dateien nicht zu verdoppeln. Fuer die
# VL55-Version dieser Plots reference_scenario entsprechend auf das
# SCENARIOS-Element mit vl=55 setzen und den Block erneut ausfuehren.
# ================================================================
reference_scenario = SCENARIOS[0]
best_label = reference_scenario["label"]
r = results[best_label]
tank_max = reference_scenario["tank_kwh"]
best_x = r["x_vals"]
best_u = r["u_vals"]
cops = cops_by_vl[reference_scenario["vl"]]
print(f"\nHinweis: Die folgenden Detail-Plots (5-11) beziehen sich auf das "
      f"Referenzszenario '{best_label}'.")

# ================================================================
# PLOT 5: Preis-Dauerlinie - wann laeuft die Kaskade?
# ================================================================
running_flags = [1 if any(best_x[m][t] for m in MODULE_NAMES) else 0
                 for t in range(N_HOURS)]
sorted_idx = sorted(range(N_HOURS), key=lambda t: prices[t])
sorted_prices = [prices[t] for t in sorted_idx]
sorted_running = [running_flags[t] for t in sorted_idx]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(N_HOURS), sorted_prices, color="gray", linewidth=1.2, label="Preis (sortiert)")
on_x = [i for i, flag in enumerate(sorted_running) if flag == 1]
on_y = [sorted_prices[i] for i in on_x]
ax.scatter(on_x, on_y, color="seagreen", s=10, zorder=3,
           label=f"Kaskade laeuft ({best_label})")
ax.set_xlabel("Stunden, nach Preis sortiert")
ax.set_ylabel("EUR/MWh")
ax.set_title("Preis-Dauerlinie: Wann laeuft die Kaskade?", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cascade_price_duration.png", dpi=150)
plt.show()
print("Saved: cascade_price_duration.png")

# ================================================================
# PLOT 6: Stromverbrauchsanteil je Modul
# ================================================================
elec_per_module = {
    m: sum(best_u[m][t] * CAPACITIES[m] / cops[t] for t in range(N_HOURS))
    for m in MODULE_NAMES
}
fig, ax = plt.subplots(figsize=(6, 6))
vals = list(elec_per_module.values())
labs = [f"{m}\n({v:.0f} kWh)" for m, v in elec_per_module.items()]
ax.pie(vals, labels=labs, colors=[colors[m] for m in MODULE_NAMES],
       autopct="%1.0f%%", startangle=90)
ax.set_title(f"Stromverbrauchsanteil je Modul - {best_label}",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("cascade_module_share.png", dpi=150)
plt.show()
print("Saved: cascade_module_share.png")

# ================================================================
# PLOT 7 & 8: Beispieltage - kaeltester und mildester Tag (Zoom-In)
# ================================================================
n_days = N_HOURS // 24
day_avg_temps = [sum(temps[d*24:(d+1)*24]) / 24 for d in range(n_days)]
cold_day_idx = day_avg_temps.index(min(day_avg_temps))
mild_day_idx = day_avg_temps.index(max(day_avg_temps))

def plot_example_day(day_idx, title_suffix, filename):
    start, end = day_idx * 24, day_idx * 24 + 24
    day_dts = dts[start:end]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                              gridspec_kw={"height_ratios": [0.8, 0.8, 1.3, 1]})
    fig.suptitle(f"Beispieltag: {title_suffix} ({day_dts[0].strftime('%d.%m.%Y')}, "
                 f"Ø {day_avg_temps[day_idx]:.1f}°C) - {best_label}", fontsize=12, fontweight="bold")

    axes[0].plot(day_dts, temps[start:end], color="steelblue", marker="o", markersize=3)
    axes[0].set_ylabel("Außentemp.\n[°C]", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(day_dts, prices[start:end], color="gray", marker="o", markersize=3)
    axes[1].set_ylabel("Preis\n[EUR/MWh]", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    bottom = [0.0] * 24
    for m in MODULE_NAMES:
        vals = [CAPACITIES[m] * r["u_vals"][m][t] for t in range(start, end)]
        axes[2].bar(day_dts, vals, bottom=bottom, width=0.9 / 24,
                    color=colors[m], label=m, align="edge")
        bottom = [b + v for b, v in zip(bottom, vals)]
    axes[2].step(day_dts, demands[start:end], color="black", linewidth=1.5,
                 linestyle="--", where="post", label="Bedarf")
    axes[2].set_ylabel("Leistung\n[kW]", fontsize=8)
    axes[2].legend(fontsize=7, ncol=4, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    tank_pct = [v / tank_max * 100 for v in r["s_vals"][start:end]]
    axes[3].plot(day_dts, tank_pct, color="seagreen", linewidth=1.5)
    axes[3].fill_between(day_dts, tank_pct, alpha=0.15, color="seagreen")
    axes[3].set_ylabel("Tank\n[% voll]", fontsize=8)
    axes[3].set_ylim(-5, 115)
    axes[3].set_xlabel("Uhrzeit")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")

plot_example_day(cold_day_idx, "Extremer Winter-Tag (kältester Tag im Zeitraum)", "cascade_example_cold_day.png")
plot_example_day(mild_day_idx, "mildester Tag im Betrachtungszeitraum", "cascade_example_mild_day.png")

# ================================================================
# Repraesentative Tage je Jahreszeit (kein Extremwert): Tag, dessen
# Tagesmitteltemperatur am naechsten an der Saison-Durchschnittstemperatur
# liegt. Ergaenzt den bereits vorhandenen Extremwinter-/Mildtag-Vergleich.
# ================================================================
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

for season, idx in representative_days.items():
    plot_example_day(idx, f"Repräsentativer {season}-Tag", f"cascade_example_{season.lower()}.png")

# ================================================================
# PLOT 9: Kalender-Heatmap - Gesamtleistung je Stunde ueber den Zeitraum
# ================================================================
heat_data = np.zeros((24, n_days))
for d in range(n_days):
    for h in range(24):
        t = d * 24 + h
        heat_data[h, d] = sum(CAPACITIES[m] * best_u[m][t] for m in MODULE_NAMES)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heat_data, aspect="auto", cmap="YlOrRd", origin="lower",
               extent=[0, n_days, 0, 24])
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Kaskaden-Gesamtleistung [kW]")
ax.set_xlabel("Tag")
ax.set_ylabel("Stunde des Tages")
ax.set_title(f"Kaskaden-Fahrweise über {n_days} Tage - {best_label}", fontsize=12, fontweight="bold")
if n_days > 40:
    # Bei einem Volljahr waeren taegliche Beschriftungen unlesbar -> nur der
    # 1. jedes Monats wird beschriftet.
    tick_days = [d for d in range(n_days) if dts[d * 24].day == 1]
    ax.set_xticks([d + 0.5 for d in tick_days])
    ax.set_xticklabels([dts[d * 24].strftime("%b") for d in tick_days], fontsize=8)
else:
    ax.set_xticks([d + 0.5 for d in range(n_days)])
    ax.set_xticklabels([dts[d * 24].strftime("%d.%m") for d in range(n_days)],
                        rotation=45, fontsize=7)
ax.set_yticks(range(0, 24, 3))
plt.tight_layout()
plt.savefig("cascade_heatmap.png", dpi=150)
plt.show()
print("Saved: cascade_heatmap.png")

# ================================================================
# PLOT 10: Gestapelte Leistungsanteile je Modul ueber den Betrachtungszeitraum
# ================================================================
fig, ax = plt.subplots(figsize=(14, 5))
stack_vals = [[CAPACITIES[m] * best_u[m][t] for t in range(N_HOURS)] for m in MODULE_NAMES]
ax.stackplot(dts, stack_vals, labels=MODULE_NAMES,
             colors=[colors[m] for m in MODULE_NAMES], alpha=0.85)
ax.plot(dts, demands, color="black", linewidth=0.8, linestyle="--", label="Bedarf")
ax.set_ylabel("Leistung [kW]")
ax.set_title(f"Kaskaden-Leistungsanteile je Modul - {best_label}", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=4, loc="upper right")
format_date_axis(ax)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("cascade_stacked_area.png", dpi=150)
plt.show()
print("Saved: cascade_stacked_area.png")

# ================================================================
# PLOT 11: Jahresdauerlinie des Bedarfs vs. Kaskaden-Leistungsstufen
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5))
sorted_demand = sorted(demands, reverse=True)
ax.plot(range(N_HOURS), sorted_demand, color="steelblue", linewidth=1.5, label="Bedarf (sortiert)")
# Volllast-Referenzstufen (alle Module bei 100%) - mit Modulation sind
# zwischen 0 und diesen Stufen jetzt auch Zwischenwerte erreichbar.
for level in [6, 12, 18, 24, 30]:
    ax.axhline(level, color="gray", linestyle=":", linewidth=0.8)
    ax.text(N_HOURS * 0.99, level + 0.3, f"{level} kW", fontsize=7, ha="right", color="gray")
# NEU (V9): kleinste erreichbare Nicht-Null-Leistung (kleinstes Modul bei
# MOD_MIN) - unterhalb dieser Linie (und oberhalb 0) klafft weiterhin eine
# kleine Luecke, in der nur deficit/spill oder Tankpufferung helfen.
mod_min_kw = MOD_MIN * min(CAPACITIES.values())
ax.axhline(mod_min_kw, color="tomato", linestyle="--", linewidth=1.0)
ax.text(N_HOURS * 0.99, mod_min_kw + 0.3, f"MOD_MIN: {mod_min_kw:.1f} kW",
        fontsize=7, ha="right", color="tomato")
ax.set_xlabel("Stunden (nach Bedarf absteigend sortiert)")
ax.set_ylabel("Wärmebedarf [kW]")
ax.set_title("Dauerlinie des Bedarfs vs. verfügbare Kaskaden-Leistungsstufen (mit Modulation)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cascade_load_duration.png", dpi=150)
plt.show()
print("Saved: cascade_load_duration.png")

print("\nDone!")