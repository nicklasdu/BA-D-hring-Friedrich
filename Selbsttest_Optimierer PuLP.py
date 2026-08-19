from pulp import LpProblem, LpStatus, lpSum, LpVariable, LpMinimize 
import pandas as pd 


# Create the model 
# Erster Schritt LpProblem erstellen und definieren 

# Zum nächsten mal per Hand Gleichungen aufschreiben
# sinnvolles und einfaches eigenes Profil für Preise und heat_demand 



# Daten zum Strompreis ct./kwh 
# prices = [10, 10, 10, 10, 10, 10, 40, 40, 40, 40, 40, 40, 10, 40, 10, 40, 40, 40, 40, 10, 10, 10, 40, 40] # 24 Werte für 24 Stunden
df_prices = pd.read_csv(r"C:\Users\janfriedrich\Desktop\JanF\Studium\BACHI4\Codes\hamburg_heating_electricity_2025.csv", sep=',', encoding='utf-8-sig')
prices = pd.to_numeric(df_prices['electricity_price_eur_per_mwh'], errors='coerce').tolist() 
if any(pd.isna(p) for p in prices):
    print("Warnung: Es gibt nicht-numerische Werte in der Preis-Spalte!")
prices = [p * 0.1 for p in prices]

# Heizlast in kW
# heat_demand = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8] # 24 Werte für 24 Stunden
df_heat_demand = pd.read_csv(r"C:\Users\janfriedrich\Desktop\JanF\Studium\BACHI4\Codes\hamburg_heating_electricity_2025.csv", sep=',', encoding='utf-8-sig')
heat_demand = pd.to_numeric(df_heat_demand['heating_output_w'], errors='coerce').tolist()
heat_demand = [wert / 1000 for wert in heat_demand]

# heat_demand = []
#for wert in t heat_stündlich
#    heat_demand.extend([wert] * 4)
# assert len(heat_demand) == 96 # Jahresanzahl an Stunden * 4 
# assert len(prices) == 96 # Jahresanzahl an Stunden * 4 

# ----------
# Modell A mit Speicher 
# -----------
model_A = LpProblem(name='A_total_cost', sense=LpMinimize)


# Wärmespeicher 
# storage_max_output = 9 # kWp kann weg, weil eigentlich kein max output im Haus 
storage_capacity = 40 # kWh  (Ergibt die Größe Sinn???) wie lange soll Speicher Wärmebedarf decken
storage_charge = [LpVariable(f'charge_{t}', lowBound=0, upBound=12, cat='continuous') for t in range(8760)] # Variable für die Ladung des Speichers (kW)
storage_discharge = [LpVariable(f'discharge_{t}', lowBound=0, cat='continuous') for t in range(8760)] # Entladung des Speichers 
storage_soc = [LpVariable(f'soc_{t}', lowBound=0, upBound=storage_capacity, cat='continuous') for t in range(8760)] # State of Charge (kWh)


# länge der Daten definieren 
assert len(prices) == 8760 and len(heat_demand) == 8760, "Daten müssen 24 Werte enthalten"


A_c = 4 # COP der Wärmepumpe
A_P_el = [LpVariable(f'A_P_el{t}', lowBound=0, upBound=3, cat='continuous') for t in range(8760)] # elektrische Leistung der Wärmepumpe
# A_P_th = [LpVariable(f'A_P_th{t}',lowBound=0, upBound=A_P_el * A_c, cat='continuous') for t in range (24)] # thermische Leistung der Wärmepumpe  

# P_th = [LpVariable(f'P_th{t}', lowBound=0, upBound= 3*c, cat='continuous') for t in range(24)]



# Modell kreiren, um die Kosten zu minimieren 
# Kostenmodel: Ziel = Kosten minimeren (Preis/h * Leistung/h) über 24 Stunden aufsummieren

model_A += lpSum(prices[t] * A_P_el[t] for t in range(8760)), 'A_total_cost' 

# model_A += lpSum(A_P_el[0] * A_c), 'A_WP_th'

# Füllstand des Speicher: State of Charger + Charge - discharge) 
model_A += (storage_soc[0] == storage_charge[0] - storage_discharge[0], 'soc_initial') # Anfangsfüllstand des Speichers (kWh)
for t in range(1, 8760):
    model_A += (storage_soc[t] == storage_soc[t-1] + storage_charge[t] - storage_discharge[t], f'soc_balance_{t}') # Füllstand des Speichers stündlich


# Heizlast muss gedeckt werden (Leistung * COP >= Heizlast)
for t in range(8760):
    model_A += (A_P_el[t]*A_c + storage_discharge[t] >= heat_demand[t], f'Heat_demand_constraint_{t}')

# Speicherbegrenzung: Leistung der Wärmepumpe darf die maximale Leistung des Speichers nicht überschreiten - Warum nicht? 
# for t in range(24):
   # model += ((P[t]*c) <= storage_max_output, f'Storage_output_constraint_{t}') 

for t in range(8760):
    model_A += (A_P_el[t]*A_c == heat_demand[t] - storage_discharge[t] + storage_charge[t], f'A_WP_output_{t}') # Der Output der Wärme muss immer auch "ein Ziel" haben 
   #  model += (storage_discharge[t] f'storage_discharge_constaints_{t}') # maximale Entladung des Speichers pro Stunde 
   # model += (storage_charge[t] <= (P_th - heat_demand[t]), f'storage_charge_constraints_{t}') # maximale Ladung des Speichers durch WP pro Stunde 
    model_A += (storage_capacity >= storage_soc[t], f'storage_capacity_constraint_{t}') # maximale Kapazität des Speichers darf nicht überschritten werden
    model_A += (A_P_el[t]*A_c + storage_discharge[t] >= heat_demand[t], f'heat_demand_{t}')  
  #   model_A += (A_P_el[t]*A_c - heat_demand[t] + storage_discharge[t]), f'storage_charge' # Ladung des Speichers 

# ----------- 
# Modell B ohne Speicher 
# -----------

model_B = LpProblem(name='B_total_cost', sense=LpMinimize)

assert len(prices) == 8760 and len(heat_demand) == 8760, "Daten müssen 24 Werte enthalten"


B_c = 4 # COP der Wärmepumpe
B_P_el = [LpVariable(f'B_P_el{t}', lowBound=0, upBound=3, cat='continuous') for t in range(8760)] # elektrische Leistung der Wärmepumpe 


# Modell kreiren, um die Kosten zu minimieren 
# Kostenmodel: Ziel = Kosten minimeren (Preis/h * Leistung/h) über 24 Stunden aufsummieren

model_B += lpSum(prices[t] * B_P_el[t] for t in range(8760)), 'B_total_cost' 




# Heizlast muss gedeckt werden (Leistung * COP >= Heizlast)
for t in range(8760):
    model_B += (B_P_el[t]*B_c == heat_demand[t], f'Heat_demand_constraint_{t}')


for t in range(8760):
    model_B += (B_P_el[t]*B_c == heat_demand[t], f'B_WP_output_{t}') # Der Output der Wärme muss immer auch "ein Ziel" haben 




# Modell lösen 

model_A.solve()

# Ergebnisse anzeigen und ausgeben lassen 
A_total_cost = sum(prices[t] * A_P_el[t].value() for t in range(8760)) 
print('Status Modell A:', LpStatus[model_A.status], '| Gesamtkosten:', round(A_total_cost, 2))




model_B.solve()


B_total_cost = sum(prices[t] * B_P_el[t].value() for t in range(8760)) 
print('Status Modell B:', LpStatus[model_B.status], '| Gesamtkosten:', round(B_total_cost, 2))


import openpyxl 
from openpyxl.styles import Font, PatternFill, Alignment

wb = openpyxl.Workbook()

# ═════════════════════════════════════════
# SHEET 1: Modell A (mit Speicher)
# ═════════════════════════════════════════
ws_A = wb.active
ws_A.title = "Modell A - Mit Speicher"

# Header
header = ["Stunde", "Preis [ct]", "Heizbedarf [kW]", "P_el [kW]",
          "P_th [kW]", "Speicher laden [kW]", "Speicher entladen [kW]",
          "SOC [kWh]", "Kosten [ct]"]

for col, h in enumerate(header, 1):
    cell = ws_A.cell(row=1, column=col, value=h)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", start_color="2E75B6")
    cell.alignment = Alignment(horizontal="center")
    ws_A.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 20

# Daten Modell A
for t in range(8760):
    ws_A.cell(row=t+2, column=1, value=t)
    ws_A.cell(row=t+2, column=2, value=prices[t])
    ws_A.cell(row=t+2, column=3, value=heat_demand[t])
    ws_A.cell(row=t+2, column=4, value=round(A_P_el[t].value(), 3))
    ws_A.cell(row=t+2, column=5, value=round(A_P_el[t].value() * A_c, 3))
    ws_A.cell(row=t+2, column=6, value=round(storage_charge[t].value(), 3))
    ws_A.cell(row=t+2, column=7, value=round(storage_discharge[t].value(), 3))
    ws_A.cell(row=t+2, column=8, value=round(storage_soc[t].value(), 3))
    ws_A.cell(row=t+2, column=9, value=round(prices[t] * A_P_el[t].value(), 3))

# Summenzeile
ws_A.cell(row=8762, column=1, value="GESAMT").font = Font(bold=True)
ws_A.cell(row=8762, column=9, value=round(A_total_cost, 2)).font = Font(bold=True)

# ═════════════════════════════════════════
# SHEET 2: Modell B (ohne Speicher)
# ═════════════════════════════════════════
ws_B = wb.create_sheet("Modell B - Ohne Speicher")

header_B = ["Stunde", "Preis [ct]", "Heizbedarf [kW]", "P_el [kW]",
            "P_th [kW]", "Kosten [ct]"]

for col, h in enumerate(header_B, 1):
    cell = ws_B.cell(row=1, column=col, value=h)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", start_color="16A34A")
    cell.alignment = Alignment(horizontal="center")
    ws_B.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 20

for t in range(8760):
    ws_B.cell(row=t+2, column=1, value=t)
    ws_B.cell(row=t+2, column=2, value=prices[t])
    ws_B.cell(row=t+2, column=3, value=heat_demand[t])
    ws_B.cell(row=t+2, column=4, value=round(B_P_el[t].value(), 3))
    ws_B.cell(row=t+2, column=5, value=round(B_P_el[t].value() * B_c, 3))
    ws_B.cell(row=t+2, column=6, value=round(prices[t] * B_P_el[t].value(), 3))

ws_B.cell(row=8762, column=1, value="GESAMT").font = Font(bold=True)
ws_B.cell(row=8762, column=6, value=round(B_total_cost, 2)).font = Font(bold=True)

# ═════════════════════════════════════════
# SHEET 3: Vergleich
# ═════════════════════════════════════════
ws_V = wb.create_sheet("Vergleich")

vergleich = [
    ["", "Modell A (mit Speicher)", "Modell B (ohne Speicher)"],
    ["Gesamtkosten [ct]",  round(A_total_cost, 2), round(B_total_cost, 2)],
    ["Gesamtkosten [EUR]", round(A_total_cost/100, 4), round(B_total_cost/100, 4)],
    ["Ersparnis [ct]",     round(B_total_cost - A_total_cost, 2), "—"],
    ["Ersparnis [%]",      round((B_total_cost - A_total_cost) / B_total_cost * 100, 1), "—"],
]

for row, data in enumerate(vergleich, 1):
    for col, val in enumerate(data, 1):
        cell = ws_V.cell(row=row, column=col, value=val)
        if row == 1 or col == 1:
            cell.font = Font(bold=True)
        ws_V.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 25

# Speichern
pfad = r"C:\Users\janfriedrich\Desktop\JanF\Studium\BACHI4\Codes\waermepumpe_ergebnisse.xlsx"
wb.save(pfad)
print(f"Excel gespeichert: {pfad}")
