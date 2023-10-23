# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:58:33 2023

@author: agrot

Version 2.1 of the project thesis optimization problem code
- implementing TES (big M)
"""

# %% ----- Interchangeable data - User Interface ----- #

# Define the set of hours in the year (step1 = 1, tf = 8760)
# or define time interval to be examined
step1 = 0
tf = 2030 #rt = 2m40s with 745, =5m20s with 1417 (jan & feb)

# Defining power production limits 
gen_maxcap = 300
gen_lowcap = 0

# Production price (cost) in NOK/MWh (marginal cost)
production_price = 100.0  # Example, will be lower
production_price_tes = production_price  # Example

# Defining ramping limits 
lower_ramp_lim = -5 #[%/min]
upper_ramp_lim = 5


# --- TES --- #

# Defining TES inflow limits 
inflow_maxcap = 300 #[MW]
inflow_lowcap = 0

# Defining TES outflow limits 
outflow_maxcap = 300 #[MW]
outflow_lowcap = 0

# Defining TES capacity
tes_maxcap = 1500 #[MWh]
tes_lowcap = 0


# %% ----- Dependencies ----- #

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import pandas as pd


# %% ----- Model setup ----- #

# Create a concrete Pyomo model
model = pyo.ConcreteModel()

# List for hour range to be examined
hours = range(step1, tf + 1)  # Assuming there are 8760 hours in a year (8761, one month: 745)


# %% ----- Reading in data ----- #

def InputData(data_file): #reading the excel file with the values of generation and cost
    inputdata = pd.read_excel(data_file, usecols='B')
    #inputdata = inputdata.set_index('Parameter', drop=True)
    #power_prices = inputdata['Power price']
    power_prices = inputdata['price'].tolist()
    return power_prices
power_prices = InputData('Power prices 8760 1.0.xlsx')


# %% ----- Variables ----- # 

model.power_generation       = pyo.Var(hours, within=pyo.NonNegativeReals) # power generation in each hour
model.power_generation_state = pyo.Var(hours, within=pyo.Binary)           # power generation state in each hour
model.power_generation_tes   = pyo.Var(hours, within=pyo.NonNegativeReals) # power generation for TES in each hour
model.inflow_tes             = pyo.Var(hours, within=pyo.NonNegativeReals) # inflow to TES in each hour
model.inflow_tes_state       = pyo.Var(hours, within=pyo.Binary)           # inflow to TES state in each hour
model.fuel_tes               = pyo.Var(hours, within=pyo.NonNegativeReals) # fuel level in TES in each hour


# %% ----- Objective Function: Maximize the surplus (profit) ----- #

def objective_rule(model):
    return sum((power_prices[hour]) * (model.power_generation[hour] + model.power_generation_tes[hour]) - (production_price) * (model.power_generation[hour] + model.inflow_tes[hour]) for hour in hours)

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


# %% ----- Generation Constraints ----- #

def production_limit_upper(model, hour):
    return model.power_generation[hour] <= gen_maxcap
model.prod_limup = pyo.Constraint(hours, rule=production_limit_upper)

def production_limit_lower(model, hour):
    return gen_lowcap <= model.power_generation[hour]
model.prod_limlow = pyo.Constraint(hours, rule=production_limit_lower)


# %% ----- Ramping Constraints ----- #


# Converting ramping limits 
gen_lowramp = lower_ramp_lim * 0.6 * gen_maxcap #[MW/h]
gen_maxramp = upper_ramp_lim * 0.6 * gen_maxcap

def production_ramping_up(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return (model.power_generation[hour]) - (model.power_generation[hour - 1]) <= gen_maxramp #(model.power_generation[hour] + model.power_generation_tes[hour]) - (model.power_generation[hour - 1] + model.power_generation_tes[hour - 1]) <= gen_maxramp 
model.prod_rampup = pyo.Constraint(hours, rule=production_ramping_up)

def production_ramping_down(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return gen_lowramp <= (model.power_generation[hour]) - (model.power_generation[hour - 1]) # gen_lowramp <= (model.power_generation[hour] + model.power_generation_tes[hour]) - (model.power_generation[hour - 1] + model.power_generation_tes[hour - 1])
model.prod_rampdown = pyo.Constraint(hours, rule=production_ramping_down)


# ----- TES Ramping constraints ----- #

# Defining ramping limits 
tes_lowramp = gen_lowramp #[MW/h]
tes_maxramp = gen_maxramp

def tes_ramping_up(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return (model.power_generation_tes[hour]) - (model.power_generation_tes[hour - 1]) <= tes_maxramp 
model.tes_rampup = pyo.Constraint(hours, rule=tes_ramping_up)

def tes_ramping_down(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return tes_lowramp <= (model.power_generation_tes[hour]) - (model.power_generation_tes[hour - 1]) 
model.tes_rampdown = pyo.Constraint(hours, rule=tes_ramping_down)


# %% ----- TES Energy Balance Constraint ----- #

def tes_balance(model, hour): 
    if hour == step1:
        return model.fuel_tes[hour] == 0
    else:
        return model.fuel_tes[hour] == model.fuel_tes[hour - 1] + model.inflow_tes[hour - 1] - model.power_generation_tes[hour - 1]
model.tes_energy_bal = pyo.Constraint(hours, rule=tes_balance)


# %% ----- TES Flow Constraints using big M ----- #

M = 10**4

# ----- Binary constraints ----- #

# Constraint to link binary variable with power generation
def binary_powgen(model, hour):
    return model.power_generation[hour] <= model.power_generation_state[hour] * M

model.binary_constraint1 = pyo.Constraint(hours, rule=binary_powgen)

# Constraint to link binary variable with TES inflow
def binary_inflow(model, hour):
    return model.inflow_tes[hour] <= model.inflow_tes_state[hour] * M

model.binary_constraint2 = pyo.Constraint(hours, rule=binary_inflow)


# ----- Multiflow constraints ----- #

def flow_const1(model, hour):
    return model.inflow_tes[hour] <= 0 + (1 - model.power_generation_state[hour]) * M

model.flow_constraint = pyo.Constraint(hours, rule=flow_const1)

def flow_const2(model, hour):
    return model.power_generation_tes[hour] <= 0 + (1 - model.inflow_tes_state[hour]) * M

model.flow_constraint2 = pyo.Constraint(hours, rule=flow_const2)



# %% ----- TES Inflow Limit Constraints ----- #

def tes_inflow_limit_upper(model, hour):
    return model.inflow_tes[hour] <= inflow_maxcap
model.tesin_limup = pyo.Constraint(hours, rule=tes_inflow_limit_upper)

def tes_inflow_limit_lower(model, hour):
    return inflow_lowcap <= model.inflow_tes[hour]
model.tesin_limlow = pyo.Constraint(hours, rule=tes_inflow_limit_lower)


# ----- TES production limit constraints ----- #

def tes_outflow_limit_upper(model, hour):
    return model.power_generation_tes[hour] <= outflow_maxcap
model.tesout_limup = pyo.Constraint(hours, rule=tes_outflow_limit_upper)

def tes_outflow_limit_lower(model, hour):
    return outflow_lowcap <= model.power_generation_tes[hour]
model.tesout_limlow = pyo.Constraint(hours, rule=tes_outflow_limit_lower)


# %% ----- TES Capacity Constraint ----- #

def tes_cap_upper(model, hour):
    return model.fuel_tes[hour] <= tes_maxcap
model.tes_up_cap = pyo.Constraint(hours, rule=tes_cap_upper)

def tes_cap_lower(model, hour):
    return tes_lowcap <= model.fuel_tes[hour]
model.tes_low_cap = pyo.Constraint(hours, rule=tes_cap_lower)


# %% ----- Solving the optimization problem ----- #

PATH_ipopt = ('C:/Users/agrot/anaconda3/Library/bin/ipopt')
#opt = SolverFactory('ipopt', executable=PATH_ipopt)
opt = SolverFactory('gurobi')

#solver = pyo.SolverFactory('ipopt', executable='C:/Users/agrot/anaconda3/Library/bin')

opt.solve(model)

# %% ----- Printing and plotting results ----- #

print("Optimal Surplus: ", pyo.value(model.objective), "NOK")
#print("Optimal Power Generation:")
#for hour in hours:
#    print(f"Hour {hour}: {pyo.value(model.power_generation[hour])} MWh")
   
# ---- Plotting distribution ---- #

hourslist = list(hours)
val_gen = list(pyo.value(model.power_generation[hour]) for hour in hours)
val_tes = list(pyo.value(model.power_generation_tes[hour]) for hour in hours)
val_inflow = list(pyo.value(model.inflow_tes[hour]) for hour in hours)
val_capacity = list(pyo.value(model.fuel_tes[hour]) for hour in hours)
#val_state = list(pyo.value(model.power_generation_state[hour]) for hour in hours)

jan= 744 #, mar, may, jul, aug, okt, dec
apr= 720 #, jun, sep, nov
feb= 672

#plt.figure().set_figwidth(15)
#plt.figure(figsize=(20,6))

# Creating plot for nom. generation
fig, ax1 = plt.subplots()

ax1.set_xlabel("Hours")
ax1.set_ylabel("Generation output [MW]")
ax1.bar(hourslist[:tf - step1], val_gen[:tf - step1], color = 'r')
#ax1.bar(hourslist[:744], val_tes[:744], color = 'g')
#ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
ax2 = ax1.twinx()
 
ax2.set_ylabel('Power prices [NOK/MWh]')
ax2.plot(hourslist[:tf - step1], power_prices[step1:tf], color = 'b') #[:timeframe - 1]
#ax2.tick_params(axis ='y', labelcolor = color)

plt.title("Generation profile of NPP")
plt.show()



# Creating plot for tes generation
fig, ax3 = plt.subplots()

ax3.set_xlabel("Hours")
ax3.set_ylabel("TES Generation output [MW]")
ax3.bar(hourslist[:tf - step1], val_tes[:tf - step1], color = 'g')
#ax1.bar(hourslist[:744], val_tes[:744], color = 'g')
#ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
ax4 = ax3.twinx()
 
ax4.set_ylabel('Power prices [NOK/MWh]')
ax4.plot(hourslist[:tf - step1], power_prices[step1:tf], color = 'b') #[:timeframe - 1]
#ax2.tick_params(axis ='y', labelcolor = color)

plt.title("Generation profile of NPP TES")
plt.show()


# Creating plot for tes inflow
fig, inf = plt.subplots()

inf.set_xlabel("Hours")
inf.set_ylabel("TES Inflow [MWh]")
inf.bar(hourslist[:tf - step1], val_inflow[:tf - step1], color = 'y')
#ax1.bar(hourslist[:744], val_tes[:744], color = 'g')
#ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
inf2 = inf.twinx()
 
inf2.set_ylabel('Power prices [NOK/MWh]')
inf2.plot(hourslist[:tf - step1], power_prices[step1:tf], color = 'b') #[:timeframe - 1]
#ax2.tick_params(axis ='y', labelcolor = color)

plt.title("Inflow profile to TES")
plt.show()


# Creating plot for tes capacity
fig, cap = plt.subplots()

cap.set_xlabel("Hours")
cap.set_ylabel("TES Capacity [MWh]")
cap.bar(hourslist[:tf - step1], val_capacity[:tf - step1], color = 'k')
#ax1.bar(hourslist[:744], val_tes[:744], color = 'g')
#ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
cap2 = cap.twinx()
 
cap2.set_ylabel('Power prices [NOK/MWh]')
cap2.plot(hourslist[:tf - step1], power_prices[step1:tf], color = 'b') #[:timeframe - 1]
#ax2.tick_params(axis ='y', labelcolor = color)

plt.title("Capacity profile of TES")
plt.show()