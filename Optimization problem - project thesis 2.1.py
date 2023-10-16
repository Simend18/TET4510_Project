# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:58:33 2023

@author: agrot

Version 2.0 of the project thesis optimization problem code
- implementing TES (big M)
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import pandas as pd

# Create a concrete Pyomo model
model = pyo.ConcreteModel()

# Define the set of hours in the year
timeframe = 745
hours = range(1, timeframe + 1)  # Assuming there are 8760 hours in a year (8761, one month: 745)

# Power prices for each hour in dollars per MWh
#power_prices = [10.0, 12.0, 15.0, 13.0, 11.0] #Example


# %% ----- Reading in data ----- #

def InputData(data_file): #reading the excel file with the values of generation and cost
    inputdata = pd.read_excel(data_file, usecols='B')
    #inputdata = inputdata.set_index('Parameter', drop=True)
    #power_prices = inputdata['Power price']
    power_prices = inputdata['price'].tolist()
    return power_prices
power_prices = InputData('Power prices 8760 1.0.xlsx')


# Production price (cost) in NOK/MWh
production_price = 150.0  # Example, will be lower
production_price_tes = production_price  # Example


# %% ----- Variables ----- # 

model.power_generation       = pyo.Var(hours, within=pyo.NonNegativeReals) # power generation in each hour
model.power_generation_state = pyo.Var(hours, within=pyo.Binary)           # power generation state in each hour
model.power_generation_tes   = pyo.Var(hours, within=pyo.NonNegativeReals) # power generation for TES in each hour
model.inflow_tes             = pyo.Var(hours, within=pyo.NonNegativeReals) # inflow to TES in each hour
model.inflow_tes_state       = pyo.Var(hours, within=pyo.Binary)           # inflow to TES state in each hour
model.fuel_tes               = pyo.Var(hours, within=pyo.NonNegativeReals) # fuel level in TES in each hour


# %% ----- Objective function: Maximize the surplus (profit) ----- #

def objective_rule(model):
    return sum((power_prices[hour]) * (model.power_generation[hour] + model.power_generation_tes[hour]) - (production_price) * (model.power_generation[hour] + model.inflow_tes[hour]) for hour in hours)

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


# %% ----- Generation constraints ----- #

# Defining power production limits 
gen_maxcap = 300
gen_lowcap = 0

def production_limit_upper(model, hour):
    return model.power_generation[hour] <= gen_maxcap
model.prod_limup = pyo.Constraint(hours, rule=production_limit_upper)

def production_limit_lower(model, hour):
    return gen_lowcap <= model.power_generation[hour]
model.prod_limlow = pyo.Constraint(hours, rule=production_limit_lower)


# %% ----- Ramping constraints ----- #

# Defining ramping limits 
gen_lowramp = -20 #[MW/h]
gen_maxramp = 20

def production_ramping_up(model, hour):
    if hour == 1:
        return pyo.Constraint.Skip
    else:
        return (model.power_generation[hour]) - (model.power_generation[hour - 1]) <= gen_maxramp #(model.power_generation[hour] + model.power_generation_tes[hour]) - (model.power_generation[hour - 1] + model.power_generation_tes[hour - 1]) <= gen_maxramp #(model.power_generation[hour]) - (model.power_generation[hour - 1]) <= gen_maxramp
model.prod_rampup = pyo.Constraint(hours, rule=production_ramping_up)

def production_ramping_down(model, hour):
    if hour == 1:
        return pyo.Constraint.Skip
    else:
        return gen_lowramp <= (model.power_generation[hour]) - (model.power_generation[hour - 1]) #gen_lowramp <= (model.power_generation[hour] + model.power_generation_tes[hour]) - (model.power_generation[hour - 1] + model.power_generation_tes[hour - 1])  
model.prod_rampdown = pyo.Constraint(hours, rule=production_ramping_down)


# ----- TES Ramping constraints ----- #

# Defining ramping limits 
tes_lowramp = gen_lowramp #[MW/h]
tes_maxramp = gen_maxramp

def tes_ramping_up(model, hour):
    if hour == 1:
        return pyo.Constraint.Skip
    else:
        return (model.power_generation_tes[hour]) - (model.power_generation_tes[hour - 1]) <= tes_maxramp 
model.tes_rampup = pyo.Constraint(hours, rule=tes_ramping_up)

def tes_ramping_down(model, hour):
    if hour == 1:
        return pyo.Constraint.Skip
    else:
        return tes_lowramp <= (model.power_generation_tes[hour]) - (model.power_generation_tes[hour - 1]) 
model.tes_rampdown = pyo.Constraint(hours, rule=tes_ramping_down)


# %% ----- TES energy balance constraint ----- #

def tes_balance(model, hour): 
    if hour ==1:
        return model.fuel_tes[hour] == 0
    else:
        return model.fuel_tes[hour] == model.fuel_tes[hour - 1] + model.inflow_tes[hour - 1] - model.power_generation_tes[hour - 1]
model.tes_energy_bal = pyo.Constraint(hours, rule=tes_balance)


# %% ----- TES Flow constraints using big M ----- #

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



# %% ----- TES inflow limit constraints ----- #

# Defining TES inflow limits 
inflow_maxcap = 300
inflow_lowcap = 0

def tes_inflow_limit_upper(model, hour):
    return model.inflow_tes[hour] <= inflow_maxcap
model.tesin_limup = pyo.Constraint(hours, rule=tes_inflow_limit_upper)

def tes_inflow_limit_lower(model, hour):
    return inflow_lowcap <= model.inflow_tes[hour]
model.tesin_limlow = pyo.Constraint(hours, rule=tes_inflow_limit_lower)


# ----- TES production limit constraints ----- #

# Defining TES inflow limits 
outflow_maxcap = 300
outflow_lowcap = 0

def tes_outflow_limit_upper(model, hour):
    return model.power_generation_tes[hour] <= outflow_maxcap
model.tesout_limup = pyo.Constraint(hours, rule=tes_outflow_limit_upper)

def tes_outflow_limit_lower(model, hour):
    return outflow_lowcap <= model.power_generation_tes[hour]
model.tesout_limlow = pyo.Constraint(hours, rule=tes_outflow_limit_lower)


# %% ----- TES capacity constraint ----- #

tes_maxcap = 1500 # [MWh]
tes_lowcap = 0

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

# Print the results

#print("Optimal Surplus: ", pyo.value(model.objective))
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

yo = timeframe

#print("boom")

#plt.figure().set_figwidth(15)
#plt.figure(figsize=(20,6))

# Creating plot for nom. generation
fig, ax1 = plt.subplots()

ax1.set_xlabel("Hours")
ax1.set_ylabel("Generation output [MW]")
ax1.bar(hourslist[:yo], val_gen[:yo], color = 'r')
#ax1.bar(hourslist[:744], val_tes[:744], color = 'g')
#ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
ax2 = ax1.twinx()
 
ax2.set_ylabel('Power prices [NOK/MWh]')
ax2.plot(hourslist[:yo], power_prices[:yo], color = 'b') #[:timeframe - 1]
#ax2.tick_params(axis ='y', labelcolor = color)

plt.title("Generation profile of NPP")
plt.show()



# Creating plot for tes generation
fig, ax3 = plt.subplots()

ax3.set_xlabel("Hours")
ax3.set_ylabel("TES Generation output [MW]")
ax3.bar(hourslist[:yo], val_tes[:yo], color = 'g')
#ax1.bar(hourslist[:744], val_tes[:744], color = 'g')
#ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
ax4 = ax3.twinx()
 
ax4.set_ylabel('Power prices [NOK/MWh]')
ax4.plot(hourslist[:yo], power_prices[:yo], color = 'b') #[:timeframe - 1]
#ax2.tick_params(axis ='y', labelcolor = color)

plt.title("Generation profile of NPP TES")
plt.show()


# Creating plot for tes inflow
fig, inf = plt.subplots()

inf.set_xlabel("Hours")
inf.set_ylabel("TES Inflow [MWh]")
inf.bar(hourslist[:yo], val_inflow[:yo], color = 'y')
#ax1.bar(hourslist[:744], val_tes[:744], color = 'g')
#ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
inf2 = inf.twinx()
 
inf2.set_ylabel('Power prices [NOK/MWh]')
inf2.plot(hourslist[:yo], power_prices[:yo], color = 'b') #[:timeframe - 1]
#ax2.tick_params(axis ='y', labelcolor = color)

plt.title("Inflow profile to TES")
plt.show()


# Creating plot for tes capacity
fig, cap = plt.subplots()

cap.set_xlabel("Hours")
cap.set_ylabel("TES Capacity [MWh]")
cap.bar(hourslist[:yo], val_capacity[:yo], color = 'k')
#ax1.bar(hourslist[:744], val_tes[:744], color = 'g')
#ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
cap2 = cap.twinx()
 
cap2.set_ylabel('Power prices [NOK/MWh]')
cap2.plot(hourslist[:yo], power_prices[:yo], color = 'b') #[:timeframe - 1]
#ax2.tick_params(axis ='y', labelcolor = color)

plt.title("Capacity profile of TES")
plt.show()

