# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:58:33 2023

@author: agrot

Version 2.0 of the project thesis optimization problem code
- implementing TES
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import pandas as pd

# Create a concrete Pyomo model
model = pyo.ConcreteModel()

# Define the set of hours in the year
timeframe = 745
hours = range(1, timeframe)  # Assuming there are 8760 hours in a year (8761, one month: 745)

# Power prices for each hour in dollars per MWh
#power_prices = [10.0, 12.0, 15.0, 13.0, 11.0] #Example


# ----- Reading in data ----- #

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


# ----- Variables ----- # 

model.power_generation       = pyo.Var(hours, within=pyo.NonNegativeReals) # power generation in each hour
model.power_generation_state = pyo.Var(hours, within=pyo.Binary)          # power generation state in each hour
model.power_generation_tes   = pyo.Var(hours, within=pyo.NonNegativeReals) # power generation for TES in each hour
model.inflow_tes             = pyo.Var(hours, within=pyo.NonNegativeReals) # inflow to TES in each hour
model.fuel_tes               = pyo.Var(hours, within=pyo.NonNegativeReals) # fuel level in TES in each hour


# ----- Objective function: Maximize the surplus (profit) ----- #

def objective_rule(model):
    return sum((power_prices[hour] - production_price) * (model.power_generation[hour]) for hour in hours) #  + model.power_generation_tes[hour]

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


"""
def objective_rule(model):
    return sum((power_prices[hour]) * (model.power_generation[hour] + model.power_generation_tes[hour]) - (production_price) * (model.power_generation[hour] - model.inflow_tes[hour]) for hour in hours)

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
"""

# ----- Generation constraints ----- #

# Defining power production limits 
gen_maxcap = 300
gen_lowcap = 0

def production_limit_upper(model, hour):
    return model.power_generation[hour] <= gen_maxcap
model.prod_limup = pyo.Constraint(hours, rule=production_limit_upper)

def production_limit_lower(model, hour):
    return gen_lowcap <= model.power_generation[hour]
model.prod_limlow = pyo.Constraint(hours, rule=production_limit_lower)


# ----- Ramping constraints ----- #

# Defining ramping limits 
gen_lowramp = -5 #[MW/h]
gen_maxramp = 5

def production_ramping_up(model, hour):
    if hour == 1:
        return pyo.Constraint.Skip
    else:
        return (model.power_generation[hour]) - (model.power_generation[hour - 1]) <= gen_maxramp#(model.power_generation[hour] + model.power_generation_tes[hour]) - (model.power_generation[hour - 1] + model.power_generation_tes[hour - 1]) <= gen_maxramp
model.prod_rampup = pyo.Constraint(hours, rule=production_ramping_up)

def production_ramping_down(model, hour):
    if hour == 1:
        return pyo.Constraint.Skip
    else:
        return gen_lowramp <= (model.power_generation[hour]) - (model.power_generation[hour - 1]) #gen_lowramp <= (model.power_generation[hour] + model.power_generation_tes[hour]) - (model.power_generation[hour - 1] + model.power_generation_tes[hour - 1])
model.prod_rampdown = pyo.Constraint(hours, rule=production_ramping_down)


# ----- TES energy balance constraint ----- #

def tes_balance(model, hour): 
    if hour == 1:
        return pyo.Constraint.Skip #model.fuel_tes[hour] == 0
    else:
        return model.fuel_tes[hour] == model.fuel_tes[hour - 1] + model.inflow_tes[hour -1] - model.power_generation_tes[hour]
model.tes_energy_bal = pyo.Constraint(hours, rule=tes_balance)


# ----- TES inflow in/out constraint ----- # 

def tes_inflow(model, hour):
    if model.power_generation_state[hour]: # and gen_active = true
        return model.inflow_tes[hour] == 0 # [MWh]
model.tes_inflow = pyo.Constraint(hours, rule=tes_inflow)


# %%
"""
# Define the constraint to link is_positive with real_variable
def is_positive_constraint_rule(model, hour):
    return model.power_generation_state[hour] == (model.power_generation[hour] > 0)

model.is_positive_constraint = pyo.Constraint(model.hours, rule=is_positive_constraint_rule)

# Define the constraint using is_positive
def tes_inflow_constraint_rule(model, hour):
    return model.inflow_tes[hour] <= (1 - model.power_generation_state[hour]) * 0

model.tes_inflow_constraint = pyo.Constraint(model.hours, rule=tes_inflow_constraint_rule)
"""
M = 10**6
# Constraint to link bin_var with x
def binary_constraint_rule(hour, model):
    return model.power_generation[hour] <= model.power_generation_state[hour] * M  # M is a big-M constant

model.binary_constraint = pyo.Constraint(model.hours, rule=binary_constraint_rule)
# %%

"""
# ----- TES inflow limit constraints ----- #

# Defining TES inflow limits 
inflow_maxcap = 300
inflow_lowcap = 0

def tes_inflow_limit_upper(model, hour):
    return model.inflow_tes[hour] <= inflow_maxcap
model.tesin_limup = pyo.Constraint(hours, rule=tes_inflow_limit_upper)

def tes_inflow_limit_lower(model, hour):
    return inflow_lowcap <= model.inflow_tes[hour]
model.tesin_limlow = pyo.Constraint(hours, rule=tes_inflow_limit_lower)


# ----- TES production constraints ----- #

def tes_production(model, hour):
    if model.inflow_tes[hour] != 0 or model.fuel_tes[hour] == 0:
        return model.power_generation_tes[hour] == 0
model.tes_prod = pyo.Constraint(hours, rule=tes_production)


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


# ----- TES capacity constraint ----- #

tes_maxcap = 1500 # [MWh]
tes_lowcap = 0

def tes_cap_upper(model, hour):
    return model.fuel_tes[hour] <= tes_maxcap
model.tes_up_cap = pyo.Constraint(hours, rule=tes_cap_upper)

def tes_cap_lower(model, hour):
    return tes_lowcap <= model.fuel_tes[hour]
model.tes_low_cap = pyo.Constraint(hours, rule=tes_cap_lower)

"""
# ----- Solving the optimization problem ----- #

solver = pyo.SolverFactory('gurobi')
solver.solve(model)

# Print the results

#print("Optimal Surplus: ", pyo.value(model.objective))
#print("Optimal Power Generation:")
#for hour in hours:
#    print(f"Hour {hour}: {pyo.value(model.power_generation[hour])} MWh")
   
# ---- Plotting distribution ---- #

hourslist = list(hours)
val_gen = list(pyo.value(model.power_generation[hour]) for hour in hours)

jan= 744 #, mar, may, jul, aug, okt, dec
apr= 720 #, jun, sep, nov
feb= 672


#plt.figure().set_figwidth(15)
#plt.figure(figsize=(20,6))

# Creating plot with dataset_1
fig, ax1 = plt.subplots()

ax1.set_xlabel("Generation technologies")
ax1.set_ylabel("Generation output [MW]")
ax1.bar(hourslist[:744], val_gen[:744], color = 'r')
#ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
ax2 = ax1.twinx()
 
ax2.set_ylabel('Power prices [NOK/MWh]')
ax2.plot(hourslist[:744], power_prices[:744], color = 'b') #[:timeframe - 1]
#ax2.tick_params(axis ='y', labelcolor = color)

plt.title("Distribution of power generation")
plt.show()
