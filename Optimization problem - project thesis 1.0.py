# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:58:33 2023

@author: agrot

Version 1.0 of the project thesis optimization problem code
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import pandas as pd

# Create a concrete Pyomo model
model = pyo.ConcreteModel()

# Define the set of hours in the year
timeframe = 2000
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

# Create a variable for the power generation in each hour
model.power_generation = pyo.Var(hours, within=pyo.NonNegativeReals)


# ----- Objective function: Maximize the surplus (profit) ----- #

def objective_rule(model):
    return sum((power_prices[hour] - production_price) * model.power_generation[hour] for hour in hours)

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


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
gen_lowramp = -5
gen_maxramp = 5

def production_ramping_up(model, hour):
    if hour == 1:
        return pyo.Constraint.Skip
    else:
        return model.power_generation[hour] - model.power_generation[hour - 1] <= gen_maxramp
model.prod_rampup = pyo.Constraint(hours, rule=production_ramping_up)

def production_ramping_down(model, hour):
    if hour == 1:
        return pyo.Constraint.Skip
    else:
        return gen_lowramp <= model.power_generation[hour] - model.power_generation[hour - 1]
model.prod_rampdown = pyo.Constraint(hours, rule=production_ramping_down)


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
ax1.bar(hourslist[:745], val_gen[:745], color = 'r')
#ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
ax2 = ax1.twinx()
 
ax2.set_ylabel('Power prices [NOK/MWh]')
ax2.plot(hourslist[:745], power_prices[:745], color = 'b') #[:timeframe - 1]
#ax2.tick_params(axis ='y', labelcolor = color)

plt.title("Distribution of power generation")
plt.show()
