# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:58:33 2023

@author: agrot

Version 2.1.1 of the project thesis optimization problem code
- cleaning up and making some adjustments (pricing in obj.fnc., flow constraints)

This model code is far from finished.
Apologizes for the inconveniences caused by the lack of model build modularity

Most code for plots are included, but commented out because of the total 
additional run-time for plotting and the unclearity all the plots would create.
This allows for selection of plots based on needs.
"""

# %% ----- Interchangeable data - User Interface ----- #

# Year to examine
year = 2022 
# Use 2022 or 2023, earlier years could be implemented
# For 2022 hour [1, 8758], and for 2023 [1, 8352] are available

# Define the set of hours in the year (step1 = 1, tf = 8760)
# or define time interval to be examined
step1 = 1
tf = 8758 # not 8760 because of missing data

# Defining reactor heat extraction limits
gen_maxcap = 300
gen_lowcap = 0

# Production price (cost) in NOK/MWh (marginal cost)
MC_gen = 100.0  # Example, will be lower
MC_turbo = MC_gen * 1.1  # Example

# Defining ramping limits for electric generator
lower_ramp_lim = -5  # [%/min]
upper_ramp_lim = 5

# --- TES --- #

# TES storage duration
t_tes = 12  # [hours]

# Defining TES inflow limits
inflow_maxcap = 300  # [MW]
inflow_lowcap = 0

# Defining TES outflow limits
outflow_maxcap = 300  # [MW]
outflow_lowcap = 0

# Defining TES capacity
tes_maxcap = outflow_maxcap * t_tes  # [MWh]
tes_lowcap = 0


# %% ----- Dependencies ----- #

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


# %% ----- Model setup ----- #

# Create a concrete Pyomo model
model = pyo.ConcreteModel()

# List for hour range to be examined
# Assuming there are 8760 hours in a year (8761, one month: 745)
hours = range(step1, tf + 1)


# %% ----- Reading in data ----- #

def InputData(data_file):  # reading the excel file with the values of generation and cost

    if year == 2022:
        col = 'B'
    elif year == 2023:
        col = 'C'
    inputdata = pd.read_excel(data_file, usecols=col, na_values='-').dropna()
    power_prices = inputdata['price ' + str(year)].tolist()

    return power_prices


power_prices = InputData('Power prices 8760 1.0.xlsx')


# %% ----- Variables ----- #

model.power_generation       = pyo.Var(hours, within=pyo.NonNegativeReals) # power generation in each hour
model.power_generation_state = pyo.Var(hours, within=pyo.Binary)           # power generation state in each hour
model.basepow                = pyo.Var(hours, within=pyo.NonNegativeReals) # base power extract in each hour
model.outflow_tes            = pyo.Var(hours, within=pyo.NonNegativeReals) # TES outflow in each hour
model.inflow_tes             = pyo.Var(hours, within=pyo.NonNegativeReals) # inflow to TES in each hour
model.inflow_tes_state       = pyo.Var(hours, within=pyo.Binary)           # inflow to TES state in each hour
model.fuel_tes               = pyo.Var(hours, within=pyo.NonNegativeReals) # fuel level in TES in each hour
model.tes_full_state         = pyo.Var(hours, within=pyo.Binary)           # tes full state

# %% ----- Objective Function: Maximize the surplus (profit) ----- #

def objective_rule(model):
    return sum((power_prices[hour] - MC_gen) * model.basepow[hour] 
               + model.outflow_tes[hour] * (power_prices[hour] - MC_turbo) for hour in hours)
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


# %% ----- Reactor Heat Extract Constraints ----- #

def production_limit_upper(model, hour):
    return model.basepow[hour] + model.inflow_tes[hour] <= gen_maxcap
model.prod_limup = pyo.Constraint(hours, rule=production_limit_upper)


def production_limit_lower(model, hour):
    return gen_lowcap <= model.basepow[hour] + model.inflow_tes[hour]
model.prod_limlow = pyo.Constraint(hours, rule=production_limit_lower)


# %% ----- Ramping Constraints ----- #


# Converting ramping limits
gen_lowramp = lower_ramp_lim * 0.6 * gen_maxcap  # [MW/h]
gen_maxramp = upper_ramp_lim * 0.6 * gen_maxcap


def production_ramping_up(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return model.power_generation[hour] - model.power_generation[hour - 1] <= gen_maxramp
model.prod_rampup = pyo.Constraint(hours, rule=production_ramping_up)


def production_ramping_down(model, hour):
    if hour == step1:
        return pyo.Constraint.Skip
    else:
        return gen_lowramp <= model.power_generation[hour]- model.power_generation[hour - 1]

# %% ----- TES Energy Balance Constraint ----- #


def tes_balance(model, hour):
    if hour == step1:
        return model.fuel_tes[hour] == 0
    else:
        return model.fuel_tes[hour] == model.fuel_tes[hour - 1] + model.inflow_tes[hour - 1] - model.outflow_tes[hour - 1]
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


# %% ----- TES Capacity Limit state link and time constraint ----- #

# Constraint to link binary variable

def binary_tescap(model, hour):
    return model.fuel_tes[hour] >= tes_maxcap * model.tes_full_state[hour]
model.binary_constraintfullcap = pyo.Constraint(hours, rule=binary_tescap)

def binary_tescap2(model, hour):
    return model.fuel_tes[hour] <= (1 - model.tes_full_state[hour]) * (tes_maxcap - 1) + M * model.tes_full_state[hour]
model.binary_constraintfullcap2 = pyo.Constraint(hours, rule=binary_tescap2)


# %% ----- TES Flow Limit Constraints ----- #

def tes_inflow_limit_upper(model, hour):
    return model.inflow_tes[hour] <= inflow_maxcap
model.tesin_limup = pyo.Constraint(hours, rule=tes_inflow_limit_upper)


def tes_inflow_limit_lower(model, hour):
    return inflow_lowcap <= model.inflow_tes[hour]
model.tesin_limlow = pyo.Constraint(hours, rule=tes_inflow_limit_lower)


# ----- TES production limit constraints ----- #

def tes_outflow_limit_upper(model, hour):
    return model.outflow_tes[hour] <= outflow_maxcap
model.tesout_limup = pyo.Constraint(hours, rule=tes_outflow_limit_upper)


def tes_outflow_limit_lower(model, hour):
    return outflow_lowcap <= model.outflow_tes[hour]
model.tesout_limlow = pyo.Constraint(hours, rule=tes_outflow_limit_lower)


# %% ----- TES Capacity Constraint ----- #

def tes_cap_upper(model, hour):
    return model.fuel_tes[hour] <= tes_maxcap
model.tes_up_cap = pyo.Constraint(hours, rule=tes_cap_upper)


def tes_cap_lower(model, hour):
    return tes_lowcap <= model.fuel_tes[hour]
model.tes_low_cap = pyo.Constraint(hours, rule=tes_cap_lower)


# %% ----- Combining power constraint ----- #

def comb_power(model, hour):
    return model.power_generation[hour] == model.basepow[hour] + model.outflow_tes[hour]
model.combining_power = pyo.Constraint(hours, rule=comb_power)

# %% ----- Solving the optimization problem ----- #

opt = SolverFactory('gurobi')
opt.options['tee'] = True
opt.solve(model)

results = opt.solve(model, tee=True)

print('\n')

# %% ----- Printing and plotting results ----- #

print("Optimal Surplus: ", pyo.value(model.objective), "NOK")

# ---- Plotting distribution ---- #


hourslist             = list(hours)
val_gen               = list(pyo.value(model.power_generation[hour]) for hour in hours)
val_base              = list(pyo.value(model.basepow[hour]) for hour in hours)
val_tes               = list(pyo.value(model.outflow_tes[hour]) for hour in hours)
val_inflow            = list(pyo.value(model.inflow_tes[hour]) for hour in hours)
val_capacity          = list(pyo.value(model.fuel_tes[hour]) for hour in hours)
val_state             = list(pyo.value(model.power_generation_state[hour]) for hour in hours)
val_tes_full_state    = list(pyo.value(model.tes_full_state[hour]) for hour in hours)


# Most code for plots are included, but commented out 
# because of the total additional run-time for plotting.
# This allows for selection of plots based on needs.

"""
# Plotting price hourly
fig, ph = plt.subplots(figsize=(50, 6))
ph.set_ylabel('[NOK/MWh]', fontsize=40)
ph.tick_params(axis='both', labelsize=35)
plt.ylim(0, 800)
ph.plot(hourslist[:], power_prices[step1-1:tf],
        color='b')
ph2 = ph.twinx()
ph2.set_ylabel('', fontsize=40)
ph2.tick_params(axis='both', labelsize=35)
ph2.axhline(y = 100, color = 'k', linestyle = 'dashed', linewidth=2)
plt.ylim(0, 800)
ph2.set_yticks([])
plt.show()


# Plotting combined power generation
fig, com = plt.subplots(figsize=(50, 6))  # Set the figure width here
com.set_ylabel("[MW]", fontsize=40)  # Adjust font size for y-axis label
com.tick_params(axis='both', labelsize=35)  # Adjust font size for ticks
com.set_yticks([0, gen_maxcap - outflow_maxcap, gen_maxcap, gen_maxcap + outflow_maxcap])
com.plot(hourslist[:tf - step1], val_gen[:tf - step1], color='r')
plt.ylim(-0.04 * (gen_maxcap + inflow_maxcap), 1.04 * (gen_maxcap + inflow_maxcap))
com2 = com.twinx()
com2.set_ylabel('[%]', fontsize=40)
com2.tick_params(axis='both', labelsize=35)
com2.set_yticks([0, 100, 200, 300])
plt.ylim(- 0.04*(100/(gen_maxcap/(gen_maxcap+inflow_maxcap))), 1.04*(100/(gen_maxcap/(gen_maxcap+inflow_maxcap))))
plt.show()


# Duration curve
sort = np.sort(val_gen)[::-1]
plt.plot(hourslist[:tf - step1], sort[:tf - step1])
plt.ylim(- 0.04*(gen_maxcap+inflow_maxcap), 1.04*(gen_maxcap+inflow_maxcap))
plt.xlabel("Hours [t]")
plt.ylabel("Produced power [MW]")
dur2 = plt.twinx()
dur2.set_ylabel('% of baseload power')
dur2.axhline(y = 100, color = 'r', linestyle = 'dashed')
plt.ylim(- 0.04*(100/(gen_maxcap/(gen_maxcap+inflow_maxcap))), 1.04*(100/(gen_maxcap/(gen_maxcap+inflow_maxcap))))
dur2.set_yticks([0, 100])
plt.show()


# Creating plot for tes capacity
fig, cap = plt.subplots(figsize=(50, 6))
cap.set_xlabel("Hours [h]", fontsize=40)
cap.set_ylabel("[MWh]", fontsize=40)  
cap.tick_params(axis='both', labelsize=35)  
cap.set_yticks([tes_maxcap/2, tes_maxcap])
cap.plot(hourslist[:tf - step1], val_capacity[:tf - step1], color='y')
cap2 = cap.twinx()
cap2.set_ylabel('[%]', fontsize=40)
cap2.tick_params(axis='both', labelsize=35)
cap2.set_yticks([0, 50, 100])
plt.ylim(- 0.04*100, 1.04*100)
plt.show()
"""


# %% ----- Calculating value factor ----- #

sumprod = 0 
nprod = 0 

for h in hourslist:
    if model.power_generation[h].value > 0:
        sumprod += power_prices[h]
        nprod += 1
        
C_avgprod = sumprod/nprod
C_avg = sum(power_prices)/len(power_prices)

vf = C_avgprod/C_avg
print(vf)

# %% ----- Calculating capacity factor ----- #

totsum = 0

for h in hourslist:
    totsum += model.power_generation[h].value
    
cf = totsum/((tf+1-step1)*gen_maxcap)
print(cf)
# %% ----- Making monthly plots ----- # 

if step1 == 1 and tf == 8758:
    mon_lens = [744, 672, 744, 720, 744, 720, 744, 744, 719, 744, 719, 744]
    mon_ins = list(range(0, 12))
    
    tot_pow = []
    tot_base = []
    tot_tes = []
    avg_pp = []
    mon_tot_cap = []
    mon_cap_lim = []
    mon_cap_lim_rels = []
    
    h_passed = 0
    h_profit = 0
    
    for m in mon_ins:
        pp_t = 0
        for h in range(1 + h_passed, 1 + mon_lens[m] + h_passed):
            if h == 1 + h_passed:
                tot_pow.append(model.power_generation[h].value)
                tot_base.append(model.basepow[h].value)
                tot_tes.append(model.outflow_tes[h].value)
                mon_tot_cap.append(model.fuel_tes[h].value)
                mon_cap_lim.append(model.tes_full_state[h].value)
                pp_t += power_prices[h]
            else:
                tot_pow[mon_ins[m]]     += model.power_generation[h].value
                tot_base[mon_ins[m]]    += model.basepow[h].value
                tot_tes[mon_ins[m]]     += model.outflow_tes[h].value
                mon_tot_cap[mon_ins[m]] += model.fuel_tes[h].value
                mon_cap_lim[mon_ins[m]] += model.tes_full_state[h].value
                pp_t += power_prices[h]
                
            if power_prices[h] > MC_gen:
                h_profit += 1
                        
            avg_pp.append(pp_t/mon_lens[m])  
                        
            h_passed += mon_lens[m]
                        
            mon_cap_lim_rels.append(mon_cap_lim[m]/mon_lens[m])
                  
# Most code for plots are included, but commented out 
# because of the total additional run-time for plotting.
# This allows for selection of plots based on needs.          
            
    """
    # Plotting monthly plots
    fig, mly = plt.subplots()
    bottom = np.zeros(len(mon_ins))
                        
    mly.set_xlabel("Month nr.")
    mly.set_ylabel("Aggregate produced energy [GWh]")
    mly.set_xticks(range(1, 13))
    mly.set_xticklabels(range(1, 13))
    mly.bar(range(1, 13), np.array(tot_pow)/1000.0, color='r', width=0.8)
                        
    # Adding Twin Axes to plot using dataset_2
    mly2 = mly.twinx()

    mly2.set_ylabel('Average monthly power prices [NOK/MWh]')
    mly2.plot(range(1, 13), avg_pp, color='b')
    
    #plt.title("Monthly NPP energy production with TES")
    plt.show()
    
    -----------------------------
    
    #... cap 

    fig, mcap = plt.subplots()
    bottom = np.zeros(len(mon_ins))
    
    mcap.set_xlabel("Month nr.")
    mcap.set_ylabel("Stored thermal energy [GWh]")
    mcap.set_xticks(range(1, 13))
    mcap.set_xticklabels(range(1, 13))
    mcap.bar(range(1, 13), np.array(mon_tot_cap)/1000.0,
             color='y', width=0.8)
    
    #% of monthly hours where tes capacity is full
    
    fig, mpcap = plt.subplots()
    bottom = np.zeros(len(mon_ins))
    
    mpcap.set_xlabel("Month nr.")
    mpcap.set_ylabel("% of hours where TES capacity is reached [%]")
    mpcap.set_xticks(range(1, 13))
    mpcap.set_xticklabels(range(1, 13))
    mpcap.bar(range(1, 13), np.array(mon_cap_lim_rels)*100,
              color='orange', width=0.8)
    plt.show()
    ----------------------------------
    
    # Comb. plots
    
    fig, mlyc = plt.subplots()
    bottom = np.zeros(len(mon_ins))
    
    mlyc.set_xlabel("Month nr.")
    mlyc.set_ylabel("Aggregate produced energy [GWh]")
    mlyc.set_xticks(range(1, 13))
    mlyc.set_xticklabels(range(1, 13))
    mlyc.bar(range(1, 13), np.array(tot_base)/1000.0, bottom = bottom,
             color='r', width=0.8, label='Baseload')
    mlyc.bar(range(1, 13), np.array(tot_tes)/1000.0, bottom = bottom + np.array(tot_base)/1000.0,
             color='darkred', width=0.8, label='TES')
    plt.legend(loc="best")
    plt.show()
    """