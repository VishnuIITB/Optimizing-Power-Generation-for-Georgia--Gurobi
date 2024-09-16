
'''

Managing the supply and demand of electricity can be a complex and challenging task. Suppose that we are in charge of
generating power for the U.S. State of Georgia. Assume that we know the set of all available power plants and the demand
for power for each hour of a day. Can we create a schedule to decide how much power each plant should generate, and when
to switch the plants "on" and "off"? How can we do so while minimizing the overall costs?

'''

from PIL import Image
import requests
from io import BytesIO

print('Nuclear power plants. Source: Daniel Prudek / Shutterstock).')
img = Image.open(BytesIO(requests.get('https://github.com/Gurobi/modeling-examples/blob/master/power_generation/image_powerplant.png?raw=true').content))
print(img)


import gurobipy as gp
from gurobipy import GRB
model = gp.Model("powergeneration")

# %pip install networkx seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Importing the dataset
df_load_curves = pd.read_csv('https://github.com/Gurobi/modeling-examples/blob/master/power_generation/demand.csv?raw=true')
df_load_curves.info()
# Select the subset of data (Select data for July 1st, 2011)
print(len(df_load_curves[(df_load_curves['YEAR']==2004)]))
print(len(df_load_curves[(df_load_curves['YEAR']==2005)]))
print(len(df_load_curves[(df_load_curves['YEAR']==2006)]))
print(len(df_load_curves[(df_load_curves['YEAR']==2007)]))
print(len(df_load_curves[(df_load_curves['YEAR']==2008)]))
print(len(df_load_curves[(df_load_curves['YEAR']==2009)]))
print(len(df_load_curves[(df_load_curves['YEAR']==2010)]))
print(len(df_load_curves[(df_load_curves['YEAR']==2011)]))
print(len(df_load_curves[(df_load_curves['YEAR']==2012)]))
print(len(df_load_curves[(df_load_curves['YEAR']==2013)]))

df_subset = df_load_curves[(df_load_curves['YEAR']==2011) & (df_load_curves['MONTH']==7) & (df_load_curves['DAY']==1)]
print(df_subset)
# Store the demand to a dictionary
dem_1_July_2011 = df_subset.set_index(['HOUR']).LOAD.to_dict()
H = set(dem_1_July_2011.keys())
print(len(dem_1_July_2011))
print(dem_1_July_2011)
# plot demand over HOUR
# fig, ax = plt.subplots(figsize=(12,6))
# demand_plot = sns.barplot(x=list(range(1,25)), y=[dem_1_day[h] for h in range(1,25)])
# demand_plot.set_xticklabels(demand_plot.get_xticklabels());
# demand_plot.set(xlabel='Hour', ylabel='Demand (MWh)');
# plt.show()

# Load plot data
df_plant_info = pd.read_csv('https://github.com/Gurobi/modeling-examples/blob/master/power_generation/small_plant_data/plant_capacities.csv?raw=true')
print(df_plant_info)
P = set(df_plant_info['Plant'].unique())     # Set of all plants
plant_type = df_plant_info.set_index('Plant').PlantType.to_dict      # plat type for each plant
P_N = set(df_plant_info[df_plant_info['PlantType']=='NUCLEAR']['Plant'])     # set of all nuclear plants
fuel_type = df_plant_info.set_index('Plant').FuelType.to_dict()        # fuel type of each plant
print(P)
print(plant_type)
print(P_N)
print(fuel_type)

df_plant_info['capacity'] = df_plant_info['Capacity']
c = df_plant_info.set_index('Plant').capacity.to_dict() # generation capacity
capacity_plot = sns.barplot(x=list(c.keys()), y=[c[k] for k in c])
capacity_plot.set_xticklabels(capacity_plot.get_xticklabels(), rotation=40);
capacity_plot.set(xlabel='Plant', ylabel='Capacity (MWh)');
# plt.show()

# Moreover, if a nuclear power plant is turned "on," it has to generate at least 80% of its maximum capacity.
# For the rest of the plants, we set this minimum limit to 1%.
m = {i: 0.8 if i in P_N else 0.01 for i in P} # min % generation when on



# ramp up / ramp down speed
r = {i: 1 if i in ['BIOMASS','GAS','HYDRO','OIL'] else 0.2 if i in P_N else 0.25 for i in P}  # ramp up/down speed (plant)


# Cost data
df_fuel_costs = pd.read_csv('https://github.com/Gurobi/modeling-examples/blob/master/power_generation/small_plant_data/fuel_costs.csv?raw=true')
# df_fuel_costs = pd.read_csv('small_plant_data/fuel_costs.csv')

# read the fuel costs and transform it from fuel-type to plant-name
f = {i: df_fuel_costs[df_fuel_costs['year']==2011].T.to_dict()[9][fuel_type[i]] for i in fuel_type} # dictionary of fuel cost for each plant
# plot the fuel costs
fuelcost_plot = sns.barplot(x=list(f.keys()), y=[f[k] for k in f])
fuelcost_plot.set_xticklabels(fuelcost_plot.get_xticklabels(), rotation=40);
fuelcost_plot.set(xlabel='Plant', ylabel='Fuel cost per MWh ($)');
# plt.show()

# Operational Cost
df_oper_costs = pd.read_csv('https://github.com/Gurobi/modeling-examples/blob/master/power_generation/small_plant_data/operating_costs.csv?raw=true')
# df_oper_costs = pd.read_csv('small_plant_data/operating_costs.csv')
o = {i: df_oper_costs[df_oper_costs['year']==2011].T.to_dict()[9][fuel_type[i]] for i in fuel_type} # operating cost/MWh (plant)

# startup and shutdown cost
df_startup_costs = pd.read_csv('https://github.com/Gurobi/modeling-examples/blob/master/power_generation/small_plant_data/startup_costs.csv?raw=true')
# df_startup_costs = pd.read_csv('small_plant_data/startup_costs.csv')
s = {i: df_startup_costs[df_startup_costs['year'] == 2011].T.to_dict()[9][fuel_type[i]] for i in fuel_type}  # operating cost/MWh (plant)
t = s.copy()  # assume that the cost of shuting down = starting up

# Health Cost
df_health_costs = pd.read_csv('https://github.com/Gurobi/modeling-examples/blob/master/power_generation/small_plant_data/health_costs.csv?raw=true')
# df_health_costs = pd.read_csv('small_plant_data/health_costs.csv')
a = df_health_costs[(df_health_costs['Year']==2007)&(df_health_costs['Day']==1)].set_index(['Plant','Hour']).to_dict()['Cost'] # operating cost/MWh (plant)
a.update({(i,h): 0 for i in P for h in H if i not in ['Bowen','Jack McDonough','Scherer']})

fig, ax = plt.subplots(figsize=(15,6))
healthcost_plot = sns.barplot(x=list(range(1,25)), y=[a['Bowen',h] for h in range(1,25)])
healthcost_plot.set_xticklabels(healthcost_plot.get_xticklabels());
healthcost_plot.set(xlabel='Hour', ylabel='Health costs in Bowen ($)');
# plt.show()



# Declare decision variables
z = model.addVars(P, H, name = 'z', vtype = GRB.CONTINUOUS, lb=0)
u = model.addVars(P, H, name = 'u', vtype = GRB.BINARY)
v = model.addVars(P, H, name = 'v', vtype = GRB.BINARY)
w = model.addVars(P, H, name = 'w', vtype = GRB.BINARY)

print(len(z))
print(len(u))
print(len(v))
print(len(w))

# Define objective function
prob_objective = gp.quicksum(f[i]*z[i,h] for i in P for h in H)     # fuel cost
prob_objective += gp.quicksum(a[i,h]*z[i,h] for i in P for h in H)  # health cost
prob_objective += gp.quicksum(o[i]*u[i,h] for i in P for h in H)    # operation cost
prob_objective += gp.quicksum(s[i]*v[i,h] for i in P for h in H)    # startup cost
prob_objective += gp.quicksum(t[i]*w[i,h] for i in P for h in H)      # shutdown cost
model.setObjective(prob_objective, GRB.MINIMIZE)
print(prob_objective)

# Constraint 1 ; Meet demand
model.addConstrs((gp.quicksum(z[i,h] for i in P) == dem_1_July_2011[h]) for h in H)
model.update()

# Constraint 2 ;  Maximum and minimum power generation levels
model.addConstrs((z[i,h] <= c[i]*u[i,h]) for i in P for h in H)
model.addConstrs((z[i,h] >= m[i]*c[i]*u[i,h]) for i in P for h in H)
model.update()

# Constraint 3 ; Nuclear plants are always ON
model.addConstrs((z[i,h] >= m[i]*c[i]) for i in P_N for h in H)
model.update()

# Constraint 4 ; Ramp up / ramp down
model.addConstrs((z[i,h] - z[i,h-1] >= -r[i]*c[i]) for i in P for h in H if h>1)
model.addConstrs((z[i,h] - z[i,h-1] <=  r[i]*c[i]) for i in P for h in H if h>1)
model.update()

# Constraint 5 ; If switched OFF, must be ON and if OFF must be OFF
model.addConstrs((u[i,h] >= v[i,h]) for i in P for h in H)
model.addConstrs((w[i,h] >= 1- u[i,h]) for i in P for h in H)
model.update()

# Constraint 6 ; Linking startup / shutdown variable to ON/OFF variable
model.addConstrs((v[i,h] - w[i,h] == u[i,h] - u[i,h-1]) for i in P for h in H if h>1)
model.update()

# Solve the model
model.optimize()



