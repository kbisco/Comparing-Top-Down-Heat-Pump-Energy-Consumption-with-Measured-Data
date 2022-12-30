# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:04:34 2022

@author: kelsi
"""

#%% Create model for natural gas consumption using 2010 data
import numpy as np
import pandas as pd
import censusdata
from scipy.optimize import lsq_linear

ct_temps_np = np.loadtxt('ct_temps_IN_2010.gz', delimiter = ',') # Load census tract temperature data from mult_days_areal_interp.py
ct_temps_np[1:,:] = ct_temps_np[1:,:] - 273.15 # Celsius
days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) # Set up number of days in each month

#%% Access and prepare heating fuel usage by percentage of households ##
## Commented lines give all categories of heating fuel ##
census_var=['B25040_001E','B25040_002E','B25040_003E','B25040_005E']
# census_var=['B25040_001E','B25040_002E','B25040_003E','B25040_004E',
#             'B25040_005E','B25040_006E','B25040_007E','B25040_008E','B25040_009E','B25040_010E']
hf_house = censusdata.download('acs5', 2010, censusdata.censusgeo([('state','18'),('tract','*')]),census_var,tabletype='detail',endpt='new')
column_names = ['total_hf','utility_gas','bottled_tank_OR_LPgas','fueloil_kerosene_etc'] # utility gas assumed to be natural gas (Waite)
# column_names = ['total_hf','utility_gas','bottled_tank_OR_LPgas','electricity','fueloil_kerosene_etc',
#                'coal_or_coke','wood','solar_energy','other_fuel','no_fuel'] 
hf_house.columns = column_names
hf_house['total_ff'] = hf_house['utility_gas'] + hf_house['bottled_tank_OR_LPgas'] + hf_house['fueloil_kerosene_etc']
hf_house['perc_ff'] = hf_house['total_ff'].div(hf_house['total_hf'])

# Organizing indices of heating fuel usage percentage into readable census tract numbers #
ct_names = []

for index in hf_house.index.tolist():
    ct_name = index.geo[0][1] + index.geo[1][1] + index.geo[2][1]
    ct_names.append(float(ct_name))

ff_perc = hf_house['perc_ff']
ff_perc = ff_perc.to_frame() # series to df
ff_perc['ct_name'] = ct_names
ff_perc = ff_perc.to_numpy()

#%% Load and process HAZUS housing square footage data ##
sqft = pd.read_excel('IN_sqft_HAZUS.xls')
sqft_df =  sqft.iloc[:, np.r_[0, 23:31]]
sqft_np = sqft_df.to_numpy()
sqft_np = sqft_np[sqft_np[:,0].argsort()]


mult_bldg_m2 = np.multiply(sqft_np[:,1:], 92.903)
m2_total = sqft_np[:,0].copy()
m2_total = np.column_stack((m2_total,np.sum(mult_bldg_m2, axis=1))) 

#%% Sort data by census tract
ff_perc = ff_perc[ff_perc[:,1].argsort()]
ambient_temp = ct_temps_np[:, ct_temps_np[0].argsort()]
ref_temp = 18.3 # Celsius

#%%

c = set(ambient_temp[0,:]).difference(m2_total[:,0])
missing_ct = np.array([list(c),np.zeros(len(c))])
m2_total = np.append(m2_total,missing_ct.T, axis=0)
m2_total = m2_total[m2_total[:,0].argsort()]

ff_perc = np.nan_to_num(ff_perc)
temp = ambient_temp.copy()
deltaT = -1 * ambient_temp[1:,:] + ref_temp
deltaT[deltaT < 0] = 0
temp[1:,:] = deltaT.copy()
#%% Calculations
## b for all months ##
A0 = 24 * np.sum(np.multiply(m2_total[:,1],ff_perc[:,0])) * days_in_month

## Temperature summation terms in coefficient of A1 for all months and census tracts##
deltaT_sum = np.zeros([12,ambient_temp.shape[1]])
start = 0

for month, days in enumerate(days_in_month):
    step = start + 24*days
    x = np.array(np.sum(deltaT[start:step,:], axis=0))
    deltaT_sum[month,:] = x.copy()
    start = step

A1 = np.empty(days_in_month.shape)
const = np.multiply(m2_total[:,1],ff_perc[:,0])
for month, days in enumerate(days_in_month):
    term = np.sum(np.multiply(const.T, deltaT_sum[month,:]))
    A1[month] = term
    
A = np.column_stack((A0,A1))
# EIA uses MMcf, Waite uses MMbtu
b_ng = np.array([30877, 24542, 14592, 6144, 4395, 2659, 2412, 2494, 2427, 5133, 14191, 28547]) * 1037 # mmbtu 
b_scale = np.sum(b_ng) + 2619439.65 + 17282000 
b_scale = b_scale / np.sum(b_ng)
b = b_scale * b_ng
lsq_result = lsq_linear(A, b)
print(lsq_result)
print("A is", A)
#%% Saving coefficients multiplied by use percentages

weight2_ct = ff_perc[:,0] * 7.60632189e-06
np.savetxt("heat_weight_ff_INct.csv", weight2_ct, delimiter=",")

