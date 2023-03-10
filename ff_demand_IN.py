# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:04:34 2022

@author: kelsi
"""

#%% Create model for natural gas consumption using 2010 data
import numpy as np
import pandas as pd
import censusdata
#from eiapy import Series 
from scipy.optimize import lsq_linear
import requests
import os
import json

os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/Comparing-Top-Down-Heat-Pump-Energy-Consumption-with-Measured-Data')

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

f = open('eiaToken.json')
API_KEY = json.load(f)['token']

def getEIAdata(link, variable, version):
    if version ==2:
        broken=link.split('?')
        url2 = broken[0] + '?api_key=' + API_KEY + '&' + broken[1]
        response = requests.get(url2)
        x = response.json()
        y = x['response']
        z = y['data']
        output = [d[variable] for d in z]
        output.reverse()
    else:
        broken=link.split('?api_key=YOUR_API_KEY_HERE&')
        url2 = broken[0] + '?api_key=' + API_KEY + '&' + broken[1]
        response = requests.get(url2)
        x = response.json()
        y = x['series']
        z = y[0]['data']
        z2 = np.array(z)
        output = int(z2[np.where(z2[:,0] == variable)[0][0],1])*1000
    return output
    
ng = getEIAdata('https://api.eia.gov/v2/natural-gas/cons/sum/data/?frequency=monthly&data[0]=value&facets[duoarea][]=SIN&facets[series][]=N3010IN2&start=2010-01&end=2010-12&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000', 'value', 2)

# prop = getEIAdata('http://api.eia.gov/series/?api_key=YOUR_API_KEY_HERE&series_id=SEDS.PQRCB.IN.A', '2010', 1)
# print(prop)
# heatOil = getEIAdata('http://api.eia.gov/series/?api_key=YOUR_API_KEY_HERE&series_id=SEDS.DFRCB.IN.A', '2010', 1)
# print(heatOil)

fuel_oil_df = pd.read_csv('residential_distillate_fuel_oil.csv') # unit is thousand gallons, not MGAL
fuel_oil_IN = np.array(fuel_oil_df.loc[fuel_oil_df['area-name']=='IN', 'value'])
fuel_oil_IN = fuel_oil_IN * 1000 / 42 * 5.775 # mmbtu

kerosene_df = pd.read_csv('residential_kerosene.csv') # unit is thousand gallons, not MGAL
kerosene_IN = np.array(kerosene_df.loc[kerosene_df['area-name']=='IN', 'value'])
kerosene_IN = kerosene_IN * 1000 / 42 * 5.670 # mmbtu

propane_df = pd.read_csv('residential_propane.csv') # thousand barrels
propane_IN = np.array(propane_df.loc[propane_df['stateId']=='IN', 'value'])
propane_IN = propane_IN * 1000 * 3.841 # mmbtu 
A = np.column_stack((A0,A1))
# EIA uses MMcf, Waite uses MMbtu
b_ng = np.array(ng)*1037
print(b_ng)
b_scale = np.sum(b_ng) + propane_IN + fuel_oil_IN + kerosene_IN #2619439.65 + 17282000 # i couldn't find these numbers on the eia api 
b_scale = b_scale / np.sum(b_ng)
b = b_scale * b_ng
lsq_result = lsq_linear(A, b)
print(lsq_result)
print("A is", A)
#%% Saving coefficients multiplied by use percentages
print(lsq_result['x'])
weight2_ct = ff_perc[:,0] * lsq_result['x'][1]
print(weight2_ct)

print(ff_perc[:,0])
np.savetxt("heat_weight_ff_INct.csv", weight2_ct, delimiter=",")
