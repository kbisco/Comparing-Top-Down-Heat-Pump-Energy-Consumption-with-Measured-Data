# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:05:05 2022

@author: kelsi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:19:44 2022

@author: kelsi
"""

#%% Create model for electricity consumption using 2010 data
import numpy as np
# from sklearn import linear_model
import pandas as pd
import censusdata
from scipy.optimize import lsq_linear
from csv import writer

ct_temps_np = np.loadtxt('ct_temps_IN_2010.gz', delimiter = ',') # Load census tract temperature data from mult_days_areal_interp.py
ct_temps_np[1:,:] = ct_temps_np[1:,:] - 273.15 # Celsius
days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) # Set up number of days in each month

#%% Access and prepare heating fuel usage by percentage of households ##
## Commented lines give all categories of heating fuel ##
census_var=['B25040_001E','B25040_004E']
# census_var=['B25040_001E','B25040_002E','B25040_003E','B25040_004E',
#             'B25040_005E','B25040_006E','B25040_007E','B25040_008E','B25040_009E','B25040_010E']
hf_house = censusdata.download('acs5', 2010, censusdata.censusgeo([('state','18'),('tract','*')]),census_var,tabletype='detail',endpt='new')
column_names = ['total_hf','electricity'] # utility gas assumed to be natural gas (Waite)
# column_names = ['total_hf','utility_gas','bottled_tank_OR_LPgas','electricity','fueloil_kerosene_etc',
#                'coal_or_coke','wood','solar_energy','other_fuel','no_fuel'] 
hf_house.columns = column_names
hf_house['perc_elec'] = hf_house['electricity'].div(hf_house['total_hf'])

# Organizing indices of heating fuel usage percentage into readable census tract numbers #
ct_names = []

for index in hf_house.index.tolist():
    ct_name = index.geo[0][1] + index.geo[1][1] + index.geo[2][1]
    ct_names.append(float(ct_name))

elec_perc = hf_house['perc_elec']
elec_perc = elec_perc.to_frame() # series to df
elec_perc['ct_name'] = ct_names
elec_perc = elec_perc.to_numpy()

#%% Load and process HAZUS housing square footage data ##
sqft = pd.read_excel('IN_sqft_HAZUS.xls')
sqft_df =  sqft.iloc[:, np.r_[0, 23:31]]
sqft_np = sqft_df.to_numpy()
sqft_np = sqft_np[sqft_np[:,0].argsort()]


mult_bldg_m2 = np.multiply(sqft_np[:,1:], 92.903)
m2_total = sqft_np[:,0].copy()
m2_total = np.column_stack((m2_total,np.sum(mult_bldg_m2, axis=1))) 

#%% Calculate percentage AC
CDD_ct = np.loadtxt('CDD_ct_IN.csv', delimiter=',')
b = 0.005796761698918909
n = 0.8256586818786841

ac_perc = np.zeros((elec_perc.shape[0],2))
for count, value in enumerate(CDD_ct[:,1]):
    entry = 1-np.exp(-b*(value**n))
    ac_perc[count,0] = CDD_ct[count,0]
    ac_perc[count,1] = entry

#%% Sort data by census tract
elec_perc = elec_perc[elec_perc[:,1].argsort()]
ambient_temp = ct_temps_np[:, ct_temps_np[0].argsort()]
ref_temp = 18.3 # Celsius

#%% Dealing with missing variables and calculate deltaT
c = set(ambient_temp[0,:]).difference(m2_total[:,0])
missing_ct = np.array([list(c),np.zeros(len(c))])
m2_total = np.append(m2_total,missing_ct.T, axis=0)
m2_total = m2_total[m2_total[:,0].argsort()]
elec_perc = np.nan_to_num(elec_perc)

deltaT_heat = ref_temp - ambient_temp[1:,:] # temp diff for heating mode
deltaT_heat[deltaT_heat < 0] = 0
deltaT_cool = ambient_temp[1:,:] - ref_temp # temp diff for AC (cooling) mode
deltaT_cool[deltaT_cool < 0] = 0

#%% Calculations
## Caluclating A0 - Constant term 
A0 = 24 * np.sum(m2_total[:,1]) * days_in_month

## Temperature summation terms in coefficient of A1 (and A2) for all months and census tracts##
# Calculating A1 - coefficient of cooling mode data
deltaT_cool_sum = np.zeros([12,ambient_temp.shape[1]])
start = 0

# Summing temperature differences across 12 months for each census tract?
for month, days in enumerate(days_in_month):
    step = start + 24*days
    x = np.array(np.sum(deltaT_cool[start:step,:], axis=0))
    deltaT_cool_sum[month,:] = x.copy()
    start = step

# Multiplying census tract household square footage, ac percentage, and deltaT for one month and summing across all census tracts
A1 = np.empty(days_in_month.shape)
const1 = np.multiply(m2_total[:,1],ac_perc[:,1])
for month, days in enumerate(days_in_month):
    term = np.sum(np.multiply(const1.T, deltaT_cool_sum[month,:]))
    A1[month] = term

# Calculating A2 - coefficient of heating mode data
deltaT_heat_sum = np.zeros([12,ambient_temp.shape[1]])
start = 0

for month, days in enumerate(days_in_month):
    step = start + 24*days
    x = np.array(np.sum(deltaT_heat[start:step,:], axis=0))
    deltaT_heat_sum[month,:] = x.copy()
    start = step
    
A2 = np.empty(days_in_month.shape)
const2 = np.multiply(m2_total[:,1],elec_perc[:,0])
for month, days in enumerate(days_in_month):
    term = np.sum(np.multiply(const2.T, deltaT_heat_sum[month,:]))
    A2[month] = term

A = np.column_stack((A0,A1,A2))

b= np.loadtxt('Retail_sales_of_electricity_IN.csv',skiprows=10,usecols=(3,4,5,6,7,8,9,10,11,12,13,14),delimiter=',') #million kwh
b = b * 10**6 # kwh
lsq_result = lsq_linear(A, b)
print(lsq_result)
print("A is", A)
coeffs = lsq_result['x'].tolist()
#%% Saving coefficients multiplied by use percentages

weight1_ct = 0.00448735
weight2_ct = ac_perc[:,1] * 0.00188293
weight3_ct = elec_perc[:,0] * 0.00179322

np.savetxt("cool_weight_elec_INct.csv", weight2_ct, delimiter=",")
np.savetxt("heat_weight_elec_INct.csv", weight3_ct, delimiter=",")
