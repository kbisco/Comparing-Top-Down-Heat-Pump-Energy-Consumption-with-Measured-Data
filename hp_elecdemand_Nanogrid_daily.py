#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:20:01 2022

@author: mohammadrezqalla
"""
import numpy as np
import hp_model as m

e_total_heating = []
e_total_cooling = []
coeffs = m.get_coeffs_c(1)
## Averaged coeffs
# coeffs_heating = coeffs[1]
# coeffs_cooling = coeffs[0]
## Measured data coeffs
coeffs_cooling = [-0.08118, 5.8698]
coeffs_heating = [0.07686000000000001, 2.456]

temp_values0 = np.loadtxt("ct_temps_IN_2020.gz", delimiter = ",")
temp_values1 = np.loadtxt("ct_temps_IN_2021.gz", delimiter = ",")[1:,:]
temp_values2 = np.loadtxt("ct_temps_IN_2022.gz", delimiter = ",")[1:,:]

temp_values = np.vstack((temp_values0, temp_values1, temp_values2))

energy_file = np.loadtxt('tc_allyears_hourly.csv', delimiter = ',', skiprows = 1)

##RELOOK CODE
temp_values_cooling = temp_values[:, temp_values[0].argsort()]
temp_values_heating = temp_values[:, temp_values[0].argsort()]
column_DC_house = np.where(temp_values_cooling[0,:] == 18157005400)
print(column_DC_house)

temp_values_heating[temp_values_heating > (18.333 + 273.15)] = 0
temp_values_cooling[temp_values_cooling <= (18.333 + 273.15)] = 0


cooling_raw_demand = energy_file[:, 5].reshape(len(temp_values[:,1]) - 1, 1)
heating_raw_demand = energy_file[:, 4].reshape(len(temp_values[:,1] ) - 1, 1)


#Heating
heating_COP = (coeffs_heating[0] * (temp_values_heating[1:,column_DC_house[0]] - 273.15)) + coeffs_heating[1]

heating_adjusted_data_house = (heating_raw_demand / heating_COP) * 208


#Cooling
cooling_COP = (coeffs_cooling[0] * (temp_values_cooling[1:,column_DC_house[0]] - 273.15)) + coeffs_cooling[1]

cooling_adjusted_data_house = (cooling_raw_demand / cooling_COP) * 208

   
e_daily_heat = []
e_daily_cool = []

daily = 0
counter = 1

for i in heating_adjusted_data_house:
    if (counter % 24 == 0):
        daily += i
        counter = 1
        e_daily_heat.append(daily)
        daily = 0
        continue
    
  
    daily += i
    counter += 1

daily = 0
counter = 1

for i in cooling_adjusted_data_house:
    if (counter % 24 == 0):
        daily += i
        counter = 1
        e_daily_cool.append(daily)
        daily = 0
        continue

    daily += i
    counter += 1


total = np.array(e_daily_heat) + np.array(e_daily_cool)

# np.savetxt("total_daily_hpec_avedata.csv", total, delimiter = ',')
np.savetxt("total_daily_hpec_testdata.csv", total, delimiter = ',')
