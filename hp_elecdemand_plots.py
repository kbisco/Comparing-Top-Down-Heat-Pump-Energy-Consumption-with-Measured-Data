#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:27:51 2022

@author: kbiscoch
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
import os
os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/Comparing-Top-Down-Heat-Pump-Energy-Consumption-with-Measured-Data')
from thermalcomfort_demand_IN import tc_energy_IN
import hp_model as m

# #%% Get thermal comfort for DC house
tc_df20, bauec20 = tc_energy_IN(2020, DC=True)
tc_df21, bauec21 = tc_energy_IN(2021, DC=True)
tc_df22, bauec22 = tc_energy_IN(2022, DC=True)
tc_df_all = pd.concat([tc_df20, tc_df21, tc_df22], axis = 0, ignore_index=True)
bauec_all = np.row_stack((bauec20,bauec21,bauec22))

tc_df_all.to_csv('tc_allyears_hourly.csv')
np.savetxt('bauec_total_allyears_hourly.csv', bauec_all, delimiter=',')

#%% Load previously saved thermal comfort data for DC House
tc_df_all = pd.read_csv("tc_allyears_hourly.csv", delimiter = ',', index_col=0)
bauec = np.loadtxt('bauec_total_allyears_hourly.csv', delimiter=',')
bauec = np.transpose(np.array(bauec, ndmin=2))

#%% Calculate heat pump energy consumption for DC house
def hpec_IN(year, tc_data, DC, hp, timestep, xaxis):
    if DC == True:
        tc_data = tc_data * 208
    
    if hp == 'ave':
        coeffs = m.get_coeffs_c(1)
        coeffs_heating = coeffs[1]
        coeffs_cooling = coeffs[0]
    elif hp == 'test':
        coeffs_heating = [0.07686, 2.456]
        coeffs_cooling = [-0.08118, 5.8698]
        # Cooling: -0.08118 * T_C + 5.8698
        # Heating: 0.07686 * T_C + 2.456
    if year == 'allyears':
        data = tc_data.iloc[0:25584]  
        # Obtain all temperatures for 2020-2021
        temp_files = ['ct_temps_IN_2020.gz','ct_temps_IN_2021.gz','ct_temps_IN_2022.gz']
        DC_temps_all = []
        for file in temp_files:
            ct_temps_data = np.loadtxt(file, delimiter = ',') # DC House: 2020-2022?
            ct_temps_data = ct_temps_data[:, ct_temps_data[0].argsort()]
            ct_temps_np = ct_temps_data[1:,:] - 273.15
            DC_tract = np.where(ct_temps_data[0,:] == 18157005400)
            DC_temp = ct_temps_np[:, DC_tract[0]]
            DC_temps_all.append(DC_temp)
        temp_flat = list(chain(*DC_temps_all))[0:25584]
        temp_flat2 = np.array(list(chain(*temp_flat)))
        temp_flat3 = np.reshape(temp_flat2,(-1,1)) # Reshape to one long list
        # flatlist=[]
        # for sublist in DC_temps_all:
        #     for element in sublist:
        #         flatlist.append(element)
        ref_temp = 18.3 # Celsius
        deltaT_heat = ref_temp - temp_flat3 # temp diff for heating mode (removing negatives)
        deltaT_heat[deltaT_heat < 0] = 0
        T_heat = deltaT_heat.copy()
        T_heat[deltaT_heat > 0] = temp_flat3[deltaT_heat > 0] # temperatures that are greater than ref temp.
        
        
        deltaT_cool = temp_flat3 - ref_temp # temp diff for AC (cooling) mode
        deltaT_cool[deltaT_cool < 0] = 0
        T_cool = deltaT_cool.copy()
        T_cool[deltaT_cool >= 0] = temp_flat3[deltaT_cool >= 0] # temperatures that are less than ref temp (removing negatives)
        
        heating_COP = coeffs_heating[0] * (T_heat) + coeffs_heating[1]
        # heating_COP = np.reshape(heating_COP, (-1,))
        cooling_COP = coeffs_cooling[0] * T_cool + coeffs_cooling[1]
        # cooling_COP = np.reshape(cooling_COP, (-1,))
        all_COP = deltaT_cool.copy()
        all_COP[deltaT_cool >= 0] = cooling_COP[deltaT_cool >= 0]
        all_COP[deltaT_heat > 0] = heating_COP[deltaT_heat > 0]
        all_COP = np.reshape(all_COP,(-1,))
        
        hpec = np.zeros(data.shape)
        
        for column in np.arange(0,data.shape[1]):
            column_df = np.array(data.iloc[:,column])
            # column_df = np.reshape(column_df, (-1,1))
            hpec[:,column] = column_df / all_COP
            # heating_adjusted_data = (column_df / heating_COP)
            # cooling_adjusted_data = (column_df / cooling_COP)
            # hpec[:,column] = heating_adjusted_data + cooling_adjusted_data       
                
        time = np.arange(0,25584,1)
        if timestep == 'month':
            days_in_month = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                                      31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                                      31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]) # Set up number of days in each month
            hours_in_month = days_in_month * 24
            def monthly_ave(data, months):
                month_ave = np.zeros((months,data.shape[1]))
                start = 0
                for month, hours in enumerate(hours_in_month):
                    step = start + hours
                    entry = np.mean(data[start:step],axis=0)
                    month_ave[month] = entry
                    start = step
                return month_ave
            y = monthly_ave(hpec, 35)
            y2 = monthly_ave(bauec[0:25584], 35)
            # y = np.zeros((35,5))
            # for column in range(6):
            #     y_temp = monthly_ave(hpec[:,column], 35)
            #     y[:,column] = y_temp
            y = pd.DataFrame(columns = ['total_ec','total_elec_ec', 
                                            'total_ff_ec', 'total_heating_ec', 'total_cooling_ec'],
                                 data = y)
            temp = monthly_ave(temp_flat3, 35)
            time = np.arange(0,35,1)
            
        if timestep == 'day':
            
            e_total = []
            e_daily = []
            daily = 0
            counter = 1
            
            for i in hpec[:,0]:
                if (counter % 24 == 0):
                    daily += i
                    counter = 1
                    e_daily.append(daily)
                    daily = 0
                    continue
            
                daily += i
                counter += 1
                
    else:
        temp_file = 'ct_temps_IN_' + str(year) + '.gz'
        ct_temps_data = np.loadtxt(temp_file, delimiter = ',') # DC House: 2020-2022?
        ct_temps_data = ct_temps_data[:, ct_temps_data[0].argsort()]
        ct_temps_np = ct_temps_data[1:,:] - 273.15
        DC_tract = np.where(ct_temps_data[0,:] == 18157005400)
        DC_temp = ct_temps_np[:, DC_tract[0]]
        temp = DC_temp
        y = tc_data
        if year == 2020:
            time = np.arange(0,8784,1)
        elif year == 2021:
            time = np.arange(0,8760,1)
        elif year == 2022:
            y = tc_data.iloc[0:8016]
            # time = np.arange(0,8760,1)
            time = np.arange(0,8016,1)
            temp = DC_temp[0:8016]
        
    if xaxis == 'time' and timestep == 'month':
        x = time
        xlab = 'Time [months]'        
    else:
        x = time
        xlab = 'Time [hours]'
    
    if xaxis == 'time':
        plt.figure(1)
        plt.plot(x, y['total_ec'], '-o')
        plt.xlabel(xlab)
        plt.ylabel('Average Monthly Heat Pump Electricity [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_totalecVStime.png', dpi=600)
        
        plt.figure(2)
        plt.plot(x, y['total_heating_ec'], '-o')
        plt.xlabel(xlab)
        plt.ylabel('Heating Energy Consumption [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_heatingecVStime.png', dpi=600)
        
        plt.figure(3)
        plt.plot(x, y['total_cooling_ec'], '-o')
        plt.xlabel(xlab)
        plt.ylabel('Cooling Energy Consumption [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_coolingecVStime.png', dpi=600)
        
    elif xaxis == 'temp':
        x = temp
        xlab = 'Temperature [Celsius]'
        # y2 = np.loadtxt('total_monthly_hpec_avedata.csv', delimiter = ',')
        
        plt.figure(1)
        plt.scatter(x, y['total_ec'])
        plt.xlabel(xlab)
        plt.ylabel('Average Monthly Heat Pump Electricity [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_totalecVStemp.png', dpi=600)
       
        plt.figure(2)
        plt.scatter(x, y['total_heating_ec'])
        plt.xlabel(xlab)
        plt.ylabel('Heating Energy Consumption [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_heatingecVStemp.png', dpi=600)
       
        plt.figure(3)
        plt.scatter(x, y['total_cooling_ec'])
        plt.xlabel(xlab)
        plt.ylabel('Cooling Energy Consumption [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_coolingecVStemp.png', dpi=600)
        
        plt.figure(4)
        plt.scatter(x, y['total_ec'], label='Ave Heat Pump Model', c='tab:orange')
        plt.scatter(x, y2, label='Business as Usual', c='tab:purple')
        plt.legend()
        plt.xlabel(xlab)
        plt.ylabel('Heat Pump Electricity Consumption [kWh/sqft]')
        ax = plt.gca()
        # ax.set_ylim([0, 7])
        plt.savefig('figure4.png', dpi=600, bbox_inches='tight')
        plt.savefig('figure4.eps', bbox_inches='tight')
    # return y2['total_ec']

    
#%% 
hpec_IN(year='allyears', tc_data=tc_df_all, DC=False, hp='ave', timestep='month', xaxis='temp')
hpec_IN(year='allyears', tc_data=tc_df_all, DC=True, hp='ave', timestep='month', xaxis='temp')
hpec_IN(year='allyears', tc_data=tc_df_all, DC=True, hp='ave', timestep='month', xaxis='time') 

hpec_IN(year='allyears', tc_data=tc_df_all, DC=True, hp='test', timestep='month', xaxis='temp') 
hpec_IN(year='allyears', tc_data=tc_df_all, DC=True, hp='test', timestep='month', xaxis='time') 
