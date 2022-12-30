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
os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/hp_proj1')
from thermalcomfort_energdemandIN_debug import tc_energy_IN_debug
import model as m

# #%% Get thermal comfort for DC house
ec_df20 = tc_energy_IN_debug(2020, DC=True)
ec_df21 = tc_energy_IN_debug(2021, DC=True)
ec_df22 = tc_energy_IN_debug(2022, DC=True)
ec_df_all = pd.concat([ec_df20, ec_df21, ec_df22], axis = 0, ignore_index=True)

ec_df_all.to_csv('tc_allyears_hourly_dchouse8_hf.csv')

#%% Load previously saved thermal comfort data for DC House
ec_df_all = pd.read_csv("tc_allyears_hourly_dchouse8_hf.csv", delimiter = ',', index_col=0)

# plt.plot(ec_df20['total_ff_ec'])
# plt.plot(ec_df20['total_elec_ec'])

#%% Calculate heat pump energy consumption for DC house
def hpec_IN(year, tc_data, DC, hp, timestep, xaxis):
    os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/hp_proj1')
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
        os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/hp_proj1/researchletter_plots/DC_hp_energy_consumption_estimate_png/' + hp + 'hp_model/VS_time')
        plt.figure(1)
        plt.plot(x, y['total_ec'], '-o')
        # plt.scatter(x, y['total_ec'])
        plt.xlabel(xlab)
        # plt.ylabel('Heating + Cooling Energy Consumption [kWh]')
        plt.ylabel('Average Monthly Heat Pump Electricity [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        plt.savefig(hp + 'hpmodel_totalecVStime.png', dpi=600)
        
        plt.figure(2)
        plt.plot(x, y['total_elec_ec'], '-o')
        # plt.scatter(x, y['total_elec_ec'])
        plt.xlabel(xlab)
        plt.ylabel('Electricity Energy Consumption [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        plt.savefig(hp + 'hpmodel_elececVStime.png', dpi=600)
        
        plt.figure(3)
        plt.plot(x, y['total_ff_ec'], '-o')
        # plt.scatter(x, y['total_ff_ec'])
        plt.xlabel(xlab)
        plt.ylabel('Fossil Fuel Energy Consumption [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        plt.savefig(hp + 'hpmodel_ffecVStime.png', dpi=600)
        
        plt.figure(4)
        plt.plot(x, y['total_heating_ec'], '-o')
        # plt.scatter(x, y['total_heating_ec'])
        plt.xlabel(xlab)
        plt.ylabel('Heating Energy Consumption [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        plt.savefig(hp + 'hpmodel_heatingecVStime.png', dpi=600)
        
        plt.figure(5)
        plt.plot(x, y['total_cooling_ec'], '-o')
        # plt.scatter(x, y['total_cooling_ec'])
        plt.xlabel(xlab)
        plt.ylabel('Cooling Energy Consumption [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        plt.savefig(hp + 'hpmodel_coolingecVStime.png', dpi=600)
        
    elif xaxis == 'temp':
        os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/hp_proj1/researchletter_plots/DC_hp_energy_consumption_estimate_eps/' + hp + 'hp_model/VS_temp')
        x = temp
        xlab = 'Temperature [Celsius]'
        plt.figure(1)
        plt.scatter(x, y['total_ec'])
        plt.xlabel(xlab)
        # plt.ylabel('Heating + Cooling Energy Consumption [kWh]')
        plt.ylabel('Average Monthly Heat Pump Electricity [kWh]')
        ax = plt.gca()
        ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_totalecVStemp.png', dpi=600)
        plt.savefig(hp + 'hpmodel_totalecVStemp.eps')
        
        # plt.figure(2)
        # plt.scatter(x, y['total_elec_ec'])
        # plt.xlabel(xlab)
        # plt.ylabel('Electricity Energy Consumption [kWh]')
        # ax = plt.gca()
        # ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_elececVStemp.png', dpi=600)
        
        # plt.figure(3)
        # plt.scatter(x, y['total_ff_ec'])
        # plt.xlabel(xlab)
        # plt.ylabel('Fossil Fuel Energy Consumption [kWh]')
        # ax = plt.gca()
        # ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_ffecVStemp.png', dpi=600)
       
        # plt.figure(4)
        # plt.scatter(x, y['total_heating_ec'])
        # plt.xlabel(xlab)
        # plt.ylabel('Heating Energy Consumption [kWh]')
        # ax = plt.gca()
        # ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_heatingecVStemp.png', dpi=600)
       
        # plt.figure(5)
        # plt.scatter(x, y['total_cooling_ec'])
        # plt.xlabel(xlab)
        # plt.ylabel('Cooling Energy Consumption [kWh]')
        # ax = plt.gca()
        # ax.set_ylim([0, 5])
        # plt.savefig(hp + 'hpmodel_coolingecVStemp.png', dpi=600)

    
#%% 
hpec_IN(year='allyears', tc_data=ec_df_all, DC=True, hp='ave', timestep='month', xaxis='temp') 
hpec_IN(year='allyears', tc_data=ec_df_all, DC=True, hp='ave', timestep='month', xaxis='time') 

hpec_IN(year='allyears', tc_data=ec_df_all, DC=True, hp='test', timestep='month', xaxis='temp') 
hpec_IN(year='allyears', tc_data=ec_df_all, DC=True, hp='test', timestep='month', xaxis='time') 
# plot_all_ec_chron('allyears', 'time', 'month')

#%% Plot DC house data vs. model


#%% Calculate energy consumption for DC House
# def calc_DChouse_ec(temp_file):
#     ct_temps_data = np.loadtxt(temp_file, delimiter = ',') # DC House: 2020-2022?
#     ct_temps_data = ct_temps_data[:, ct_temps_data[0].argsort()]
#     # ct_names = ct_temps_data[0,:]
#     ct_temps_np = ct_temps_data[1:,:] - 273.15
#     cool_weight_elec_ct = np.loadtxt("cool_weight_elec_INct.csv", delimiter=",")
#     heat_weight_elec_ct = np.loadtxt("heat_weight_elec_INct.csv", delimiter=",")
#     heat_weight_ff_ct = np.loadtxt("heat_weight_ff_INct.csv", delimiter=",")
#     DC_tract = np.where(ct_temps_data[0,:] == 18157005400)
#     DC_temp = ct_temps_np[:, DC_tract[0]]
#     cool_weight_DC = cool_weight_elec_ct[DC_tract[0]]
#     heat_weight_elec_DC = heat_weight_elec_ct[DC_tract[0]]
#     heat_weight_ff_DC = heat_weight_ff_ct[DC_tract[0]]

#     ref_temp = 18.3 # Celsius
#     deltaT_heat = ref_temp - DC_temp # temp diff for heating mode
#     deltaT_heat[deltaT_heat < 0] = 0
#     T_heat = deltaT_heat.copy()
#     T_heat[T_heat > 0] = DC_temp[T_heat > 0]
    
#     deltaT_cool = DC_temp - ref_temp # temp diff for AC (cooling) mode
#     deltaT_cool[deltaT_cool < 0] = 0
#     T_cool = deltaT_cool.copy()
#     T_cool[T_cool > 0] = DC_temp[T_cool > 0]
    
#     dchouse_area = 208 # m2
#     ff_demand_heating = heat_weight_ff_DC * deltaT_heat * 10**6 * 0.000293071 * dchouse_area# also convert to kwh
#     elec_demand_cooling = cool_weight_DC * deltaT_cool * dchouse_area
#     elec_demand_heating = heat_weight_elec_DC * deltaT_heat * dchouse_area
    
#     total_elec_ec = elec_demand_cooling + elec_demand_heating 
#     total_ff_ec = ff_demand_heating
    
#     total_heating_ec = ff_demand_heating + elec_demand_heating
#     total_cooling_ec = elec_demand_cooling
    
#     total_ec = ff_demand_heating + elec_demand_cooling + elec_demand_heating
    
#     df = np.column_stack((total_ec, total_elec_ec, total_ff_ec, total_heating_ec, total_cooling_ec))
    
#     ec_df = pd.DataFrame(columns = ['total_ec','total_elec_ec', 
#                                     'total_ff_ec', 'total_heating_ec', 'total_cooling_ec'],
#                          data = df)
    
#     return ec_df

# temp_files = ['ct_temps_IN_2020.gz','ct_temps_IN_2021.gz','ct_temps_IN_2022.gz']

# # ec_df20 = pd.DataFrame(columns = ['total_ec','total_elec_ec', 
# #                                 'total_ff_ec', 'total_heating_ec', 'total_cooling_ec'])

# ec_df20 = calc_DChouse_ec(temp_files[0])
# ec_df21 = calc_DChouse_ec(temp_files[1])
# ec_df22 = calc_DChouse_ec(temp_files[2])
# ec_df_all = pd.concat([ec_df20, ec_df21, ec_df22], axis = 0)

# # ec_df['year'] = [2020, 2021, 2022]   
 
#%% Plotting
# plot all plots for 1 year vs. time or temp chronologically
def plot_all_ec_chron(year, xaxis, timestep, ec_data):
#     if year == 'allyears':
#         data = ec_data.iloc[0:25584]  
#         temp_files = ['ct_temps_IN_2020.gz','ct_temps_IN_2021.gz','ct_temps_IN_2022.gz']
#         DC_temps_all = []
#         for file in temp_files:
#             ct_temps_data = np.loadtxt(file, delimiter = ',') # DC House: 2020-2022?
#             ct_temps_data = ct_temps_data[:, ct_temps_data[0].argsort()]
#             ct_temps_np = ct_temps_data[1:,:] - 273.15
#             DC_tract = np.where(ct_temps_data[0,:] == 18157005400)
#             DC_temp = ct_temps_np[:, DC_tract[0]]
#             DC_temps_all.append(DC_temp)
#         temp_flat = list(chain(*DC_temps_all))[0:25584]
#         temp_flat2 = np.array(list(chain(*temp_flat)))
#         temp_flat3 = np.reshape(temp_flat2,(-1,1))
#         flatlist=[]
#         for sublist in DC_temps_all:
#             for element in sublist:
#                 flatlist.append(element)
#         time = np.arange(0,25584,1)
#         if timestep == 'month':
#             days_in_month = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
#                                       31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
#                                       31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]) # Set up number of days in each month
#             hours_in_month = days_in_month * 24
#             def monthly_ave(data, months):
#                 month_ave = np.zeros((months,data.shape[1]))
#                 start = 0
#                 for month, hours in enumerate(hours_in_month):
#                     step = start + hours
#                     entry = np.mean(data[start:step],axis=0)
#                     month_ave[month] = entry
#                     start = step
#                 return month_ave
#             y = monthly_ave(data, 35)
#             y = pd.DataFrame(columns = ['total_ec','total_elec_ec', 
#                                             'total_ff_ec', 'total_heating_ec', 'total_cooling_ec'],
#                                  data = y)
#             temp = monthly_ave(temp_flat3, 35)
#             time = np.arange(0,35,1)
#     else:
#         temp_file = 'ct_temps_IN_' + str(year) + '.gz'
#         ct_temps_data = np.loadtxt(temp_file, delimiter = ',') # DC House: 2020-2022?
#         ct_temps_data = ct_temps_data[:, ct_temps_data[0].argsort()]
#         ct_temps_np = ct_temps_data[1:,:] - 273.15
#         DC_tract = np.where(ct_temps_data[0,:] == 18157005400)
#         DC_temp = ct_temps_np[:, DC_tract[0]]
#         temp = DC_temp
#         if year == 2020:
#             y = ec_df20
#             time = np.arange(0,8784,1)
#         elif year == 2021:
#             y = ec_df21
#             time = np.arange(0,8760,1)
#         elif year == 2022:
#             y = ec_df22.iloc[0:8016]
#             # time = np.arange(0,8760,1)
#             time = np.arange(0,8016,1)
#             temp = DC_temp[0:8016]
        
        
#     if xaxis == 'time':
#         x = time
#         xlab = 'Time [hours]'
#     elif xaxis == 'time' & timestep == 'month':
#         xlab = 'Time [months]'
    
#     if xaxis == 'time':
#         plt.figure(1)
#         plt.plot(x, y['total_ec'])
#         plt.xlabel(xlab)
#         plt.ylabel('Heating + Cooling Energy Consumption [kWh]')
        
#         plt.figure(2)
#         plt.plot(x, y['total_elec_ec'])
#         plt.xlabel(xlab)
#         plt.ylabel('Electricity Energy Consumption [kWh]')
        
#         plt.figure(3)
#         plt.plot(x, y['total_ff_ec'])
#         plt.xlabel(xlab)
#         plt.ylabel('Fossil Fuel Energy Consumption [kWh]')
        
#         plt.figure(4)
#         plt.plot(x, y['total_heating_ec'])
#         plt.xlabel(xlab)
#         plt.ylabel('Heating Energy Consumption [kWh]')
        
#         plt.figure(5)
#         plt.plot(x, y['total_cooling_ec'])
#         plt.xlabel(xlab)
#         plt.ylabel('Cooling Energy Consumption [kWh]')
        
#     elif xaxis == 'temp':
#         x = temp
#         xlab = 'Temperature [Celsius]'
#         plt.figure(1)
#         plt.scatter(x, y['total_ec'])
#         plt.xlabel(xlab)
#         plt.ylabel('Heating + Cooling Energy Consumption [kWh]')
        
#         plt.figure(2)
#         plt.scatter(x, y['total_elec_ec'])
#         plt.xlabel(xlab)
#         plt.ylabel('Electricity Energy Consumption [kWh]')
        
#         plt.figure(3)
#         plt.scatter(x, y['total_ff_ec'])
#         plt.xlabel(xlab)
#         plt.ylabel('Fossil Fuel Energy Consumption [kWh]')
        
#         plt.figure(4)
#         plt.scatter(x, y['total_heating_ec'])
#         plt.xlabel(xlab)
#         plt.ylabel('Heating Energy Consumption [kWh]')
        
#         plt.figure(5)
#         plt.scatter(x, y['total_cooling_ec'])
#         plt.xlabel(xlab)
#         plt.ylabel('Cooling Energy Consumption [kWh]')
    
    

    
# plot_all_ec_chron('allyears', 'time', 'month')

#%%
    
# # plot all plots for 1 year vs. temp (linear fit)
# def plot_all_ec_VStemp(year):
    
#     if year == 'all':
#         y = ec_df_all  
#         temp_files = ['ct_temps_IN_2020.gz','ct_temps_IN_2021.gz','ct_temps_IN_2022.gz']
#         DC_temps_all = []
#         for file in temp_files:
#             ct_temps_data = np.loadtxt(file, delimiter = ',') # DC House: 2020-2022?
#             ct_temps_data = ct_temps_data[:, ct_temps_data[0].argsort()]
#             ct_temps_np = ct_temps_data[1:,:] - 273.15
#             DC_tract = np.where(ct_temps_data[0,:] == 18157005400)
#             DC_temp = ct_temps_np[:, DC_tract[0]]
#             DC_temps_all.append(DC_temp)
#         x = list(chain(*DC_temps_all))[0:25544]
        
#     else:
#         temp_file = 'ct_temps_IN_' + str(year) + '.gz'
#         ct_temps_data = np.loadtxt(temp_file, delimiter = ',') # DC House: 2020-2022?
#         ct_temps_data = ct_temps_data[:, ct_temps_data[0].argsort()]
#         ct_temps_np = ct_temps_data[1:,:] - 273.15
#         DC_tract = np.where(ct_temps_data[0,:] == 18157005400)
#         DC_temp = ct_temps_np[:, DC_tract[0]]
#         x = DC_temp
#         if year == 2020:
#             y = ec_df20
#         elif year == 2021:
#             y = ec_df21
#         elif year == 2022:
#             y = ec_df22[0:8000]
#             x = x[0:8000]
        
#     y['DC_temp'] = x
#     y.sort_values(by=['DC_temp'])

#     plt.figure(1)
#     plt.scatter(x, y['total_ec'])
#     plt.xlabel('Temperature [Celsius]')
#     plt.ylabel('Total Heating + Cooling Energy Consumption [kWh]')
    
#     plt.figure(2)
#     plt.plot(x, y['total_elec_ec'])
#     plt.xlabel('Temperature [Celsius]')
#     plt.ylabel('Electricity Energy Consumption [kWh]')
    
#     plt.figure(3)
#     plt.plot(x, y['total_ff_ec'])
#     plt.xlabel('Temperature [Celsius]')
#     plt.ylabel('Fossil Fuel Energy Consumption [kWh]')
    
#     plt.figure(4)
#     plt.plot(x, y['total_heating_ec'])
#     plt.xlabel('Temperature [Celsius]')
#     plt.ylabel('Heating Energy Consumption [kWh]')
    
#     plt.figure(5)
#     plt.plot(x, y['total_cooling_ec'])
#     plt.xlabel('Temperature [Celsius]')
#     plt.ylabel('Cooling Energy Consumption [kWh]')
    
# plot_all_ec_VStemp(2020)
    
    
# fig, ax = plt.subplots()
# ax.plot(x, y, linewidth=2.0)
# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))
# plt.show()  