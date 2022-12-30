#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:07:10 2022

@author: kbiscoch
"""

def tc_energy_IN_debug(year, DC):
## Convert business as usual (BAU) electricity demand to heat pump electricity demand
    #%% Load packages
    import pandas as pd
    import numpy as np
    
    #%% Initializing data
    # Data needed: MERRA temp, housing areas
    ct_temps_data = np.loadtxt('ct_temps_IN_'+str(year)+'.gz', delimiter = ',')
    ct_temps_data = ct_temps_data[:, ct_temps_data[0].argsort()]
    ct_temps_np = ct_temps_data[1:,:] - 273.15
    
    # weights are already multiplied by percentage of homes with elec heat, elec cool, and ff heat
    cool_weight_elec_ct = np.loadtxt("cool_weight_elec_INct.csv", delimiter=",")
    heat_weight_elec_ct = np.loadtxt("heat_weight_elec_INct.csv", delimiter=",")
    heat_weight_ff_ct = np.loadtxt("heat_weight_ff_INct.csv", delimiter=",")
    
    ref_temp = 18.3 # Celsius
    deltaT_heat = ref_temp - ct_temps_np # temp diff for heating mode
    deltaT_heat[deltaT_heat < 0] = 0
    T_heat = deltaT_heat.copy()
    T_heat[T_heat > 0] = ct_temps_np[T_heat > 0] # temperatures less than ref temp
    
    deltaT_cool = ct_temps_np - ref_temp # temp diff for AC (cooling) mode
    deltaT_cool[deltaT_cool < 0] = 0
    T_cool = deltaT_cool.copy()
    T_cool[T_cool > 0] = ct_temps_np[T_cool > 0] # temperatures greater than ref temp
    #%% Load BAU energy consumption for a particular year 
    # Put 2021 data through ff demand and elec demand models
    
    ff_demand_heating = heat_weight_ff_ct * deltaT_heat 
    elec_demand_cooling = cool_weight_elec_ct * deltaT_cool
    elec_demand_heating =  heat_weight_elec_ct * deltaT_heat
    
    #%% ff efficiency assumption
    ff_efficiency = 0.8 # assumes all ff heating equip has 80% eff
    
    #%% elec heating efficiency assumption (HPs, electric resistance heaters (erh))
    erh_efficiency = 1
    elec_heating_efficiency = erh_efficiency 
    
    #%% elec cooling efficiency assumption
    seer_wa = (13 * 12.75 + 10 * 4.3) / (12.75 + 4.3) # SEER weighted average by age of AC equipment
    seer_wa = seer_wa * 0.293071 # convert from btu/w to w/w
    
    elec_cooling_efficiency = seer_wa
    #%% Thermal comfort demands
    tc_ff_demand_heating = ff_demand_heating * ff_efficiency
    tc_ff_demand_heating_kwh = tc_ff_demand_heating * 10**6 * 0.000293071# also convert to kwh

    tc_elec_demand_heating_kwh = elec_demand_heating * elec_heating_efficiency

    tc_elec_demand_cooling_kwh = elec_demand_cooling * elec_cooling_efficiency
    
#%% Post processing
    tc_elec_demand_cooling_kwh[np.isnan(tc_elec_demand_cooling_kwh)] = 0

    # tc_elec_demand_cooling_kwh_hp[np.isnan(tc_elec_demand_cooling_kwh_hp)] = 0
    tc_elec_demand_heating_kwh[np.isnan(tc_elec_demand_heating_kwh)] = 0
    
    tc_elec_demand_kwh = tc_elec_demand_cooling_kwh + tc_elec_demand_heating_kwh
    tc_heating_demand_kwh = tc_elec_demand_heating_kwh + tc_ff_demand_heating_kwh
    tc_total_demand_kwh = tc_elec_demand_kwh + tc_ff_demand_heating_kwh
    
    if DC == True:
        DC_tract = np.where(ct_temps_data[0,:] == 18157005400)
        tc_total_demand_kwh = tc_total_demand_kwh[:,DC_tract[0]]
        tc_elec_demand_kwh = tc_elec_demand_kwh[:,DC_tract[0]]
        tc_ff_demand_heating_kwh = tc_ff_demand_heating_kwh[:,DC_tract[0]]
        tc_heating_demand_kwh = tc_heating_demand_kwh[:,DC_tract[0]]
        tc_elec_demand_cooling_kwh = tc_elec_demand_cooling_kwh[:,DC_tract[0]]
        # tc_elec_demand_cooling_kwh_seerwa = tc_elec_demand_cooling_kwh_seerwa[:,DC_tract[0]]
    
    df = np.column_stack((tc_total_demand_kwh, tc_elec_demand_kwh, tc_ff_demand_heating_kwh, tc_heating_demand_kwh, tc_elec_demand_cooling_kwh))
    
    ec_df = pd.DataFrame(columns = ['total_ec','total_elec_ec', 
                                    'total_ff_ec', 'total_heating_ec', 'total_cooling_ec'],
                         data = df)
    
    return ec_df

