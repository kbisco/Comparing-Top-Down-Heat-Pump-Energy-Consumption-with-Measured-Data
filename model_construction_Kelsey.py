#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:33:25 2023

@author: kbiscoch
"""

import os
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression

os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/HP_Modelling/Data Scraping')
col_names = ['temp', 'COP']

ATMO_cool = np.loadtxt('ATMOcool_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5))
DF_cool = np.loadtxt('DFcool_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5,6))

ATMO_heat = np.loadtxt('ATMOheat_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5))
DF_heat = np.loadtxt('DFheat_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5,6))

temp1_cool = np.tile(ATMO_cool[:,0],reps=4)
cops1 = np.concatenate((ATMO_cool[:,1],ATMO_cool[:,2],ATMO_cool[:,3],ATMO_cool[:,4]), axis=0)
ATMO_cool_data = np.column_stack((temp1_cool,cops1))

temp2_cool = np.tile(DF_cool[:,0],reps=5)
cops2 = np.concatenate((DF_cool[:,1],DF_cool[:,2],DF_cool[:,3],DF_cool[:,4],DF_cool[:,5]), axis=0)
DF_cool_data = np.column_stack((temp2_cool,cops2))

temp1_heat = np.tile(ATMO_heat[:,0],reps=4)
cops1 = np.concatenate((ATMO_heat[:,1],ATMO_heat[:,2],ATMO_heat[:,3],ATMO_heat[:,4]), axis=0)
ATMO_heat_data = np.column_stack((temp1_heat,cops1))

temp2_heat = np.tile(DF_heat[:,0],reps=5)
cops2 = np.concatenate((DF_heat[:,1],DF_heat[:,2],DF_heat[:,3],DF_heat[:,4],DF_heat[:,5]), axis=0)
DF_heat_data = np.column_stack((temp2_heat,cops2))

def runRegModels(inData, fData, cData, c_x, f_x):
    for n in range(inData.shape[1]-1):
        lmF = LinearRegression().fit(inData[:,0].reshape(-1,1), inData[:,n+1].reshape(-1,1))
        fData = np.append(fData, [np.array([lmF.intercept_[0], lmF.coef_[0,0]])], axis=0)
        lmFpred = inData[:,0]*lmF.coef_ + lmF.intercept_
        lmC = LinearRegression().fit(((inData[:,0]-32)*5/9).reshape(-1,1), inData[:,n+1].reshape(-1,1))
        cData = np.append(cData, [np.array([lmC.intercept_[0], lmC.coef_[0,0]])], axis=0)
    return (fData, cData)

fDatacool_ATMO = np.zeros([1,2])
cDatacool_ATMO = np.zeros((1,2))
fDatacool_DF = np.zeros([1,2])
cDatacool_DF = np.zeros((1,2))
c_x = np.arange(-10, 18.3, 10)
f_x = np.arange(-10*9/5+32,65,10)
fDatacool_ATMO, cDatacool_ATMO = runRegModels(ATMO_cool, fDatacool_ATMO, cDatacool_ATMO, c_x, f_x)
fDatacool_DF, cDatacool_DF = runRegModels(DF_cool, fDatacool_DF, cDatacool_DF, c_x, f_x)
fDatacool_ATMO = np.delete(fDatacool_ATMO,0,0)
cDatacool_ATMO = np.delete(cDatacool_ATMO,0,0)
fDatacool_DF = np.delete(fDatacool_DF,0,0)
cDatacool_DF = np.delete(cDatacool_DF,0,0)
print(fDatacool_ATMO)
print(cDatacool_ATMO)
print(np.average(fDatacool_ATMO, axis=0))
print(np.average(cDatacool_ATMO, axis=0))
print(fDatacool_DF)
print(cDatacool_DF)
print(np.average(fDatacool_DF, axis=0))
print(np.average(cDatacool_DF, axis=0))

fDataheat_ATMO = np.zeros([1,2])
cDataheat_ATMO = np.zeros((1,2))
fDataheat_DF = np.zeros([1,2])
cDataheat_DF = np.zeros((1,2))
c_x_h = np.arange(18.3, 38, 10)
f_x_h = np.arange(65, 100, 10)
fDataheat_ATMO, cDataheat_ATMO = runRegModels(ATMO_heat, fDataheat_ATMO, cDataheat_ATMO, c_x, f_x)
fDataheat_DF, cDataheat_DF = runRegModels(DF_heat, fDataheat_DF, cDataheat_DF, c_x, f_x)
fDataheat_ATMO = np.delete(fDataheat_ATMO,0,0)
cDataheat_ATMO = np.delete(cDataheat_ATMO,0,0)
fDataheat_DF = np.delete(fDataheat_DF,0,0)
cDataheat_DF = np.delete(cDataheat_DF,0,0)
print(fDataheat_ATMO)
print(cDataheat_ATMO)
print(np.average(fDataheat_ATMO, axis=0))
print(np.average(cDataheat_ATMO, axis=0))
print(fDataheat_DF)
print(cDataheat_DF)
print(np.average(fDataheat_DF, axis=0))
print(np.average(cDataheat_DF, axis=0))
