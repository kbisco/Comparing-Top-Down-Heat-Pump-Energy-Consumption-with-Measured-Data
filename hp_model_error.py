#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:58:12 2023

@author: kbiscoch
"""

import numpy as np
D1_cool = np.loadtxt('D1cool_COP_temp.csv', delimiter=',')
D2_cool = np.loadtxt('D2cool_COP_temp.csv', delimiter=',')
D3_cool = np.loadtxt('D3cool_COP_temp.csv', delimiter=',')
D4_cool = np.loadtxt('D4cool_COP_temp.csv', delimiter=',')
D5_cool = np.loadtxt('D5cool_COP_temp.csv', delimiter=',')
HL_cool = np.loadtxt('', delimter=',')

D1_heat = np.loadtxt('D1heat_COP_temp.csv', delimiter=',')
D2_heat = np.loadtxt('D2heat_COP_temp.csv', delimiter=',')
D3_heat = np.loadtxt('D3heat_COP_temp.csv', delimiter=',')
D4_heat = np.loadtxt('D4heat_COP_temp.csv', delimiter=',')
D5_heat = np.loadtxt('D5heat_COP_temp.csv', delimiter=',')
HL_heat = np.loadtxt('', delimiter=',')