# Created by Mohammad Rezqalla

import matplotlib.pyplot as plt
import numpy as np

T_ref = 18.333
x = np.linspace(T_ref,(120-32)*5/9)
x_heating = np.linspace((-20-32)*5/9,T_ref)

# these are with only temp (and not deltaT) as the dependent variable
coeff_c_R410a_F = [-0.0335,-0.0372,-0.0343,-0.0559,-0.0631,-0.0531,-0.0610,-0.0528,-0.0528,-0.0451,-0.0567,-0.0683,-0.0639,-0.0639,-0.0667,-0.0729,-0.0641,-0.0583]
coeff_c_R410a_F  = np.array(coeff_c_R410a_F)
coeff_c_R410a_C = coeff_c_R410a_F * (9/5)
coeff_c_R410a_C = coeff_c_R410a_C.tolist()
names_c_R410a = ["D1","D2","D3","D4","D2_1","D2_2","D2_3","D2_4","D2_5","HL","D3_1","D3_2","D4_1","D4_2","D4_3","D5_1","D5_2","D5_3"]
intrcpts_c_R410a_F = [6.638,7.446,6.886,8.532,10.298,8.661,9.865,8.598,8.538,7.313,8.558,9.753,9.662,9.667,10.083,11.2865,9.9430,9.118]
intrcpts_c_R410a_F  = np.array(intrcpts_c_R410a_F)
intrcpts_c_R410a_C = intrcpts_c_R410a_F + (160/5) * coeff_c_R410a_F
intrcpts_c_R410a_C = intrcpts_c_R410a_C.tolist()
coeff_h_R410a_F = [0.0378, 0.0513, 0.0418,0.0397, 0.0354,0.0348,0.0323, 0.0295,0.0273,0.0427,0.0387,0.0343,0.0228,0.03442,0.0270,0.0396,0.0355,0.0350]
coeff_h_R410a_F  = np.array(coeff_h_R410a_F)
coeff_h_R410a_C = coeff_h_R410a_F * (9/5)
coeff_h_R410a_C = coeff_h_R410a_C.tolist()
names_h_R410a = ["D1_h","D2_h","D3_h","D4_h","D2_1_h","D2_2_h","D2_3_h","D2_4_h","D2_5_h","HL","D3_1_h","D3_2_h","D4_1_h","D4_2_h","D4_3_h","D5_1_h","D5_2_h","D5_3_h"]
intrcpts_h_R410a_F = [1.886,1.526,1.703,1.725,2.953,2.879,2.638,2.445,2.214,1.0896,2.2404,2.0080,1.7482,1.9978,1.9857,2.7962,2.5103,2.4816]
intrcpts_h_R410a_F  = np.array(intrcpts_h_R410a_F)
intrcpts_h_R410a_C = intrcpts_h_R410a_F + (160/5) * coeff_h_R410a_F
intrcpts_h_R410a_C = intrcpts_h_R410a_C.tolist()

colors = ["","","","","","","","","","","","","","","","","","",]

plt.figure(1)
for (c,i,name) in zip(coeff_c_R410a_C, intrcpts_c_R410a_C,names_c_R410a):
    k = x * c + i
    plt.plot(x,k)
plt.gca().set_prop_cycle(None)
for (c,i,name) in zip(coeff_h_R410a_C, intrcpts_h_R410a_C,names_h_R410a):
    k = x_heating * c + i
    plt.plot(x_heating,k)
    
plt.title("R410A Heat Pumps")
plt.xlabel("Temperature (Celsius)")
plt.ylabel("COP")
plt.grid(alpha=0.15)
plt.savefig('R410a_hpmodels_plot.eps')
plt.savefig('R410a_hpmodels_plot.png', dpi=600)

plt.figure(2)
for (c,i,name) in zip(coeff_c_R410a_C, intrcpts_c_R410a_C, names_c_R410a):
    k = x * c + i
    if(name != "HL"):
        plt.plot(x,k,alpha=0.2)
    else:
        plt.plot(x,k,linewidth=3.0,linestyle='dashed')
plt.gca().set_prop_cycle(None)
for (c,i,name) in zip(coeff_h_R410a_C, intrcpts_h_R410a_C, names_h_R410a):
    k = x_heating * c + i
    if (name != "HL"):
        plt.plot(x_heating,k,alpha=0.2)
    else:
        plt.plot(x_heating,k,linewidth=3.0,linestyle='dashed')
    
plt.title("R410A Heat Pumps")
plt.xlabel("Temperature (Celsius)")
plt.ylabel("COP")
plt.grid(alpha=0.15)
