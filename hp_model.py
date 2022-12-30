TREF_F = 65 #The temperature where heat pumps switch from heating/cooling to cooling/heating
TREF_C = 18.333 #In Celsius
 
#First three functions take the temperature (if not specified otherwise )in fahrenhiet and the demand in whatever units (consumption is retuned in a scaled unit of that)

def consumption_R410a(temp, demand, unit='f'):

    if (unit == 'c'):
        if (temp >= 65):
            COP = -0.10036 * (temp) + 7.15168
   
        elif (temp < 65):
            COP = 0.063992 * (temp) + 3.29468
        EC = demand / COP
        return EC
    
    #Cooling
    if (temp >= 65):
        COP = -0.05575555555555556 * (temp - TREF_F) + 5.31175
    #Heating
    elif (temp < 65):
        COP = 0.03555111111111112 * (temp - TREF_F) + 4.4678666666666675
        
    #Energy demand scaled using COP to give consumption
    EC = demand / COP
    return EC

#Returns (cooling, heating) coeffs (slope, intercept at 65F) for each tech
#1 - R410a

def get_coeffs_f(choice): # fahrenheit
    if choice == 1:
        return [-0.05575555555555556,5.31175], [0.03555111111111112,4.4678666666666675]

def get_coeffs_c(choice): # celsius
    if choice == 1:
        return [-0.10036,7.15168], [0.063992,3.29468]
