# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 10:18:54 2021

@author: Cosima
"""

from collections import namedtuple
import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import leastsq
import yaml
#%%
#############################
#         SCHALTER          #
#############################

use_maom = True

unit_factor = 1e6/12

use_optimizer = True
predict_beyond_measurements = True


#%%
def initial_state(initial_C, initialFractionCbio, solubleC):
    
        C_bio  = initial_C * initialFractionCbio  
        C_pom  = ((initial_C - (initial_C * initialFractionCbio)) * (1-solubleC))
        C_doc  = ((initial_C - (initial_C * initialFractionCbio)) * solubleC)* 0.3    # 0.3 
        C_maom = ((initial_C - (initial_C * initialFractionCbio)) * solubleC)* 0.7    # 0.7
        C_atm = 0
        
        return np.array([C_bio, C_pom, C_doc, C_maom, C_atm]) 
#%%

def commission(maxSpecAssimilRate, maxSpecDepolymRate, halfSaturationConstantAssimil, halfSaturationConstantDepolym, mortalityRate, unsolubleFractionOfDeadMicrobes,
               CUE, adsorptionRate, desorptionRate, maxSorptionCapacity):#, initial_C, initialFractionCbio, solubleC): #eigentlich ohne die letzten 3

    def tendencies(t, state):
    
        C_pom, C_doc, C_bio, C_maom, C_atm = state    
    
        depolymerization = maxSpecDepolymRate * C_pom * C_bio / (C_bio + halfSaturationConstantDepolym)
        assimilation     = maxSpecAssimilRate * C_bio * C_doc / (C_doc + halfSaturationConstantAssimil)
    
        if use_maom:
            adsorption       = adsorptionRate * C_doc * (maxSorptionCapacity - C_maom)
            desorption       = desorptionRate * C_maom
            
        else:
            desorption = 0
            adsorption = 0
    
        deltaC_pom       = unsolubleFractionOfDeadMicrobes * mortalityRate * C_bio - depolymerization
        deltaC_doc       = (1-unsolubleFractionOfDeadMicrobes) * mortalityRate * C_bio + \
            depolymerization - assimilation - adsorption + desorption
        deltaC_bio       = assimilation * CUE - mortalityRate * C_bio
        deltaC_maom      = adsorption - desorption
        deltaC_atm       = assimilation * (1-CUE)
        
        return np.array([deltaC_pom, deltaC_doc, deltaC_bio, deltaC_maom, deltaC_atm])
    return tendencies


#%%
def _main():
    with open("data/09-1351_2.yaml") as infile:
    #with open("data/09-1352_2.yaml") as infile:
    #with open("data/09-1353_2.yaml") as infile:
    #with open("data/09-1367_2.yaml") as infile:
    #with open("data/09-1369_2.yaml") as infile:
    #with open("data/09-1370_2.yaml") as infile:
        data = yaml.load(infile)
        
    def predict_system(days, maxSpecAssimilRate, maxSpecDepolymRate, halfSaturationConstantAssimil, halfSaturationConstantDepolym, mortalityRate, unsolubleFractionOfDeadMicrobes,
           CUE, adsorptionRate, desorptionRate, maxSorptionCapacity, initialFractionCbio, solubleC):
        state = initial_state(data["c_initial"], initialFractionCbio, solubleC)   
        model = commission(maxSpecAssimilRate, maxSpecDepolymRate, halfSaturationConstantAssimil, halfSaturationConstantDepolym, mortalityRate, unsolubleFractionOfDeadMicrobes,
               CUE, adsorptionRate, desorptionRate, maxSorptionCapacity)#,initial_C, initialFractionCbio, solubleC)
    
        result = solve_ivp(model, (0, max(days)), state, t_eval=days) #initial value problem
        
        system_state_for_days = result.y
        system_state_for_days[4] *= unit_factor # ist das gleiche wie: system_state_for_days[4] = system_state_for_days[4]*unit_factor
        
        return result.y
    
    # lb =    np.array([ 1e-3,     1e-15,   1e-5,  1e-5,  1e-12,  1e-12,     0.2,   1e-6,    1e-15,1e-12,  1e-10, 0.000001])
    # ub =    np.array([  0.1,      1e-5,     10,    10,      1,    0.8,     0.8,      1,     1e-8,    1,    0.8, 0.5])
    # guess = np.array([ 0.01,      1e-9,      1,  0.01, 0.0001,   1e-8,     0.5,   1e-3,    1e-10,  0.1,   0.05, 0.01])
    lb =     [ 1e-12,    1e-15,  1e-10, 1e-10,  1e-12,  1e-12,     0.2,  0.001,  1e-10,      1e-6,    1e-15,1e-12]
    ub =     [  0.1,      1e-5,     10,    10,      1,    0.8,     0.8,    0.1,    0.8,         1,     1e-8,    1]
    #guess =  [ 0.01,      1e-9,      1,  0.01, 0.0001,   1e-8,     0.3,   0.04,   0.05,      1e-3,    1e-10,  0.1]
    guess= [0.09999,4.99e-08,0.2596,7.273,1.0e-12,0.5025236, 0.4247,0.01266323, 5.27e-04,0.503,1.874e-09,0.018694]
    
    
    days = np.array(data["day"])
    co2_cum = np.array(data["co2_cum"])
    
    if use_optimizer:
    
        def atm_for_fit(*args):
            complete_state = predict_system(*args)
            return complete_state[4] # hole atm werte aus dem gesamten verlauf aller pools
        
        popt, pcov = curve_fit(atm_for_fit,  # war vorher nu f , (heißt jetzt predict_system)
                              days,
                              co2_cum,
                              p0=guess,
                              bounds=(lb, ub))
   
    #popt=leastsq(f, x0=guess, args=(days, co2_cum)) 
    #popt = least_squares(f, co2_cum, days, initial_state, jac='2-point', bounds=(lb, ub))
        print(guess)
        print(popt)
    
        if predict_beyond_measurements:
            prediction_days = range(0,5000)
            
        else:
            prediction_days = days
    
        co2_opt = predict_system(prediction_days, *popt)
        
        residual = np.sum((co2_opt[4] - co2_cum)**2)
       # print(residual)
       
        plt.plot(prediction_days, co2_opt[2],'-r', label = 'biomass')
        plt.title('bio')
       
        plt.figure()
        plt.plot(prediction_days, co2_opt[4], "-", label="optimized")
    
    else:
        co2_guess = predict_system(days, *guess)
        plt.plot(days, co2_guess, "-x", label="guess")
        
        
    plt.plot(days, co2_cum, "o", label="measured")
    plt.title(data["name"])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _main()
    

#%%
