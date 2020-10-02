



def humidity(wvin, Tamb_K, unitIn, unitOut, P_mb):
    '''
    def humidity(wvin, Tamb_K, unitIn, unitOut, P_mb):
    By Penny M. Rowe
    Converted from matlab code 2013/5/12
    '''
    
    import numpy as np

    
    def satpressure_ice(T_c):

        # By Penny M. Rowe
        # Converted from matlab code 2013/5/12

        # Notes from satpressure_ice.m
        #
        # function Psat_ice = satpressure(T_c)
        #
        # units:  Input temp in celcius, will convert to Kelvin, pressure in Pa
        #
        # Reference:
        #
        # Marti, James and Konrad Mauersberger, A Survey of New Measurements of Ice Vapor Pressure 
        # at Temperatures Between 170 and 250K, Geophys. Res. Let., vol. 20, No. 5, pp. 363-366, 1993.
        #
        # Notes:  The frost point is the point to which you must cool air at constant pressure in
        # 	order for frost to form.  Thus at the frost point, Pambient = Psat over ice, and
        #	Psat(T) as shown below.
        #
        #
        # Penny Rowe
        #
        # Dec. 30, 1999

        # constants
        A = -2663.5     # +-.8, Kelvin
        B = 12.537      # +- 0.11

        T_K      = T_c+273.15     # convert celcius to kelvin
        logp     = A/T_K + B
        Psat_ice = 10**(logp)    # Pa

        return Psat_ice
    
    

    # ERROR ANALYSIS
    #
    # dP/dT = A*P/(T^2);
    # 
    # sigmaP = (A*P/(T^2)) * sigmaT
    #
    # QC: check dimensions
    if np.shape(wvin) != np.shape(Tamb_K):
        print('Error: wvin and Tamb_K have different sizes.')
        return
         
    if isinstance(P_mb,float):
        Pflag = 0
        # Indicates P_mb is a 1x1 float
    elif np.shape(wvin) != np.shape(P_mb):
        print('Error: wvin and P_mb have different sizes.')
        return
    else:
        Pflag = 1
        # Indicates P_mb is an array
    

    
    # INPUT UNITS
 
    if unitIn == 'Pa':
        Ph2o = wvin

    elif unitIn=='rhw' or unitIn=='rh' or unitIn=='rh_w':
        # input is rh_w, calculate Ph2o using Wexler
        rh_w = wvin
        c0 = -2.9912729e3;   c1 = -6.0170128e3;   c2 = 1.887643854e1
        c3 = -2.8354721e-2;  c4 = 1.7838301e-5;   c5 = -8.4150417e-10
        c6 = 4.4412543e-13;   D = 2.858487
        term = c0*Tamb_K**(-2) + c1*Tamb_K**(-1) + c2*Tamb_K**0 + \
          c3*Tamb_K**1 + c4*Tamb_K**2+ c5*Tamb_K**3 + c6*Tamb_K**4
          
        Psat_w = np.exp(term + D*np.log(Tamb_K))   	#Pa
        Ph2o   = 0.01 * rh_w * Psat_w 
    
    # input is dewpoint temperature in deg C
    elif unitIn=='dewpoint_Wexler' or unitIn=='dewpoint' or unitIn=='dewpt':
        Td_K = wvin + 273.15 
        T = Td_K 
  
        c0 = -2.9912729e3;   c1 = -6.0170128e3;   c2 = 1.887643854e1
        c3 = -2.8354721e-2;   c4 = 1.7838301e-5;   c5 = -8.4150417e-10
        c6 = 4.4412543e-13;   D = 2.858487
        term = c0*T**(-2) + c1*T**(-1) + c2*T**0 + \
          c3*T**1 + c4*T**2 + c5*T**3 + c6*T**4
  
        PwvWex = np.exp(term + D*np.log(T))       # Pa
        Ph2o = PwvWex 
  
        # Get Psat_w too, starting from Ph2o
        term = c0*Tamb_K**(-2) + c1*Tamb_K**(-1) + c2*Tamb_K**0 + \
          c3*Tamb_K**1 + c4*Tamb_K**2 + c5*Tamb_K**3 + c6*Tamb_K**4
  
        Psat_w = np.exp(term + D*np.log(Tamb_K))  # Pa
  
    #elif unitIn=='dewpoint_Goff_Gratch':
    elif (unitIn == 'ppmv') or (unitIn == 'ppm'):
        ppp  = 1e-6 * wvin
        Ph2o = ppp * P_mb             # water vapor amount in mb
        Ph2o = ppp * (P_mb - Ph2o)    # was ppmv with respect to DRY air
        Ph2o = ppp * (P_mb - Ph2o)    # Repeat
        Ph2o = ppp * (P_mb - Ph2o)    # Repeat - this should be acceptable
        Ph2o = Ph2o * 100             # convert from mb to Pa

    else:
        print("Input units not available: modify code to implement")
        return -999
        




    # OUTPUT UNITS

    # ouput is partial pressure of H2O in pascal
    if unitOut=='Pa' or unitOut=='pascal' or unitOut=='Pascal':
        return Ph2o
    
    # output is relative humidity with respect to ice
    elif unitOut == 'rhi' or unitOut == 'rh_i':
  
        if 'Psat_i' in locals():
            print("warning: I am not sure which Psat_i I should use")
        elif 'Psat_w' in locals():
            print("warning: I have Psat_w, will calculate Psat_i using satpressure_ice.")
        
        Tamb_C = Tamb_K - 273.15
        satpressure_ice.satpressure_ice(Tamb_C) 
        rh_i = 100*Ph2o / satpressure_ice.satpressure_ice(Tamb_C) 
        
        return rh_i 
        
    # output is rh_w (i.e. undo previous calc.)
    elif (unitOut == 'rhw') or (unitOut == 'rh'):
  
        if (unitIn == 'ppmv') or (unitIn == 'ppm'): #('Psat_w' not in locals):
            c0 = -2.9912729e3;   c1 = -6.0170128e3;   c2 = 1.887643854e1
            c3 = -2.8354721e-2;   c4 = 1.7838301e-5;   c5 = -8.4150417e-10
            c6 = 4.4412543e-13;   D = 2.858487
            term = c0*Tamb_K**(-2) + c1*Tamb_K**(-1) + c2*Tamb_K**0 + \
              c3*Tamb_K**1 + c4*Tamb_K**2 + c5*Tamb_K**3 + c6*Tamb_K**4
    
            Psat_w = np.exp(term + D*np.log(Tamb_K))  # Pa
  
  
        rh_w = 100 * Ph2o / Psat_w 
  
        return rh_w
 
    # output is Psat_wv
    elif unitOut=='Psat_wv':  

        if 'Psat_w' in locals()==0:
            print("Warning: Creating Psat_w, see code.")
    
            c0 = -2.9912729e3;   c1 = -6.0170128e3;   c2 = 1.887643854e1
            c3 = -2.8354721e-2;   c4 = 1.7838301e-5;   c5 = -8.4150417e-10
            c6 = 4.4412543e-13;   D = 2.858487
            term = c0*Tamb_K**(-2) + c1*Tamb_K**(-1) + c2*Tamb_K**0 + \
               c3*Tamb_K**1 + c4*Tamb_K**2 + c5*Tamb_K**3 + c6*Tamb_K**4
             
            Psat_w = np.exp(term + D*np.log(Tamb_K))   # Pa
           
        return Psat_w                              # Pa
  
    elif unitOut=='ppmv':
  
        wvout = 1e6*Ph2o/(100*P_mb-Ph2o) 
        return wvout
  
    # output is Precipitable Water Vapor in m
    elif unitOut=='pwv':  
        # From AMS glossary
        # PWV = (1/g) integral_P1_P2[x dp]
        # where x(p) = mixing ratio, .622e / (p-e)
        # where p=pressure, e=vapor pressure
  
        MW_H2O   = 18.01528               # g/mol  
        MWdryAir = 28.9644                # g/mol 
        rho_h2o  = 999.8395               # kg/m3 density of water at STP
        g        = 9.80665                # m/s2
   
        # input in pascals
        ppp = Ph2o / (P_mb*100 - Ph2o )   # parts water per part dry air
        gpg = ppp * MW_H2O / MWdryAir     # grams per gram
  
        x     = gpg/ rho_h2o           # g/g m3 kg-1
        wvout = -np.trapz(P_mb*100,x); # m3 kg-1 Pa = m3 kg-1 kg m-1 s-2 = m2 s-2
        wvout = wvout/g                # m2 s-2 * m-1 s2 = m

        return wvout
           
    # output is mixing ratio as g/kg: g water / kg air
    elif unitOut=="mixingratio" or unitOut=="g_per_kg":  
  
        MW_H2O   = 18.01528               # g/mol  
        MWdryAir = 28.9644                # g/mol 
          
        ppp = Ph2o / (P_mb*100 - Ph2o )   # parts per part
        gpg = ppp*MW_H2O / MWdryAir       # grams per gram
        wvout = gpg *1000                 # g/g g/kg = g/kg
           
        return wvout
  
    # output is molecules / cm3
    elif unitOut=="molec_cm3":  
 
        k     = 1.3806504e-23             # W K-1, Boltzman's constant 
        wvout = Ph2o/(k*Tamb_K) /100^3    # molec / m3 * (1 m / 100 cm)^3
                                          # molecules / cm3
        return wvout
    
    else:
        print("Error: I did not understand your output units.")
        return
    
    
