#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:56:49 2025

Script to generate a table of GlobalSkyModel2016 Tsky values that can be read by PsrPopPy2

resolution of 1 degree, starting at 0, -89.5 and increasing in b then l to end at 359.5, 89.5

@author: jturner
@author_email: james.turner-2@manchester.ac.uk
"""

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pygdsm
from pygdsm import GlobalSkyModel2016

GSM2016 = pygdsm.GlobalSkyModel2016()
freq=2406.25 # MHz L-band=1284, S-band=2406.25

#l = np.arange(0,360,0.25) # 0 to 359.75 inclusive
#b = np.arange(-90, 90, 0.25) # -90 to 89.75 inclusive
""" Take measurement from centre of the PsrPopPy integer grid """
l = np.arange(2,360,4) # 0 to 358 inclusive
b = np.arange(-89.5, 90.5, 1) # -89.5 to 90 inclusive
print("resolution in l, b is {}, {} degrees".format(360.0/len(l), 180.0/len(b)))

with open("lib/fortran/lookuptables/GSM2016_{}MHz.ascii".format(freq), 'w') as file:
    n=1
    for i in range(0, len(l)):
        for j in range(0, len(b)):
            coords = SkyCoord(l[i], b[j], frame='galactic', unit=(u.deg, u.deg))
            tsky = GSM2016.get_sky_temperature(coords, freq)
            tsky_str = f'{tsky:.1f}'
            
            """ Must be written as 5 characters """
            if len(tsky_str)==3:
                file.write(f'  {tsky:.1f}')
            elif len(tsky_str)==4:
                file.write(f' {tsky:.1f}')
            else:
                file.write(f'{tsky:.1f}')
            if(n % 16 == 0):
                file.write("\n")
            n+=1