#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 18:33:50 2025

A code to build and analyse a population of pulsars from PsrPop(Py)
Due to constraints in evolving very large populations of (non-beaming) pulsars, this script combines a random set from smaller populations

@author: jturner
@author_email: james.turner-2@manchester.ac.uk

Run this script in the psrpoppy conda env, as python packages versions wont clash with population generation
export PYTHONPATH="${PYTHONPATH}":<path to PsrPopPy2>/PsrPopPy2/lib/python

Instructions on how to use PsrPopPy at: http://samb8s.github.io/PsrPopPy/
PsrPopPy2 has extra commands: https://github.com/devanshkv/PsrPopPy2

generate a normalised population in psrpoppy2 conda env
e.g. using the make_population.py script in this fork

"""

import os, sys
try:
    import cPickle
except ImportError:
    print(ImportError)
    import pickle as cPickle
import galacticops as go

import argparse
import json
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import random
import random as rand
import scipy
from scipy import stats
from scipy.integrate import quad
from scipy.interpolate import interp1d # python2.7 compatible 1d inverse transform sampling
import astropy.units as u
import astropy.coordinates
from astropy.coordinates import SkyCoord, Galactocentric # astropy v2.0.7, docs available here: https://docs.astropy.org/en/older-docs-archive/v2.0.7/index.html
from matplotlib.ticker import AutoMinorLocator

pi = math.pi
Mo = 1.989e30 # solar mass, kg
pc = 3.086e16 # 1 parsec, m
mH = 1.67e-27 # mass of hydrogen, kg
yr = 3.15576e7 # 1 year, s 
xsun = 0 # kpc
ysun = 8.5 # kpc
zsun = 6 # pc
snr_threshold = 9.0 # Spectral S/N threshold of TRAPUM survey
Ntargs = 119 # sample size (number of pulsars/SNR pairs to 'search')

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'STIXGeneral' 
plt.rcParams['mathtext.fontset'] = 'stix'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a population(s) of pulsars through the TRAPUM targeted survey of SNRs')
    parser.add_argument('-n', help='Of the set you provide, how many should be combined', type=int, required=True)
    parser.add_argument('-N', help='How many times to run this script. A random combination of n files will be used each time', type=int, default=1)
    parser.add_argument('-i', help='Interactive mode, this will show and write plots of the first population you provide', action='store_true')
    parser.add_argument('fpath', help='Model file(s)', type=os.path.realpath, nargs='*')
    
    return parser.parse_args()

def maxwell_boltzmann_pdf(x, mu):
    
    """probability distribution function of maxwellian with mean=mu"""
    
    a = mu*math.sqrt(pi/8)
    pdf = math.sqrt(2/pi) * (x**2/a**3) * np.exp( -x**2 / (2*a**2) )
    #cdf = error_func(x/(math.sqrt(2)*a)) - math.sqrt(2/pi) * (x/a) * np.exp( -x**2 / (2*a**2) )
    return pdf


# function to assign a 3D velocity to each pulsar
def assign_vel(psr, inverse_cdf_approx, count):
    
    """Assign the pulsar a 3D velocity vector in km/s"""
    
    if count==1:
        vels = inverse_cdf_approx(random.random(1000))
        plt.hist(vels, bins=100, density=True)
        plt.plot(vels, maxwell_boltzmann_pdf(vels, 400), ls='')
        plt.ylabel('PDF')
        plt.show()
    # assign pulsar a 3D velocity magnitude
    V3 = inverse_cdf_approx(random.random())
    psr.V3 = float(V3)
    # assign random direction
    theta = 2*pi*random.random()
    phi = pi*random.random()
    # convert to vx, vy, vz
    ### GALACTOCENTRIC COORDS IN PSRPOPPY ARE: 
    ### +y from gal centre to sun, +x in right when looking top down, +z out of page
    psr.vx = V3*math.sin(theta)*math.cos(phi) # in km/s
    psr.vy = V3*math.sin(theta)*math.sin(phi)
    psr.vz = V3*math.cos(theta)
    #print(psr.vx, psr.vy, psr.vz)
    #print(math.sqrt(psr.vx**2+psr.vy**2+psr.vz**2), psr.V3)

def remnant_expansion(psr, count):
    
    """Calculate the size and position of the remnant"""
    
    #make array in units of seconds, with steps in years
    t = psr.age*yr #psr age in s
    #t = np.linspace(0, 1e5*86400*365.25, 1e5)
    
    # phase 1 free expansion until M_ej = M_swept
    M_ej = Mo #                                  ejecta mass, kg
    E_sn = 1e51 #                                SN energy, erg
    n0 = 1#5.4e-21/mH #                          ISM desnity, /cm3 from 0.08Mo/pc3 Chakrabati+2020
    v_exp = math.sqrt(2*E_sn*1e-7/M_ej)/1000 #   expansion velocity, km/s 
    R1 = 5.8 * (M_ej/(10*Mo))**(1.0/3) * (n0/0.5)**(-1.0/3) * pc / 1000# radius in km at end of phase 1 from eq3 in Bamba2022
    
    # phase 2 sedov-Taylor phase
    #R2 = 1e6*(n0*1e51/(E_sn*1.5e10))**3.0 * pc # radius in m at end of ST phase (T=10^6K, eq 7 in Bamba2022)
    
    def expansion_model(v, t, R1):
        if (v*t)<R1:
            return v * t
        else:
            return (5.0* n0**-0.2 * (E_sn/1.0e51)**0.2 * ((t)/(yr*1000))**0.4 * pc)/1000 - ((5*(R1/(v*yr*1000))**0.4)*pc/1000 - R1) # eq 5.12 in Vink2020 in km and a second term for contact discontinuity
    
    psr.remnant_r = expansion_model(v_exp, t, R1)*1000 # in m
    #assign galactocentric coords of remnant centre
    psr.remnant_gx = psr.galCoords[0]-(psr.vx*t/pc) # in kpc
    psr.remnant_gy = psr.galCoords[1]-(psr.vy*t/pc)
    psr.remnant_gz = psr.galCoords[2]-(psr.vz*t/pc)
    psr.remnant_r0 = math.sqrt(abs(psr.remnant_gx**2 + psr.remnant_gy**2))# in kpc
    #assign remnant distance, sun at (0, 8.2, 0.006)
    psr.remnant_dist = math.sqrt(abs((psr.remnant_gx-xsun)**2 + (psr.remnant_gy-ysun)**2 + (psr.remnant_gz-zsun/1000)**2))
    # assign boolean for if distance from pulsar to remnant is bigger than radius
    if psr.remnant_r/pc < psr.V3*t*1.0e3/pc:
        psr.left = True
    else:
        psr.left = False
    #print("Pulsar has moved {} pc".format(psr.V3*t*1.0e3/pc))
    
    # uncomment to show plot
    if count==1:
        expansion_model_vec = np.vectorize(expansion_model) # vectorize func
        t=np.linspace(0,1.0e5*yr,10000)
        r = expansion_model_vec(v_exp, t, R1) # calc expansion, r for each t
        plt.plot(t/yr,r*1000/pc); plt.xlabel('time, yr'); plt.ylabel('radius, pc'); plt.xlim(0,1.0e5); plt.ylim(0,40); 
        plt.show()

def in_or_out(psr):
    
    """Add a flag indiciation if pulsar appears in or out of remnant"""
    
    # add galactic coords to object
    psr.remnant_gl, psr.remnant_gb = go.xyz_to_lb((psr.remnant_gx, psr.remnant_gy, psr.remnant_gz))
    # convert radius of remnant to angle on the sky
    ang_rad = math.atan(psr.remnant_r/(psr.remnant_dist*pc*1000))
    
    #print("Separation vs Angular radius of remnant: (radians)")
    pulsar_gal = SkyCoord(psr.gl, psr.gb, unit=(u.deg, u.deg), frame='galactic')
    remnant_gal = SkyCoord(psr.remnant_gl, psr.remnant_gb, unit=(u.deg, u.deg), frame='galactic')
    #print("Distance to pulsar and remnant = {}, {} kpc".format(psr.dtrue, psr.remnant_dist))
    sep = pulsar_gal.separation(remnant_gal)
    # if angular separation between remnant centre and pulsar > radius angle, assign boolean True for outside
    if sep.radian < ang_rad:
        psr.inside = True
    else:
        psr.inside = False
    
    #print("PULSAR gl, gb = {}, {}".format(psr.gl, psr.gb))
    #print("REMNANT gl, gb = {}, {}".format(psr.remnant_gl, psr.remnant_gb))
 
    return

def readtskyfile():
    
    """Read in tsky.ascii into a list from which temps can be retrieved"""

    tskypath = 'GSM2016_1284MHz.ascii'
    tskylist = []
    with open(tskypath) as f:
        for line in f:
            str_idx = 0
            while str_idx < len(line):
                # each temperature occupies space of 5 chars
                temp_string = line[str_idx:str_idx+5]
                try:
                    tskylist.append(float(temp_string))
                except:
                    pass
                str_idx += 5

    return tskylist

def get_Tsky(psr, tskylist):
    
    """Get the sky temperature using GSM2016 in K (copied from survey.py)"""
    
    # ensure l is in range 0 -> 360
    b = psr.remnant_gb
    if psr.remnant_gl < 0.:
        l = 360 + psr.remnant_gl
    else:
        l = psr.remnant_gl

    # convert from l and b to list indices
    j = b + 90.5
    if j > 179:
        j = 179

    nl = l - 0.5
    if l < 0.5:
        nl = 359
    i = float(nl) / 4.

    return tskylist[180*int(i) + int(j)]



def get_Weff(psr, t_samp):
    
    """Calculate the effective width seen by TRAPUM in ms"""
    
    psr.t_dm = 8.3e6 * psr.dm * 856.0/4096 / math.pow(1284.0, 3.0)
    psr.t_scatt_1284 = go.scale_bhat(psr.t_scatter, 1284.0, psr.scindex) # scale to L-band
    
    psr.width_ms = psr.width_degree * psr.period / 360.0 
    
    return math.sqrt(psr.width_ms**2 + (1000*t_samp)**2 + psr.t_dm**2 + psr.t_scatt_1284**2)
    
def calc_snr(flux, beta, Tsys, gain, n_p, t_obs, bw, duty):

    """Calculate the S/N ratios for TRAPUM assuming radiometer equation (from psrpoppy radiometer.py)"""

    signal = 1000 * beta * Tsys * math.sqrt(duty / (1.0 - duty)) \
            / gain / math.sqrt(n_p * t_obs * bw) # mJy

    return flux / signal

def calc_sensitivity(beta, Tsys, gain, n_p, t_obs, bw, duty):
    
    """Calculate the TRAPUM's sensitivity to a pulsar assuming radiometer equation """
    
    flux = 1000 * beta * Tsys * math.sqrt(duty / (1.0 - duty)) \
            / gain / math.sqrt(n_p * t_obs * bw) # mJy
    
    return snr_threshold * flux # mJy

def trapum_snr(yng_pop):
    # to consider; gain of meerkat, spectral index, obs freq, smearing contributions, Tsky?
    # do the cuts in stages to track number of pulsars
    bw = 856.0e6 # this is in Hz, so radiometer will be in Jy unless explicity converted!
    Trec = 18 + 4.5 # rec+spill
    beta = 1.00 / 0.65 / 0.7 # digitisation, multibeam overlap, spectral
    for psr in yng_pop:
        psr.t_samp = random.choice([32, 64])*4096/bw # us, random 153 or 306 
        psr.t_obs = random.choice([1200.0, 1800.0, 2400.0], 1, p=[12./Ntargs, 9./Ntargs, 98./Ntargs])
        psr.gain = 2.8 * random.choice([45.0, 59.2])/64.0 # K/Jy,  mean core or mean full Ndish
        Tsys = Trec + psr.tsky # K
        psr.s_1284 = psr.s_1400() * (1284.0/1400.0)**psr.spindex # mJy, L-band freq conversion
        duty = get_Weff(psr, psr.t_samp) / psr.period
        psr.duty = duty # overwriting intrinsic with effective
        sensitivity_unsmeared = calc_sensitivity(beta, Tsys, psr.gain, 2, psr.t_obs, bw, psr.width_degree/360.0) # calc flux limit to non-smeared pulse
        psr.detectable_unsmeared = False # set to false
        if psr.beaming == False:
            psr.spectral_snr = -1 # beaming away
            psr.sensitivity = -1
        elif duty > 0.75:
            psr.spectral_snr = -2 # too smeared
            psr.sensitivity = -2
            if psr.s_1284 > sensitivity_unsmeared:
                psr.detectable_unsmeared = True # but bright enough to be detected without smearing!
        else:
            psr.spectral_snr = calc_snr(psr.s_1284, beta, Tsys, psr.gain, 2, psr.t_obs, bw, duty)
            psr.sensitivity = calc_sensitivity(beta, Tsys, psr.gain, 2, psr.t_obs, bw, psr.duty)
    return

def do_trapum_survey(sample_pop, yng_pop):
    
    """ Sample the population for targets in the TRAPUM survey """
    
    r_min =  0.75 # arcmin, min radius of smalled SNR that TRAPUM searched (G28.56+0.00, THOR)
    
    # apply selection criteria
    sample = [psr for psr in sample_pop if any([psr.remnant_r  * 180 * 60 / 1000 / pi / psr.remnant_dist / pc > r_min and psr.detected==False])] # converts radians to arcmin and kpc to m
    #print(len(sample))
    sample1 = [psr for psr in sample if psr.remnant_gl>=15.0]
    sample2 = [psr for psr in sample if psr.remnant_gl<15.0]
    # weight remnants from 15<l<30 (group 1) twice as probable as -80<l<15 (group 2)
    p1 = 1. / (len(sample1) + len(sample2)/2)
    p2 = 1. / (2*len(sample1) + len(sample2))
    
    weights = []
    for psr in sample:
        if psr.remnant_gl>=15.0:
            weights.append(p1)
        else:
            weights.append(p2)
    try:
        sample = random.choice(sample, Ntargs, p=weights, replace=False)
    except:
        print("Probabilities are {}, {}".format(p1, p2))
        print("Samples total {}, {}".format(len(sample1), len(sample2)))
        print("Warning! Weights do not sum to 1, adjusting...")
        print(float(sum(weights)))
        diff = float(sum(weights)) - 1.0
        print("difference from 1 = {}".format(diff))
        weights[0] =  weights[0] - diff
        print(float(sum(weights)))
        sample = random.choice(sample, Ntargs, p=weights, replace=False)
    for psr in yng_pop:
        if psr in sample:
            psr.trapum = True
    return

def make_plots(yng_pop):
    
    ### for thesis
    #PPdot
    fig, ax = plt.subplots(figsize=(8,6))
    edots=[np.log10(psr.edot()) for psr in yng_pop]
    cb = plt.cm.get_cmap('Oranges')
    theplot = ax.scatter([psr.period/1000 for psr in yng_pop], [np.log10(psr.pdot) for psr in yng_pop], marker='*', c=edots, cmap=cb)
    plt.colorbar(theplot, label='log$_{10}\dot{E}$, erg s$^{-1}$', ax=ax)
    #axs[0,1].set_yscale('log'); axs[0,1].set_xscale('log'); 
    #axs[0,1].set_ylim(1.0e-16, 1.0e-10); axs[0,1].set_xlim(0.001, math.log10(2))
    ax.set_xlabel('$P$, s'); ax.set_ylabel('log$_{10}\dot{P}$')
    ax.grid()
    ax.set_xscale('log')
    xlim = ax.get_xlim()
    ax.plot([xlim[0], xlim[1]], [np.log10((xlim[0])/(2*1.0e5*yr)), np.log10((xlim[1])/(2*1.0e5*yr))], label='$\\tau_{c}$ = 100 kyr')
    ax.legend()
    #fig.savefig('/home/mbcxajt2/Documents/Theses/JTurner/Plots/synthetic-PPdot.pdf', format="pdf", bbox_inches="tight")
    
    ##### SOME PLOTS ######
    fig, axs = plt.subplots(figsize=(18,18), nrows=2, ncols=2)
    # scattering vs dm
    axs[0,0].scatter([psr.dm for psr in yng_pop], [np.log10(psr.t_scatter) for psr in yng_pop], marker='.', color='grey', alpha=0.7, label='synthetic pulsars')
    axs[0,0].set_xscale('log'); #axs[0,0].set_yscale('log'); axs[0,0].set_ylim(1.0e-6, 1.0e8); axs[0,0].set_xlim(0.1, math.log10(2000))
    axs[0,0].set_xlabel('log(DM), pc/cm$^{3}$'); axs[0,0].set_ylabel('log($\\tau_{sc}$ (1400 MHz), ms')
    axs[0,0].grid()
    dm = range(10, 3000)
    ts = [(1400./1000)**-3.86 * 10**(-6.344 + 1.467*np.log10(d) + 0.509*(np.log10(d))**2) for d in dm]
    axs[0,0].plot(dm, np.log10(ts), color='k', ls='--', label='Lewandowski+2015')
    ts = [(1400./327)**-3.86 * 1000 * 3.6e-9 * d**2.2 * (1 + 1.94 + 0.001*d**2) for d in dm] # ms
    axs[0,0].plot(dm, np.log10(ts), color='k', ls=':', label='Krishnakumar+2015')
    axs[0,0].legend()
    # PPdot
    edots=[np.log10(psr.edot()) for psr in yng_pop]
    cb = plt.cm.get_cmap('Oranges')
    theplot = axs[0,1].scatter([np.log10(psr.period/1000) for psr in yng_pop], [np.log10(psr.pdot) for psr in yng_pop], marker='*', c=edots, cmap=cb)
    plt.colorbar(theplot, label='log$_{10}\dot{E}$, erg s$^{-1}$', ax=axs[0,1])
    #axs[0,1].set_yscale('log'); axs[0,1].set_xscale('log'); 
    #axs[0,1].set_ylim(1.0e-16, 1.0e-10); axs[0,1].set_xlim(0.001, math.log10(2))
    axs[0,1].set_xlabel('log$_{10}P$, s'); axs[0,1].set_ylabel('log$_{10}\dot{P}$')
    axs[0,1].grid()
    xlim = axs[0,1].get_xlim()
    axs[0,1].plot([xlim[0], xlim[1]], [np.log10(10**(xlim[0])/(2*1.0e5*yr)), np.log10(10**(xlim[1])/(2*1.0e5*yr))], label='$\\tau_{c}$ = 100 kyr')
    axs[0,1].legend()
    #Positions
    tsky=[psr.tsky for psr in yng_pop]
    cb = plt.cm.get_cmap('magma')
    theplot = axs[1,0].scatter([psr.gl for psr in yng_pop], [psr.gb for psr in yng_pop], marker='x', c=tsky, cmap=cb)
    plt.colorbar(theplot, label='Tsky, K', ax=axs[1,0])
    axs[1,0].set_title("Before sky cut"); axs[1,0].set_ylabel('$b$, degrees'); axs[1,0].set_xlabel('$l$, degrees')
    # Luminosity distribution
    axs[1,1].hist([np.log10(psr.lum_1400) for psr in yng_pop], bins=15, density=True, log=True, facecolor='grey', alpha=0.5, edgecolor='k')
    L = np.arange(-3.5, 3.5, 0.1)
    axs[1,1].plot(L, stats.norm.pdf(L, 0.5, 0.8), color='k', label='FK06')
    axs[1,1].set_xlabel('Luminosity (1400 MHz), mJy kpc$^{2}$'); axs[1,1].set_ylabel('Probability Density')
    axs[1,1].grid(); axs[1,1].legend()
            
    fig.tight_layout()
    fig.savefig('/home/mbcxajt2/jturner/Population/plots/fake-popualtion-example-plots.png')
    plt.show()

def main():
    
    args = parse_args()
    
    n=1
    sets = args.fpath
    
    for N in range(1,1+args.N):
        
        rand.shuffle(sets) # shuffle them
        random_paths = rand.sample(sets, args.n) # ensures no repetition

        # merge populations 
        with open(random_paths[0], 'rb') as file: # the one you'll add the others to
            path, filename = os.path.split(file.name)
            pop = cPickle.load(file)
            pop.population = [psr for psr in pop.population if psr.age<1.0e5]
            
        if n <= args.n:
            for fpath in random_paths[1:]:
                with open(fpath, 'rb') as file:
                    pop_add = cPickle.load(file)
                    pop_add.population = [psr for psr in pop_add.population if psr.age<1.0e5]
                #print("should be instance:{} and list:{}".format(type(pop), type(pop.population)))
                    pop.population = pop.population + pop_add.population
                    print("Size of population: {}".format(len([psr for psr in pop.population])))
        
        # now we have built a combined list of pulsars representing a full population
        
        Ntot = len([psr for psr in pop.population])
        Nbeam = len([psr for psr in pop.population if psr.beaming==True])
        print("Total number of pulsars: {}".format(Ntot))
        print("Pulsars beaming towards Earth: {} = {}%".format(Nbeam, 100.0*Nbeam/Ntot))
        
        # age cut using list comprehension (also keeps memory usage low)
        yng_pop = [psr for psr in pop.population if psr.age<1.0e5]
        Nyng_det = len([psr for psr in yng_pop if psr.detected==True])
        Nyng_beam = len([psr for psr in yng_pop if psr.beaming==True])
        print("Number of young pulsars: {}".format(len(yng_pop)))
        print("Fraction of young pulsars that are beaming towards Earth: {}%".format(100.0*Nyng_beam/len(yng_pop)))
        print("Number of young pulsars that have been detected by dosurvey: {}".format(Nyng_det))

        # Define velocity distribution
        # Hobbs 2005, maxwellian with mean of 400(40) derived from the 1D
        mu = 400.0
        v = np.linspace(0, 1500, 1e5) # 10,000 bins
        cdf_approx = np.cumsum(maxwell_boltzmann_pdf(v, mu)) # compute the CDF by just doing the sum
        cdf_approx -= cdf_approx.min() # subtract lowest val
        inverse_cdf_approx = interp1d(cdf_approx/cdf_approx.max(), v) # cant directly interpolate the cdf, do normalised cumsum of pdf and return scipy object
        
        if args.i:
            count=1# counter for showing plot
            plt.scatter([psr.galCoords[0] for psr in yng_pop], [psr.galCoords[1] for psr in yng_pop], marker='x')
            plt.ylabel('Galactic Y, kpc'); plt.xlabel('Galactic X, kpc'); plt.xlim(-20, 20); plt.ylim(-20, 20)
            plt.savefig('/home/mbcxajt2/jturner/Population/plots/galactocentric-positions.png'); plt.show()
        else:
            count=0
            
        tskylist = readtskyfile()
        for psr in yng_pop:
            
            psr.trapum = False # set initial boolean as not searched
            
            # give each pulsar a new 3D veloicty
            assign_vel(psr, inverse_cdf_approx, count)
            #print("Velocity: {} km/s".format(psr.V3))
            #print("Age: {} kyr".format(psr.age/1000))
                
            # give each pulsar a supernova remnant
            remnant_expansion(psr, count)
            #print("Remnant radius: {} pc".format(psr.remnant_r/pc))
            # still inside as seen from Earth? (psr.left can still be True if seen through the remnant)
            in_or_out(psr)
            #print('Pulsar has left the remnant: {}\nPulsar appears inside the remnant: {}'.format(psr.left, psr.inside))
            #print("{} <- was pulsar detected in dosurvey with S/N {}".format(psr.detected, psr.snr))
            #print("----------------------------------------")
            psr.tsky = get_Tsky(psr, tskylist)
            count=0
        
        if args.i:
            make_plots(yng_pop)
        
        # yng_pop is a list containing all young pulsars with a remnant
        # 2D sky cut for TRAPUM survey footprint
        print("Number of pulsars before sky cut: {}".format(len([psr for psr in yng_pop])))
        sample_pop=[psr for psr in yng_pop if any([abs(psr.remnant_gb) <= 2.0 and  psr.remnant_gl <= 30.0 and psr.remnant_gl >=-100.0 and psr.detected==False])]
        print("Number of pulsars after sky cut: {}".format(len([psr for psr in sample_pop])))
        plt.rcParams['figure.figsize'] = [10.325, 3.5]
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['font.size'] = 20
        fig, ax = plt.subplots(1,2,sharey=False)
        ax[0].scatter([psr.remnant_gl for psr in sample_pop], [psr.remnant_gb for psr in sample_pop], marker='x', alpha=0.5, color='k')
        ax[0].set_title("Galactic"); ax[0].set_ylabel('$b$, deg'); ax[0].set_xlabel('$l$, deg'); ax[0].set_xlim(30.0, -100.0)
        ax[1].scatter([psr.galCoords[0] for psr in sample_pop], [psr.galCoords[1] for psr in sample_pop], marker='x', alpha=0.5, color='k')
        ax[1].set_title("Galactocentric"); ax[1].set_ylabel('Y, kpc'); ax[1].set_xlabel('X, kpc')
        for ax in [ax[0], ax[1]]:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(axis='both', direction='in', which='both', width=2)
            ax.grid(color='grey')
        fig.savefig('/home/mbcxajt2/jturner/Population/plots/after-skycut.pdf', format='pdf', bbox_inches="tight"); 
        if args.i:
            plt.show()
        
        # recompute S/N for TRAPUM survey
        trapum_snr(yng_pop)
        print("Survey {} complete".format(N))
        # acquire sample of remnants for trapum survey
        do_trapum_survey(sample_pop, yng_pop)
        
        
        print("Number of young pulsars in the TRAPUM survey region: {}".format(len(sample_pop)))
        print("Number of these searched by TRAPUM (should be {}): {}".format(Ntargs, len([psr for psr in yng_pop if psr.trapum==True])))
        print("Number of detections by TRAPUM: {}".format(len([psr for psr in yng_pop if any([psr.trapum==True and psr.spectral_snr > snr_threshold and psr.inside==True])])))
        
        # write out to new dictionary
        output_file = os.path.join(path, 'test.json'.format(N))
        if os.path.exists(output_file):
            print('Warning: writing to file that already exists')
        with open(output_file, 'w') as file:
            survey_population = {}
            i=0
            for psr in yng_pop:
                survey_population['{}'.format(i)] = {
                    'period': float(psr.period), # ms
                    'pdot': float(psr.pdot), # log
                    'dm': float(psr.dm), # pc/cm3
                    'duty': float(psr.duty), # frac
                    'age': float(psr.age), # yr
                    'gal_position': [psr.galCoords[0], psr.galCoords[1], psr.galCoords[2]],
                    'velocity': [psr.vx, psr.vy, psr.vz], # km/s
                    'gal_skycoords': [psr.gl, psr.gb],
                    'dtrue': float(psr.dtrue), # kpc
                    'spectral_snr': float(psr.spectral_snr),
                    'sensitivity': psr.sensitivity, # mJy, TRAPUM
                    'detectable_unsmeared': psr.detectable_unsmeared, # if width intrinsic, detectable?
                    's_1284': float(psr.s_1284), # mJy
                    'lum_1400': float(psr.lum_1400), # mJy kpc2
                    's_1400': float(psr.s_1400()), # mJy
                    'spindex': float(psr.spindex), # -ve
                    'char_age': float(psr.period / 2 / (psr.pdot*yr) / 1000), # yr
                    'edot': float(psr.edot()), # erg/s
                    'width_ms': float(psr.width_ms), # ms, unsmeared
                    't_scatt_1284': float(psr.t_scatt_1284), # ms at L-band
                    't_dm': float(psr.t_dm), # ms
                    't_samp': float(psr.t_samp), # s, sampling time for the observation
                    'gain': float(psr.gain), # K/Jy, gain for the observation
                    't_obs': float(psr.t_obs), # s integration time of the observation
                    'tsky': float(psr.tsky), # K
                    'beaming': psr.beaming, # True if beaming towards Earth
                    'detected': psr.detected,# in the previous surveys
                    'outside_remnant': psr.left, # True if actually outside shell
                    'appears_inside': psr.inside, # True is appears inside shell
                    'remnant_r': float(psr.remnant_r/pc), # pc
                    'remnant_position': [psr.remnant_gx, psr.remnant_gy, psr.remnant_gz], # kpc, galactocentric
                    'in_trapum': psr.trapum # True if searched by TRAPUM
                    }
                i+=1
            json.dump(survey_population, file, indent=4, sort_keys=False)
            print("-----------------------------------------------------------")

            N+=1
    
    print("Finished")
    
main()
