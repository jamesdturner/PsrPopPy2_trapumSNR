#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:45:09 2025

Make a population of pulsars using PsrPopPy2/evolve then PsrPopPy/dosurvey

----------------------------------------------------------------------------
Run multiple times with a bash script e.g. 4 times
#!/bin/sh
python make_population.py -o evolution_model_1 -N 1099 &
python make_population.py -o evolution_model_2 -N 1099 &
python make_population.py -o evolution_model_3 -N 1099 &
python make_population.py -o evolution_model_4 -N 1099 &
wait

Then launch the bash script from the command line
    time bash <script> 

view these jobs with 
    htop

kill these jobs with
    kill -9 PID
----------------------------------------------------------------------------

The following path(s) should be in your PYTHON PATH 
<path to PsrPopPy2>/PsrPopPy2/lib/python
otherwise the dependecies wont allow packages to be called

export PYTHONPATH="${PYTHONPATH}":<path to PsrPopPy2>/PsrPopPy2/lib/python

@author: jturner
@author_email: james.turner-2@manchester.ac.uk
"""
import dosurvey as dosurvey, populate, evolve
import cPickle
import os, sys
import argparse

# set defaults here
parser = argparse.ArgumentParser(
    description='This script makes a population using PsrPopPy2')

# number of pulsars to detect
parser.add_argument('-N', type=int, required=True,
                    help='number of pulsars to by detected by the survey(s)')
# maximum initial age of pulsars
parser.add_argument('-o', type=str, required=True,
                    help='name of output model file')
parser.add_argument('fpath', help='Model file(s)', type=os.path.realpath, nargs='*')
# =============================================================================
# pop = populate.generate(10, 
#                         surveyList=['PMSURV'], 
#                         electronModel='ymw16', 
#                         radialDistType='lfl06', 
#                         scindex=-4)
# =============================================================================

args = parser.parse_args()

if not os.path.isdir('sets'):
    os.system('mkdir sets')
    

# =============================================================================
# pop = evolve.generate(args.N, 
#                       surveyList=['PMSURV', 'HTRU_low', 'HTRU_mid', 'PALFAa', 'PALFAb', 'GBNCC', 'LOTAAS'],
#                       age_max=1.0E5,
#                       electronModel='ne2001',
# #                      pDistPars=[-1.04, 0.51],
# #                      pDistType='fk06'
#                       lumDistType='fk06',
# #                      lumDistPars=[-1.1, 0.9],
# #                      bFieldPars=[12.68, 0.48],
#                       scindex=-3.86,
# #                      spinModel=?,
# #                      beamModel=?,
# #                      widthModel=?,
# #                      duty=?
#                       birthVModel='gaussian',
#                       birthVPars=[0.0, 180.],
#                       alignModel='random',
#                       alignTime='None',
# #                      braking_index=?,#give value or give 0 to get random between 2-3
#                       zscale=0.05
#                       )
# #surveyPopulations = dosurvey.run(pop, ['PMSURV', 'HTRU_low', 'HTRU_mid', 'PALFAa', 'PALFAb', 'GBNCC', 'LOTAAS'], nostdout=True)
# #dosurvey.write(surveyPopulations, nores=True, summary=False, asc=False) # turn these on to see survey outputs, but these will get overwritten
# pop.write(outf='testsets/{}_{}.model'.format(args.N, args.o))
# =============================================================================

pop = evolve.generate(args.N,
                      surveyList=['PMSURV'],
#                      age_max=1.0E9,
                      electronModel='ne2001',
#                      pDistPars=[-1.04, 0.51],
#                      pDistType='fk06'
#                      lumDistType='fk06',
#                      lumDistPars=[-1.1, 0.9],
#                      bFieldPars=[12.68, 0.48],
                      scindex=-3.86,
#                      spinModel=?,
#                      beamModel=?,
#                      widthModel=?,
#                      duty=?
#                      birthVModel='gaussian',
#                      birthVPars=[0.0, 180.],
                      alignModel='random',
#                      alignTime='None',
#                      braking_index=?,#give value or give 0 to get random between 2-3
                      zscale=0.05,
                      nospiralarms=True
                      )


surveyPopulations = dosurvey.run(pop, ['PMSURV', 'HTRU_low', 'HTRU_mid'], nostdout=True)
dosurvey.write(surveyPopulations, nores=True, summary=True, asc=False) # turn these on to see survey outputs, but these will get overwritten
pop.write(outf='{}_{}.model'.format(args.N, args.o))



sys.exit(0)
