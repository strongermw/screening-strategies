# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:13:04 2022
Simulate the transmission of Omicrion with R0=10 for scenarios.
@author: MathGIS_KLC
"""



#%% Importing the model code
from models import *
from networks import *
from sim_loops import *
from utilities import *

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame





#%%  Generate a community network

########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                                                    @#
#@   Generate a community network                     @#
#@                                                    @#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
########################################################


# China household data, as an argument for the function 'generate_demographic_contact_network'
China_household_data = {
    'household_size_distn':{ 1: 0.2837034254,
                             2: 0.3455088159,
                             3: 0.2238563271,
                             4: 0.0845357644,
                             5: 0.0466034563,
                             6: 0.0126443446,
                             7: 0.0021816290,
                             8: 0.0006086290,
                             9: 0.0001926461,
                             10: 0.0001649623},   # 上海数据
        
    'age_distn':{'0-9':   0.1093,
                 '10-19': 0.1043,
                 '20-29': 0.1437,
                 '30-39': 0.1881 ,
                 '40-49': 0.1552,
                 '50-59': 0.1439,
                 '60-69': 0.0908,
                 '70-79': 0.0437,
                 '80+'  : 0.0210  },    # 2020城市数据

    'household_stats':{'pct_with_under20':          0.3368,
                       'pct_with_over60':           0.3801,
                       'pct_with_under20_over60':   0.0341,
                       'pct_with_over60_givenSingleOccupant':       0.110,
                       'mean_num_under20_givenAtLeastOneUnder20':   1.91 }     # default
    }

########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                                                    @#
#@   Parameters on Omicron and simulation             @#
#@                                                    @#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
########################################################


#%% Specify parameters on Omicron and simulation

 
## Set disease progression rate parameters

### Latent periods (time in Exposed state) 
### Backer Jantien A, et al. Shorter serial intervals in SARS-CoV-2 cases with Omicron BA.1 variant compared with Delta variant, 
### the Netherlands, 13 to 26 December 2021. Euro Surveill. 2022;27(6):pii=2200042. 
### https://doi.org/10.2807/1560-7917.ES.2022.27.6.2200042
latentPeriod_mean, latentPeriod_coeffvar = 3.2, 0.6875        # the second values 


### Presymptomatic periods (time in Pre-symptomatic infectious state)
### Parameters for Delta variant were used here.
### Ma, X. et al. Contact tracing period and epidemiological characteristics of an outbreak of the SARS-CoV-2 Delta variant in Guangzhou. 
### International journal of infectious diseases : IJID : official publication of the International Society for Infectious Diseases 117, 18-23, doi:10.1016/j.ijid.2022.01.034 (2022).
presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar = 3.83, 0.5979     


### (A)symptomatic periods (time in symptomatic or asymptomatic state)
### Menni, C., et al. Symptom prevalence, duration, and risk of hospital admission in individuals infected with SARS-CoV-2 during periods of omicron and delta variant dominance: a prospective observational study from the ZOE COVID Study. 
### The Lancet, 2022. 399(10335): p. 1618-1624.
### Average symptom duration, days, mean (SD): 6.87 (5.21)
symptomaticPeriod_mean, symptomaticPeriod_coeffvar = 6.87, 0.7584           # default


### Onset-to-hospitalization periods (time in symptomatic state before entering hospitalized state for those with severe cases).
### Shen Y, Zheng F, Sun D, et al. Epidemiology and clinical course of COVID-19 in #Shanghai, China. 
### Emerg Microbes Infect. 2020;9(1):1537-1545. #doi:10.1080/22221751.2020.1787103
onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar = 4.0,  0.9027      # 2-7.5 换算CV (4为中位数，CV为SD除以Mean得到)


### Hospitalization-to-discharge periods (time in hospitalized state for those with non-fatal cases).
### Shen Y, Zheng F, Sun D, et al. Epidemiology and clinical course of COVID-19 in #Shanghai, China. 
### Emerg Microbes Infect. 2020;9(1):1537-1545. #doi:10.1080/22221751.2020.1787103
hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar = 16.0, 0.4178    # 16(12-21) 换算CV


### Hospitalization-to-death periods (time in hospitalized state for those with fatal cases)     # Assumed
### Shen Y, Zheng F, Sun D, et al. Epidemiology and clinical course of COVID-19 in #Shanghai, China. 
### Emerg Microbes Infect. 2020;9(1):1537-1545. #doi:10.1080/22221751.2020.1787103
### Z=X+Y, assuming X and Y are independent, then deduce the variance of Y, and then CV
hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar = 12.0, 0.9545


### age-stratified case hospitalization rates and fatality rates for hospitalized cases taken from Verity et al. (2020).
### Verity R, Okell LC, Dorigatti I, Winskill P, Whittaker C, Imai N, Cuomo-Dannenburg G, Thompson H, Walker PGT, Fu H, 
### Dighe A, Griffin JT, Baguelin M, Bhatia S, Boonyasiri A, Cori A, Cucunubá Z, FitzJohn R, Gaythorpe K, Green W, Hamlet A, Hinsley W, Laydon D, Nedjati-Gilani G, Riley S, van Elsland S, Volz E, Wang H, Wang Y, Xi X, Donnelly CA, Ghani AC, Ferguson NM. Estimates of the severity of coronavirus disease 2019: a model-based analysis. Lancet Infect Dis. 2020 Jun;20(6):669-677. doi: 10.1016/S1473-3099(20)30243-7. Epub 2020 Mar 30. Erratum in: Lancet Infect Dis. 2020 Apr 15;: Erratum in: Lancet Infect Dis. 2020 May 4;: PMID: 32240634; PMCID: PMC7158570.
ageGroup_pctHospitalized = {'0-9':      0.0000, 
                            '10-19':    0.0004,
                            '20-29':    0.0104,
                            '30-39':    0.0343,
                            '40-49':    0.0425,
                            '50-59':    0.0816,
                            '60-69':    0.118,
                            '70-79':    0.166,
                            '80+':      0.184 }

ageGroup_hospitalFatalityRate = {'0-9':     0.0000,
                                 '10-19':   0.3627,
                                 '20-29':   0.0577,
                                 '30-39':   0.0426,
                                 '40-49':   0.0694,
                                 '50-59':   0.1532,
                                 '60-69':   0.3381,
                                 '70-79':   0.5187,
                                 '80+':     0.7283 }

 


## Set transmission parameters
R0_mean     = 10.0
R0_coeffvar = 0.35


## Specify the percentage of cases that are asymptomatic.
PCT_ASYMPTOMATIC = 0.90       


### set the transmissibility of quarantined individuals such that the quarantined individuals cannot transmit SARS_CoV-2. 
BETA_Q = 0.0


### Set Pairwise Transmissibility Value to be the Individual Transmissibility Value of the infected individual
BETA_PAIRWISE_MODE  = 'infected'


### Set the Connectivity Correction Factors, used to to weight the transmissibility of interactions according
### to the connectivity of the interacting individuals (or other arbitrary
# DELTA_PAIRWISE_MODE = None


### Set individual susceptibilities (here, we use the default susceptibility is 1).
ALPHA = 1.0


###  Set p to reflect 5% of interactions being with incidental or casual contacts outside their set of close contacts.
P_GLOBALINTXN = 0.05


### Set q to 0.0, which supposes that global interactions are zero for quarantined individuals.
Q_GLOBALINTXN = 0.0


### Set the transition mode, default is "exponential_rates"
TRANSITION_MODE = "time_in_state"


DELTA_PAIRWISE_MODE = None




########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                                                    @#
#@   Parameters on NPIs                               @#
#@                                                    @#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
########################################################


#%% Set Testing, Tracing, & Isolation (TTI) intervention protocol parameters


INTERVENTION_START_PCT_INFECTED = 5/10000       # population disease prevalence that triggers the start of the TTI interventions
AVERAGE_INTRODUCTIONS_PER_DAY   = 0             # average ongoing introductions of the disease from outside


PCT_TESTED_PER_DAY              = 1.0           # max daily test allotment defined as a percent of population size
TEST_FALSENEG_RATE              = 'temporal'    # test false negative rate, will use FN rate that varies with disease time
MAX_PCT_TESTS_FOR_SYMPTOMATICS  = 1.0           # max percent of daily test allotment to use on self-reporting symptomatics
MAX_PCT_TESTS_FOR_TRACES        = 1.0           # max percent of daily test allotment to use on contact traces
RANDOM_TESTING_DEGREE_BIAS      = 0             # magnitude of degree bias in random selections for testing, none here
                                                # 可以设置一个较大的值，以体现家庭中与外界交流多的个体优先被检测
PCT_CONTACTS_TO_TRACE           = 0.90          # percentage of primary cases' contacts that are traced
TRACING_LAG                     = 1             # number of cadence testing days between primary tests and tracing tests

ISOLATION_LAG_SYMPTOMATIC       = 1             # number of days between onset of symptoms and self-isolation of symptomatics
ISOLATION_LAG_POSITIVE          = 1             # test turn-around time (TAT): number of days between administration of test and isolation of positive cases
ISOLATION_LAG_CONTACT           = 0             # number of days between a contact being traced and that contact self-isolating


### Specify the compliance rates (i.e., the percentage of individuals who are compliant) for each intervention type
TESTING_COMPLIANCE_RATE_SYMPTOMATIC                  = 0.5
TESTING_COMPLIANCE_RATE_TRACED                       = 1.0
TESTING_COMPLIANCE_RATE_RANDOM                       = 1.0

TRACING_COMPLIANCE_RATE                              = 1.0

ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL     = 1.0
ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE      = 1.0
ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL        = 1.0
ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE         = 1.0
ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT           = 1.0
ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE  = 1.0





#%% Specify the cadence days (scenarios) to simulate

## Scenarios testing all
cadence_testing_days_SA  = {
    'ScenarioA1_day1~7':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'ScenarioA2_day1~6':   [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    'ScenarioA3_day1~5':   [0, 1, 2, 3, 4, 7, 8, 9, 10, 11],
    'ScenarioA4_day1~7-10-14':   [0, 1, 2, 3, 4, 5, 6, 9, 13],
    'ScenarioA5_day1~4':   [0, 1, 2, 3, 7, 8, 9, 10],
    'ScenarioA6_day1~3':   [0, 1, 2, 7, 8, 9],
    'ScenarioA7_day1247-14':   [0, 1, 3, 6, 13],
    'ScenarioA8_day147-10-14': [0, 3, 6, 9, 13], 
    'ScenarioA9_day12':   [0, 1, 7, 8],
    'ScenarioA10_day1':   [0, 7],
    'ScenarioA11_day1biweekly':   [0],
    'ScenarioA12_day1357':   [0, 2, 4, 6, 7, 9, 11, 13],
    'ScenarioA13_day3456':   [2, 3, 4, 5, 9, 10, 11, 12],
    'ScenarioA14_day4567':   [3, 4, 5, 6, 10, 11, 12, 13],
    'ScenarioA15_day147':   [0, 3, 6, 7, 10, 13],
    'ScenarioA16_day345':   [2, 3, 4, 9, 10, 11],
    'ScenarioA17_day567':   [4, 5, 6, 11, 12, 13]    
     }


cadence_testing_days_SA_Distibute = {
    'ScenarioA18_day1~7':     [0, 1, 2, 3, 4, 5, 6],
    'ScenarioA19_day1~6':   [0, 1, 2, 3, 4, 5],
    'ScenarioA20_day1~5':   [0, 1, 2, 3, 4],
    'ScenarioA21_day1~4':   [0, 1, 2, 3],
    'ScenarioA22_day1~3':   [0, 1, 2],
    'ScenarioA23_day12':   [0, 1],
    'ScenarioA24_day1357':   [0, 2, 4, 6],
    'ScenarioA25_day3456':   [2, 3, 4, 5],
    'ScenarioA26_day4567':   [3, 4, 5, 6],
    'ScenarioA27_day147':   [0, 3, 6],
    'ScenarioA28_day345':   [2, 3, 4],
    'ScenarioA29_day567':   [4, 5, 6]
     }


cadence_testing_days_SB_HSHD     = {
    'ScenarioB1_day1~7':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'ScenarioB2_day1~6':   [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    'ScenarioB3_day1~5':   [0, 1, 2, 3, 4, 7, 8, 9, 10, 11],
    'ScenarioB4_day1~7-10-14':   [0, 1, 2, 3, 4, 5, 6, 9, 13],
    'ScenarioB5_day1~4':   [0, 1, 2, 3, 7, 8, 9, 10],
    'ScenarioB6_day1~3':   [0, 1, 2, 7, 8, 9],
    'ScenarioB7_day1247-14':   [0, 1, 3, 6, 13],
    'ScenarioB8_day147-10-14': [0, 3, 6, 9, 13], 
    'ScenarioB9_day12':   [0, 1, 7, 8],
    'ScenarioB10_day1':   [0, 7],
    'ScenarioB11_day1biweekly':   [0],
    'ScenarioB12_day1357':   [0, 2, 4, 6, 7, 9, 11, 13],
    'ScenarioB13_day3456':   [2, 3, 4, 5, 9, 10, 11, 12],
    'ScenarioB14_day4567':   [3, 4, 5, 6, 10, 11, 12, 13],
    'ScenarioB15_day147':   [0, 3, 6, 7, 10, 13],
    'ScenarioB16_day345':   [2, 3, 4, 9, 10, 11],
    'ScenarioB17_day567':   [4, 5, 6, 11, 12, 13]
     }


cadence_testing_days_SB_HSHD_Distribute     = {
    'ScenarioB18_day1~7':     [0, 1, 2, 3, 4, 5, 6],
    'ScenarioB19_day1~6':   [0, 1, 2, 3, 4, 5],
    'ScenarioB20_day1~5':   [0, 1, 2, 3, 4],
    'ScenarioB21_day1~4':   [0, 1, 2, 3],
    'ScenarioB22_day1~3':   [0, 1, 2],
    'ScenarioB23_day12':   [0, 1],
    'ScenarioB24_day1357':   [0, 2, 4, 6],
    'ScenarioB25_day3456':   [2, 3, 4, 5],
    'ScenarioB26_day4567':   [3, 4, 5, 6],
    'ScenarioB27_day147':   [0, 3, 6],
    'ScenarioB28_day345':   [2, 3, 4],
    'ScenarioB29_day567':   [4, 5, 6]
     }

cadence_testing_days_SC_fixSample    = {
    'ScenarioC1_day1~7':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'ScenarioC2_day1~6':   [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    'ScenarioC3_day1~5':   [0, 1, 2, 3, 4, 7, 8, 9, 10, 11],
    'ScenarioC4_day1~7-10-14':   [0, 1, 2, 3, 4, 5, 6, 9, 13],
    'ScenarioC5_day1~4':   [0, 1, 2, 3, 7, 8, 9, 10],
    'ScenarioC6_day1~3':   [0, 1, 2, 7, 8, 9],
    'ScenarioC7_day1247-14':   [0, 1, 3, 6, 13],
    'ScenarioC8_day147-10-14': [0, 3, 6, 9, 13], 
    'ScenarioC9_day12':   [0, 1, 7, 8],
    'ScenarioC10_day1':   [0, 7],
    'ScenarioC11_day1biweekly':   [0],
    'ScenarioC12_day1357':   [0, 2, 4, 6, 7, 9, 11, 13],
    'ScenarioC13_day3456':   [2, 3, 4, 5, 9, 10, 11, 12],
    'ScenarioC14_day4567':   [3, 4, 5, 6, 10, 11, 12, 13],
    'ScenarioC15_day147':   [0, 3, 6, 7, 10, 13],
    'ScenarioC16_day345':   [2, 3, 4, 9, 10, 11],
    'ScenarioC17_day567':   [4, 5, 6, 11, 12, 13]
     }

cadence_testing_days_SC_fixSample_distribute     = {
    'ScenarioC18_day1~7':     [0, 1, 2, 3, 4, 5, 6],
    'ScenarioC19_day1~6':   [0, 1, 2, 3, 4, 5],
    'ScenarioC20_day1~5':   [0, 1, 2, 3, 4],
    'ScenarioC21_day1~4':   [0, 1, 2, 3],
    'ScenarioC22_day1~3':   [0, 1, 2],
    'ScenarioC23_day12':   [0, 1],
    'ScenarioC24_day1357':   [0, 2, 4, 6],
    'ScenarioC25_day3456':   [2, 3, 4, 5],
    'ScenarioC26_day4567':   [3, 4, 5, 6],
    'ScenarioC27_day147':   [0, 3, 6],
    'ScenarioC28_day345':   [2, 3, 4],
    'ScenarioC29_day567':   [4, 5, 6]
     }








#%% Set parameters on the simulation and then run them by cycle


## Set the total number of population
N = 10000

## Set the initial prevalence
INIT_EXPOSED = 5   

### Set the max simulation time to 100 days.
T = 100

N_sim=180        # total number of simulation
    
current_Sim=161    # current simulation

while current_Sim <= N_sim:
    
    # Generate community-level contact networks
    # Distancing scale, to generate a quarantine version of the network where a majority of the out-of-household edges have been removed.
    # distancing_scales=[1.442695]    
    # The distancing scale value of 1.442695 is chosen so that 95% of individuals have no more than a single out-of-household contact (edge) in the quarantine network.
    demographic_graphs, individual_ageGroups, households = generate_demographic_contact_network(
        N=N, demographic_data=China_household_data,
        distancing_scales=[1.442695],
        isolation_groups=[])
    
    
    # Rename each network in the list
    G_baseline   = demographic_graphs['baseline']
    G_quarantine = demographic_graphs['distancingScale1.442695']
    
    
    # Information on households
    households_indices = [household['indices'] for household in households]
    households_sizes=[household['size'] for household in households]
    households_ageBrackets=[household['ageBrackets'] for household in households]
    
    ## Set severity parameters

    ### Specify age-stratified case hospitalization rates for hospitalized cases
    PCT_HOSPITALIZED = [ageGroup_pctHospitalized[ageGroup] for ageGroup in individual_ageGroups]
    
    
    ### Specify age-stratified case fatality rates for hospitalized cases, again using rates taken from Verity et al. (2020).
    PCT_FATALITY = [ageGroup_hospitalFatalityRate[ageGroup] for ageGroup in individual_ageGroups]
 

        
    ## Here we generate distributions of values for each parameter, 
    ## and some parameter value are specified in an age-stratified manner, 
    ## thus specifying a realistically heterogeneous population.
    
    ### Latent periods (time in Exposed state) 
    SIGMA   = 1 / gamma_dist(latentPeriod_mean, latentPeriod_coeffvar, N)
    
    
    ### Presymptomatic periods (time in Pre-symptomatic infectious state)
    LAMDA   = 1 / gamma_dist(presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar, N)
    
    
    ### (A)symptomatic periods (time in symptomatic or asymptomatic state)
    GAMMA   = 1 / gamma_dist(symptomaticPeriod_mean, symptomaticPeriod_coeffvar, N)
    
    
    ### Infectious period = Presymptomatic period + (A)symptomatic periods
    infectiousPeriod = 1/LAMDA + 1/GAMMA
    
    
    ### Onset-to-hospitalization periods (time in symptomatic state before entering hospitalized state for those with severe cases).
    ETA     = 1 / gamma_dist(onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar, N)
    
    
    ### Hospitalization-to-discharge periods (time in hospitalized state for those with non-fatal cases).
    GAMMA_H = 1 / gamma_dist(hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar, N)
    
    
    ### Hospitalization-to-death periods (time in hospitalized state for those with fatal cases)     # default
    MU_H    = 1 / gamma_dist(hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar, N)

    
    ### Generate individual R0s randomly with Gamma distribution
    R0 = gamma_dist(R0_mean, R0_coeffvar, N)
    
    ### The means of the Individual Transmissibility Values for infectious subpopulations are used to calculate the global transmission terms.
    BETA = 1/infectiousPeriod * R0
    
    ### transmissibility of presymptomatic and asymptomatic individuals
    BETA_ASYM = 0.50 * BETA
    
    ## Randomly assign a True/False compliance to each individual according to the rates set above. Individuals whose compliance is set to True for a given intervention will participate in that intervention, individuals set to False will not.
    TESTING_COMPLIANCE_RANDOM                        = (np.random.rand(N) < TESTING_COMPLIANCE_RATE_RANDOM)
    TESTING_COMPLIANCE_TRACED                        = (np.random.rand(N) < TESTING_COMPLIANCE_RATE_TRACED)
    TESTING_COMPLIANCE_SYMPTOMATIC                   = (np.random.rand(N) < TESTING_COMPLIANCE_RATE_SYMPTOMATIC)
    
    TRACING_COMPLIANCE                               = (np.random.rand(N) < TRACING_COMPLIANCE_RATE)
    
    ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL      = (np.random.rand(N) < ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL)
    ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE       = (np.random.rand(N) < ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE)
    ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL         = (np.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL)
    ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE          = (np.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE)
    ISOLATION_COMPLIANCE_POSITIVE_CONTACT            = (np.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT)
    ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE   = (np.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE)

   
    ##########################################################################
    ## Simulate scenarios that test all people on each cadence day
    for TESTING_CADENCE in cadence_testing_days_SA:  
        
        # Initializing the model    
        model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                      beta=BETA, beta_asym=BETA_ASYM,
                                      sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                      gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H, 
                                      a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,              
                                      beta_pairwise_mode=BETA_PAIRWISE_MODE, 
                                      transition_mode = TRANSITION_MODE,
                                      G_Q=G_quarantine, q=0.0, beta_Q=BETA_Q, isolation_time=14,
                                      initE=INIT_EXPOSED)    
      
        ## Running the model
        run_tti_sim_test_random(model, T, current_sim=current_Sim, max_dt=1.0,
                    intervention_start_pct_infected=INTERVENTION_START_PCT_INFECTED, average_introductions_per_day=AVERAGE_INTRODUCTIONS_PER_DAY,
                    testing_cadence=TESTING_CADENCE, pct_tested_per_day=PCT_TESTED_PER_DAY, test_falseneg_rate=TEST_FALSENEG_RATE, 
                    testing_compliance_symptomatic=TESTING_COMPLIANCE_SYMPTOMATIC, max_pct_tests_for_symptomatics=MAX_PCT_TESTS_FOR_SYMPTOMATICS,
                    testing_compliance_traced=TESTING_COMPLIANCE_TRACED, max_pct_tests_for_traces=MAX_PCT_TESTS_FOR_TRACES,
                    testing_compliance_random=TESTING_COMPLIANCE_RANDOM, random_testing_degree_bias=RANDOM_TESTING_DEGREE_BIAS,
                    tracing_compliance=TRACING_COMPLIANCE, pct_contacts_to_trace=PCT_CONTACTS_TO_TRACE, tracing_lag=TRACING_LAG,
                    isolation_compliance_symptomatic_individual=ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL, isolation_compliance_symptomatic_groupmate=ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE, 
                    isolation_compliance_positive_individual=ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL, isolation_compliance_positive_groupmate=ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE,
                    isolation_compliance_positive_contact=ISOLATION_COMPLIANCE_POSITIVE_CONTACT, isolation_compliance_positive_contactgroupmate=ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE,
                    isolation_lag_symptomatic=ISOLATION_LAG_SYMPTOMATIC, isolation_lag_positive=ISOLATION_LAG_POSITIVE, 
                    isolation_groups=households_indices,
                    cadence_cycle_length=14, 
                    cadence_testing_days=cadence_testing_days_SA)
    ##########################################################################
    
    
    
    ##########################################################################
    ## Simulate scenarios that distribute the testing people equally on each cadence day
    for TESTING_CADENCE in cadence_testing_days_SA_Distibute: 
        
        # # 每轮检测所用天数        
        CADENCE_CYCLE_LEN = len(cadence_testing_days_SA_Distibute[TESTING_CADENCE])
        
        # # max daily test allotment defined as a percent of population size    
        PCT_TESTED_PER_DAY_SA = 1.0/CADENCE_CYCLE_LEN   
        
        # Initializing the model    
        model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                     beta=BETA, beta_asym=BETA_ASYM,
                                     sigma=SIGMA, lamda=LAMDA, gamma=GAMMA, 
                                     gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H, 
                                     a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,              
                                     alpha=ALPHA, beta_pairwise_mode=BETA_PAIRWISE_MODE, 
                                     delta_pairwise_mode=DELTA_PAIRWISE_MODE,
                                     G_Q=G_quarantine, q=0, beta_Q=BETA_Q, isolation_time=14,
                                     initE=INIT_EXPOSED)     
      
        ## Running the model
        run_tti_sim_random_distribute(model, T, max_dt=1.0, current_sim=current_Sim,
            intervention_start_pct_infected=INTERVENTION_START_PCT_INFECTED, 
            average_introductions_per_day=AVERAGE_INTRODUCTIONS_PER_DAY,
            testing_cadence=TESTING_CADENCE, 
            pct_tested_per_day=PCT_TESTED_PER_DAY_SA, 
            test_falseneg_rate=TEST_FALSENEG_RATE, 
            testing_compliance_symptomatic=TESTING_COMPLIANCE_SYMPTOMATIC, 
            max_pct_tests_for_symptomatics=MAX_PCT_TESTS_FOR_SYMPTOMATICS,
            testing_compliance_traced=TESTING_COMPLIANCE_TRACED, 
            max_pct_tests_for_traces=MAX_PCT_TESTS_FOR_TRACES,
            testing_compliance_random=TESTING_COMPLIANCE_RANDOM, 
            random_testing_degree_bias=RANDOM_TESTING_DEGREE_BIAS,
            # households_testing_degree_bias=HOUSEHOLDS_TESTING_DEGREE_BIAS,
            tracing_compliance=TRACING_COMPLIANCE, 
            pct_contacts_to_trace=PCT_CONTACTS_TO_TRACE, 
            tracing_lag=TRACING_LAG,
            isolation_compliance_symptomatic_individual=ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL, 
            isolation_compliance_symptomatic_groupmate=ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE, 
            isolation_compliance_positive_individual=ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL, 
            isolation_compliance_positive_groupmate=ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE,
            isolation_compliance_positive_contact=ISOLATION_COMPLIANCE_POSITIVE_CONTACT, 
            isolation_compliance_positive_contactgroupmate=ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE,
            isolation_lag_symptomatic=ISOLATION_LAG_SYMPTOMATIC, 
            isolation_lag_positive=ISOLATION_LAG_POSITIVE,
            isolation_groups=households_indices,
            cadence_cycle_length=7, 
            cadence_testing_days=cadence_testing_days_SA_Distibute)
    ##########################################################################
        
    
        
    
    ##########################################################################
    ## Simulate scenarios that distribute the testing people chosen from each household
    for TESTING_CADENCE in cadence_testing_days_SB_HSHD: 
        
        # Initializing the model    
        model = ExtSEIRSNetworkModel_HSHD(G=G_baseline,
                                     HI=households_indices, HS=households_sizes, HAGE= households_ageBrackets,
                                     p=P_GLOBALINTXN,
                                     beta=BETA, beta_asym=BETA_ASYM,
                                     sigma=SIGMA, lamda=LAMDA, gamma=GAMMA, 
                                     gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H, 
                                     a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,              
                                     alpha=ALPHA, beta_pairwise_mode=BETA_PAIRWISE_MODE, 
                                     delta_pairwise_mode=DELTA_PAIRWISE_MODE,
                                     G_Q=G_quarantine, q=0, beta_Q=BETA_Q, isolation_time=14,
                                     initE=INIT_EXPOSED)     
      
        ## Running the model
        run_tti_sim_household(model, T, max_dt=1.0,
                    current_sim=current_Sim, choose_person=1,            
                    intervention_start_pct_infected=INTERVENTION_START_PCT_INFECTED,             
                    average_introductions_per_day=AVERAGE_INTRODUCTIONS_PER_DAY,
                    testing_cadence=TESTING_CADENCE, 
                    pct_tested_per_day=PCT_TESTED_PER_DAY, 
                    test_falseneg_rate=TEST_FALSENEG_RATE, 
                    testing_compliance_symptomatic=TESTING_COMPLIANCE_SYMPTOMATIC, 
                    max_pct_tests_for_symptomatics=MAX_PCT_TESTS_FOR_SYMPTOMATICS,
                    testing_compliance_traced=TESTING_COMPLIANCE_TRACED, 
                    max_pct_tests_for_traces=MAX_PCT_TESTS_FOR_TRACES,
                    testing_compliance_random=TESTING_COMPLIANCE_RANDOM, 
                    random_testing_degree_bias=RANDOM_TESTING_DEGREE_BIAS,
                    tracing_compliance=TRACING_COMPLIANCE, 
                    pct_contacts_to_trace=PCT_CONTACTS_TO_TRACE, 
                    tracing_lag=TRACING_LAG,
                    isolation_compliance_symptomatic_individual=ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL, 
                    isolation_compliance_symptomatic_groupmate=ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE, 
                    isolation_compliance_positive_individual=ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL, 
                    isolation_compliance_positive_groupmate=ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE,
                    isolation_compliance_positive_contact=ISOLATION_COMPLIANCE_POSITIVE_CONTACT, 
                    isolation_compliance_positive_contactgroupmate=ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE,
                    isolation_lag_symptomatic=ISOLATION_LAG_SYMPTOMATIC, 
                    isolation_lag_positive=ISOLATION_LAG_POSITIVE,
                    isolation_groups=households_indices,
                    cadence_cycle_length=14, 
                    cadence_testing_days=cadence_testing_days_SB_HSHD)
    ##########################################################################
        
    
        
    
    ##########################################################################
    ## Simulate scenarios that distribute the testing people chosen from each household
    for TESTING_CADENCE in cadence_testing_days_SB_HSHD_Distribute: 
        
        # Initializing the model    
        model = ExtSEIRSNetworkModel_HSHD(G=G_baseline,
                                     HI=households_indices, HS=households_sizes, HAGE= households_ageBrackets,
                                     p=P_GLOBALINTXN,
                                     beta=BETA, beta_asym=BETA_ASYM,
                                     sigma=SIGMA, lamda=LAMDA, gamma=GAMMA, 
                                     gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H, 
                                     a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,              
                                     alpha=ALPHA, beta_pairwise_mode=BETA_PAIRWISE_MODE, 
                                     delta_pairwise_mode=DELTA_PAIRWISE_MODE,
                                     G_Q=G_quarantine, q=0, beta_Q=BETA_Q, isolation_time=14,
                                     initE=INIT_EXPOSED)     
      
        ## Running the model
        run_tti_sim_household_distribute(model, T, max_dt=1.0,
                    current_sim=current_Sim, choose_person=1,
                    intervention_start_pct_infected=INTERVENTION_START_PCT_INFECTED, 
                    average_introductions_per_day=AVERAGE_INTRODUCTIONS_PER_DAY,
                    testing_cadence=TESTING_CADENCE, 
                    pct_tested_per_day=PCT_TESTED_PER_DAY, 
                    test_falseneg_rate=TEST_FALSENEG_RATE, 
                    testing_compliance_symptomatic=TESTING_COMPLIANCE_SYMPTOMATIC, 
                    max_pct_tests_for_symptomatics=MAX_PCT_TESTS_FOR_SYMPTOMATICS,
                    testing_compliance_traced=TESTING_COMPLIANCE_TRACED, 
                    max_pct_tests_for_traces=MAX_PCT_TESTS_FOR_TRACES,
                    testing_compliance_random=TESTING_COMPLIANCE_RANDOM, 
                    random_testing_degree_bias=RANDOM_TESTING_DEGREE_BIAS,
                    # households_testing_degree_bias=HOUSEHOLDS_TESTING_DEGREE_BIAS,
                    tracing_compliance=TRACING_COMPLIANCE, 
                    pct_contacts_to_trace=PCT_CONTACTS_TO_TRACE, 
                    tracing_lag=TRACING_LAG,
                    isolation_compliance_symptomatic_individual=ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL, 
                    isolation_compliance_symptomatic_groupmate=ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE, 
                    isolation_compliance_positive_individual=ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL, 
                    isolation_compliance_positive_groupmate=ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE,
                    isolation_compliance_positive_contact=ISOLATION_COMPLIANCE_POSITIVE_CONTACT, 
                    isolation_compliance_positive_contactgroupmate=ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE,
                    isolation_lag_symptomatic=ISOLATION_LAG_SYMPTOMATIC, 
                    isolation_lag_positive=ISOLATION_LAG_POSITIVE,
                    isolation_groups=households_indices,
                    cadence_cycle_length=7, 
                    cadence_testing_days=cadence_testing_days_SB_HSHD_Distribute)
    ##########################################################################
        
    
        
    
    ##########################################################################
    ## Simulate scenarios that test people chosen then fixed
    for TESTING_CADENCE in cadence_testing_days_SC_fixSample: 
        
        PCT_TESTED_PER_DAY_SC=len(households_indices)/N
        
        # Initializing the model  
        model = ExtSEIRSNetworkModel(G=G_baseline,
                                     p=P_GLOBALINTXN,
                                     beta=BETA, beta_asym=BETA_ASYM,
                                     sigma=SIGMA, lamda=LAMDA, gamma=GAMMA, 
                                     gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H, 
                                     a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,              
                                     alpha=ALPHA, beta_pairwise_mode=BETA_PAIRWISE_MODE, 
                                     delta_pairwise_mode=DELTA_PAIRWISE_MODE,
                                     G_Q=G_quarantine, q=0, beta_Q=BETA_Q, isolation_time=14,
                                     initE=INIT_EXPOSED)     
      
        ## Running the model
        run_tti_sim_fixSample(model, T, current_sim=current_Sim, max_dt=1.0,
            intervention_start_pct_infected=INTERVENTION_START_PCT_INFECTED, 
            average_introductions_per_day=AVERAGE_INTRODUCTIONS_PER_DAY,
            testing_cadence=TESTING_CADENCE, 
            pct_tested_per_day=PCT_TESTED_PER_DAY_SC, 
            test_falseneg_rate=TEST_FALSENEG_RATE, 
            testing_compliance_symptomatic=TESTING_COMPLIANCE_SYMPTOMATIC, 
            max_pct_tests_for_symptomatics=MAX_PCT_TESTS_FOR_SYMPTOMATICS,
            testing_compliance_traced=TESTING_COMPLIANCE_TRACED, 
            max_pct_tests_for_traces=MAX_PCT_TESTS_FOR_TRACES,
            testing_compliance_random=TESTING_COMPLIANCE_RANDOM, 
            random_testing_degree_bias=RANDOM_TESTING_DEGREE_BIAS,
            # households_testing_degree_bias=HOUSEHOLDS_TESTING_DEGREE_BIAS,
            tracing_compliance=TRACING_COMPLIANCE, 
            pct_contacts_to_trace=PCT_CONTACTS_TO_TRACE, 
            tracing_lag=TRACING_LAG,
            isolation_compliance_symptomatic_individual=ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL, 
            isolation_compliance_symptomatic_groupmate=ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE, 
            isolation_compliance_positive_individual=ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL, 
            isolation_compliance_positive_groupmate=ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE,
            isolation_compliance_positive_contact=ISOLATION_COMPLIANCE_POSITIVE_CONTACT, 
            isolation_compliance_positive_contactgroupmate=ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE,
            isolation_lag_symptomatic=ISOLATION_LAG_SYMPTOMATIC, 
            isolation_lag_positive=ISOLATION_LAG_POSITIVE,
            isolation_groups=households_indices,
            cadence_cycle_length=14, 
            cadence_testing_days=cadence_testing_days_SC_fixSample)
    ##########################################################################
        
    
        
    
    ##########################################################################
    ## Simulate scenarios that distribute the testing people chosen then fixed
    for TESTING_CADENCE in cadence_testing_days_SC_fixSample_distribute: 
        
        PCT_TESTED_PER_DAY_SCD=len(households_indices)/N
        
        # Initializing the model    
        model = ExtSEIRSNetworkModel(G=G_baseline,
                                     p=P_GLOBALINTXN,
                                     beta=BETA, beta_asym=BETA_ASYM,
                                     sigma=SIGMA, lamda=LAMDA, gamma=GAMMA, 
                                     gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H, 
                                     a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,              
                                     alpha=ALPHA, beta_pairwise_mode=BETA_PAIRWISE_MODE, 
                                     delta_pairwise_mode=DELTA_PAIRWISE_MODE,
                                     G_Q=G_quarantine, q=0, beta_Q=BETA_Q, isolation_time=14,
                                     initE=INIT_EXPOSED)     
      
        ## Running the model
        run_tti_sim_fixSample_distribute(model, T, current_sim=current_Sim, max_dt=1.0,
            intervention_start_pct_infected=INTERVENTION_START_PCT_INFECTED, 
            average_introductions_per_day=AVERAGE_INTRODUCTIONS_PER_DAY,
            testing_cadence=TESTING_CADENCE, 
            pct_tested_per_day=PCT_TESTED_PER_DAY_SCD, 
            test_falseneg_rate=TEST_FALSENEG_RATE, 
            testing_compliance_symptomatic=TESTING_COMPLIANCE_SYMPTOMATIC, 
            max_pct_tests_for_symptomatics=MAX_PCT_TESTS_FOR_SYMPTOMATICS,
            testing_compliance_traced=TESTING_COMPLIANCE_TRACED, 
            max_pct_tests_for_traces=MAX_PCT_TESTS_FOR_TRACES,
            testing_compliance_random=TESTING_COMPLIANCE_RANDOM, 
            random_testing_degree_bias=RANDOM_TESTING_DEGREE_BIAS,
            # households_testing_degree_bias=HOUSEHOLDS_TESTING_DEGREE_BIAS,
            tracing_compliance=TRACING_COMPLIANCE, 
            pct_contacts_to_trace=PCT_CONTACTS_TO_TRACE, 
            tracing_lag=TRACING_LAG,
            isolation_compliance_symptomatic_individual=ISOLATION_COMPLIANCE_SYMPTOMATIC_INDIVIDUAL, 
            isolation_compliance_symptomatic_groupmate=ISOLATION_COMPLIANCE_SYMPTOMATIC_GROUPMATE, 
            isolation_compliance_positive_individual=ISOLATION_COMPLIANCE_POSITIVE_INDIVIDUAL, 
            isolation_compliance_positive_groupmate=ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE,
            isolation_compliance_positive_contact=ISOLATION_COMPLIANCE_POSITIVE_CONTACT, 
            isolation_compliance_positive_contactgroupmate=ISOLATION_COMPLIANCE_POSITIVE_CONTACTGROUPMATE,
            isolation_lag_symptomatic=ISOLATION_LAG_SYMPTOMATIC, 
            isolation_lag_positive=ISOLATION_LAG_POSITIVE,
            isolation_groups=households_indices,
            cadence_cycle_length=7, 
            cadence_testing_days=cadence_testing_days_SC_fixSample_distribute)
     
        # End the current simulaiton.
    
        
    # End the current simulaiton.    
    current_Sim+=1   