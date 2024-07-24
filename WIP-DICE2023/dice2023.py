import numpy as np

# Time periods
t = range(1, 82)  # 1 to 81

params = {
    "T": 81,  # Time periods
    "ifopt": 1,  # Indicator where optimized is 1 and base is 0
    "gama": 0.300,  # Capital elasticity in production function
    "pop1": 7752.9,  # Initial world population 2020 (millions)
    "popadj": 0.145,  # Growth rate to calibrate to 2050 pop projection
    "popasym": 10825.,  # Asymptotic population (millions)
    "dk": 0.100,  # Depreciation rate on capital (per year)
    "q1": 135.7,  # Initial world output 2020 (trill 2019 USD)
    "AL1": 5.84,  # Initial level of total factor productivity
    "gA1": 0.066,  # Initial growth rate for TFP per 5 years
    "delA": 0.0015,  # Decline rate of TFP per 5 years

    # Emissions parameters
    "gsigma1": -0.015,  # Initial growth of sigma (per year)
    "delgsig": 0.96,  # Decline rate of gsigma per period
    "asymgsig": -0.005,  # Asymptotic gsigma
    "e1": 37.56,  # Industrial emissions 2020 (GtCO2 per year)
    "miu1": 0.05,  # Emissions control rate historical 2020
    "fosslim": 6000,  # Maximum cumulative extraction fossil fuels (GtC)
    "CumEmiss0": 633.5,  # Cumulative emissions 2020 (GtC)

    # Climate damage parameters
    "a1": 0,  # Damage intercept
    "a2base": 0.003467,  # Damage quadratic term
    "a3": 2.00,  # Damage exponent

    # Abatement cost
    "expcost2": 2.6,  # Exponent of control cost function
    "pback2050": 515.,  # Cost of backstop 2019$ per tCO2 2050
    "gback": -0.012,  # Initial cost decline backstop cost per year
    "cprice1": 6,  # Carbon price 2020 2019$ per tCO2
    "gcprice": 0.025,  # Growth rate of base carbon price per year

    # Limits on emissions controls
    "limmiu2070": 1.0,  # Emission control limit from 2070
    "limmiu2120": 1.1,  # Emission control limit from 2120
    "limmiu2200": 1.05,  # Emission control limit from 2220
    "limmiu2300": 1.0,  # Emission control limit from 2300
    "delmiumax": 0.12,  # Emission control delta limit per period

    # Preferences and other parameters
    "betaclim": 0.5,  # Climate beta
    "elasmu": 0.95,  # Elasticity of marginal utility of consumption
    "prstp": 0.001,  # Pure rate of social time preference
    "pi": 0.05,  # Capital risk premium
    "k0": 295,  # Initial capital stock calibrated (1012 2019 USD)
    "siggc1": 0.01,  # Annual standard deviation of consumption growth

    # Scaling parameters
    "tstep": 5,  # Years per Period
    "SRF": 1000000,  # Scaling factor discounting
    "scale1": 0.00891061,  # Multiplicative scaling coefficient
    "scale2": -6275.91  # Additive scaling coefficient
}
T = params["T"]

L = np.zeros(T)
aL = np.zeros(T)
sigma = np.zeros(T)
MIU = np.zeros(T)
TATM = np.zeros(T)
MAT = np.zeros(T)
ML = np.zeros(T)
Y = np.zeros(T)
YGROSS = np.zeros(T)
C = np.zeros(T)
K = np.zeros(T)
I = np.zeros(T)
S = np.zeros(T)
CCATOT = np.zeros(T)
DAMAGES = np.zeros(T)
ABATECOST = np.zeros(T)
ECO2 = np.zeros(T)

L = np.zeros(T)###  Level of population and labor
aL = np.zeros(T)### Level of total factor productivity
sigma = np.zeros(T)## CO2-emissions output ratio
sigmatot = np.zeros(T)# GHG-output ratio
gA = np.zeros(T)### Growth rate of productivity from
gL = np.zeros(T)### Growth rate of labor and population
gcost1 = 0###Growth of cost factor
gsig = np.zeros(T)##  Change in sigma (rate of decarbonization)
eland = np.zeros(T)## Emissions from deforestation (GtCO2 per year)
cost1tot = np.zeros(T)# Abatement cost adjusted for backstop and sigma
pbacktime = np.zeros(T)#Backstop price 2019$ per ton CO2
optlrsav = 0 ## Optimal long-run savings rate used for transversality
scc = np.zeros(T)###Social cost of carbon
cpricebase = np.zeros(T)  Carbon price in base case
ppm = np.zeros(T)###Atmospheric concentrations parts per million
atfrac2020 = np.zeros(T)  Atmospheric share since 2020
atfrac1765 = np.zeros(T)  Atmospheric fraction of emissions since 1765
abaterat = np.zeros(T)# Abatement cost per net output
miuup = np.zeros(T)## Upper bound on miu
gbacktime = np.zeros(T)#Decline rate of backstop price

def create_time_indicators(t):
    t = np.array(t)
    tfirst = np.zeros_like(t, dtype=bool)
    tlast = np.zeros_like(t, dtype=bool)
    
    tfirst[0] = True  # First element
    tlast[-1] = True  # Last element
    
    return tfirst, tlast

# model eq dynamics

def eco2eq(t, sigma, YGROSS, eland, MIU):
    # Calculate effective CO2 emissions considering emission control rate.
    # sigma: CO2-emissions output ratio
    # YGROSS: Gross world product gross of abatement and damages
    # eland: Emissions from deforestation
    # MIU: Emission control rate
    return (sigma[t] * YGROSS[t] + eland[t]) * (1 - MIU[t])

def eindeq(t, sigma, YGROSS, MIU):
    # Calculate industrial CO2 emissions.
    # sigma: CO2-emissions output ratio
    # YGROSS: Gross world product gross of abatement and damages
    # MIU: Emission control rate
    return (sigma[t] * YGROSS[t]) * (1 - MIU[t])

def eco2Eeq(t, sigma, YGROSS, eland, CO2E_GHGabateB, MIU):
    # Calculate total effective CO2 emissions including GHG abatement.
    # sigma: CO2-emissions output ratio
    # YGROSS: Gross world product gross of abatement and damages
    # eland: Emissions from deforestation
    # CO2E_GHGabateB: Abated GHG emissions
    # MIU: Emission control rate
    return (sigma[t] * YGROSS[t] + eland[t] + CO2E_GHGabateB[t]) * (1 - MIU[t])

def F_GHGabateEQ(t, F_GHGabate, CO2E_GHGabateB, MIU, Fcoef1, Fcoef2):
    # Calculate GHG abatement factor.
    # F_GHGabate: GHG abatement
    # CO2E_GHGabateB: Abated GHG emissions
    # MIU: Emission control rate
    # Fcoef1, Fcoef2: Coefficients for abatement calculation
    return Fcoef2 * F_GHGabate[t] + Fcoef1 * CO2E_GHGabateB[t] * (1 - MIU[t])

def ccatoteq(t, CCATOT, ECO2):
    # Calculate total carbon emissions in GtC.
    # CCATOT: Total carbon emissions
    # ECO2: Effective CO2 emissions
    return CCATOT[t] + ECO2[t] * (5 / 3.666)

def damfraceq(t, TATM, a1, a2base, a3):
    # Calculate damage fraction based on atmospheric temperature.
    # TATM: Atmospheric temperature
    # a1, a2base, a3: Damage fraction coefficients
    return (a1 * TATM[t]) + (a2base * (TATM[t])**a3)

def dameq(t, YGROSS, DAMFRAC):
    # Calculate economic damages due to climate change.
    # YGROSS: Gross world product gross of abatement and damages
    # DAMFRAC: Damages as fraction of gross output
    return YGROSS[t] * DAMFRAC[t]

def abateeq(t, YGROSS, COST1TOT, MIU, EXPCOST2):
    # Calculate abatement cost.
    # YGROSS: Gross world product gross of abatement and damages
    # COST1TOT: Total abatement cost coefficient
    # MIU: Emission control rate
    # EXPCOST2: Exponent for abatement cost calculation
    return YGROSS[t] * COST1TOT[t] * (MIU[t]**EXPCOST2)

def mcabateeq(t, pbacktime, MIU, expcost2):
    # Calculate marginal cost of abatement.
    # pbacktime: Backstop price
    # MIU: Emission control rate
    # expcost2: Exponent for abatement cost calculation
    return pbacktime[t] * MIU[t]**(expcost2-1)

def carbpriceeq(t, pbacktime, MIU, expcost2):
    # Calculate carbon price.
    # pbacktime: Backstop price
    # MIU: Emission control rate
    # expcost2: Exponent for abatement cost calculation
    return pbacktime[t] * (MIU[t])**(expcost2-1)

def ygrosseq(t, AL, L, K, gama):
    # Calculate gross world product.
    # AL: Total factor productivity
    # L: Population and labor
    # K: Capital stock
    # gama: Output elasticity of capital
    return (AL[t] * (L[t]/1000)**(1-gama)) * (K[t]**gama)

def yneteq(t, YGROSS, damfrac):
    # Calculate net world product considering damages.
    # YGROSS: Gross world product gross of abatement and damages
    # damfrac: Damages as fraction of gross output
    return YGROSS[t] * (1 - damfrac[t])

def yy(t, YNET, ABATECOST):
    # Calculate net world product after abatement costs.
    # YNET: Output net of damages
    # ABATECOST: Cost of emissions reductions
    return YNET[t] - ABATECOST[t]

def cc(t, Y, I):
    # Calculate consumption.
    # Y: Gross world product net of abatement and damages
    # I: Investment
    return Y[t] - I[t]

def cpce(t, C, L):
    # Calculate per capita consumption.
    # C: Consumption
    # L: Population and labor
    return 1000 * C[t] / L[t]

def seq(t, S, Y):
    # Calculate savings rate.
    # S: Gross savings rate
    # Y: Gross world product net of abatement and damages
    return S[t] * Y[t]

def kk(t, K, I, dk, tstep):
    # Calculate capital stock.
    # K: Capital stock
    # I: Investment
    # dk: Depreciation rate
    # tstep: Time step
    return (1-dk)**tstep * K[t] + tstep * I[t]

def RFACTLONGeq(t, cpc, rr, SRF, elasmu):
    # Calculate long-term interest rate factor.
    # cpc: Per capita consumption
    # rr: Social time preference rate
    # SRF: Scaling factor
    # elasmu: Elasticity of marginal utility of consumption
    return SRF * (cpc[t+1]/cpc[0])**(-elasmu) * rr[t+1]

def RLONGeq(t, RFACTLONG, SRF):
    # Calculate long-term real interest rate.
    # RFACTLONG: Long-term interest rate factor
    # SRF: Scaling factor
    return -np.log(RFACTLONG[t+1]/SRF) / (5*t)

def RSHORTeq(t, RFACTLONG):
    # Calculate short-term real interest rate.
    # RFACTLONG: Long-term interest rate factor
    return -np.log(RFACTLONG[t+1]/RFACTLONG[t]) / 5

def periodueq(t, C, L, elasmu):
    # Calculate one period utility.
    # C: Consumption
    # L: Population and labor
    # elasmu: Elasticity of marginal utility of consumption
    return ((C[t]*1000/L[t])**(1-elasmu) - 1) / (1-elasmu) - 1

def totperiodueq(t, PERIODU, L, RR):
    # Calculate total period utility.
    # PERIODU: One period utility
    # L: Population and labor
    # RR: Social time preference rate with precautionary factor
    return PERIODU[t] * L[t] * RR[t]

def utileq(TOTPERIODU, tstep, scale1, scale2):
    # Calculate overall utility.
    # TOTPERIODU: Total period utility
    # tstep: Time step
    # scale1, scale2: Scaling factors
    return tstep * scale1 * np.sum(TOTPERIODU) + scale2


import numpy as np

# Assuming T is the number of time periods, and all necessary parameters are defined

# Time preference for climate investments and precautionary effect
# rartp: Risk-adjusted rate of time preference
rartp = np.exp(params['prstp'] + params['betaclim'] * params['pi']) - 1

# Initialize arrays for various parameters over time
varpcc = np.zeros(T)    # Variance of per capita consumption
rprecaut = np.zeros(T)  # Precautionary rate of return
RR1 = np.zeros(T)       # Discount factor without precautionary factor
RR = np.zeros(T)        # Discount factor with precautionary factor
L = np.zeros(T)         # Population and labor level
gA = np.zeros(T)        # Growth rate of productivity
aL = np.zeros(T)        # Total factor productivity level
cpricebase = np.zeros(T) # Base carbon price
pbacktime = np.zeros(T) # Backstop price
gsig = np.zeros(T)      # Rate of change of sigma (rate of decarbonization)
sigma = np.zeros(T)     # CO2-emissions output ratio
miuup = np.zeros(T)     # Upper bound on emission control rate

# Loop to calculate variance of per capita consumption, precautionary rate of return, 
# discount factors (with and without precautionary factor)
# Loop over each time period from 0 to T-1
for t in range(T):
    # Calculate the variance of per capita consumption growth for time period t
    # The variance is limited by the value for the 47th time period to avoid overestimation
    # Annual standard deviation of consumption growth
    varpcc[t] = min(params['Siggc1']**2 * 5 * t, params['Siggc1']**2 * 5 * 47) 
    
    # Calculate the precautionary rate of return for time period t
    # This is a negative adjustment due to the risk aversion parameter (elasmu)
    # elasmu is Elasticity of marginal utility of consumption
    rprecaut[t] = -0.5 * varpcc[t] * params['elasmu']**2
    
    # Calculate the discount factor without the precautionary factor for time period t
    # The discount factor is based on the risk-adjusted rate of time preference (rartp) and the time step (tstep)
    RR1[t] = 1 / ((1 + rartp)**(params['tstep'] * t))
    
    # Calculate the discount factor with the precautionary factor for time period t
    # This combines the base discount factor (RR1) and the precautionary adjustment (rprecaut)
    RR[t] = RR1[t] * (1 + rprecaut[t])**(-params['tstep'] * t)


# Population dynamics
# Initial population level
L[0] = params['pop1']
# Loop to calculate population for each time period
for t in range(1, T):
    L[t] = L[t-1] * (params['popasym'] / L[t-1])**params['popadj']

# Productivity
# Growth rate of total factor productivity
gA = params['gA1'] * np.exp(-params['delA'] * 5 * np.arange(T))
# Initial total factor productivity level
aL[0] = params['AL1']
# Loop to calculate total factor productivity for each time period
for t in range(1, T):
    aL[t] = aL[t-1] / (1 - gA[t-1])

# Optimal long-run savings rate
optlrsav = (params['dk'] + 0.004) / (params['dk'] + 0.004 * params['elasmu'] + rartp) * params['gama']

# Carbon price base
cpricebase = params['cprice1'] * (1 + params['gcprice'])**(5 * np.arange(T))

# Backstop price
# Loop to calculate backstop price for each time period
pbacktime = np.where(np.arange(T) <= 7, 
                     params['pback2050'] * np.exp(-5 * 0.01 * (np.arange(T) - 7)),
                     params['pback2050'] * np.exp(-5 * 0.001 * (np.arange(T) - 7)))

# Carbon intensity
# Initial carbon intensity
sig1 = params['e1'] / (params['q1'] * (1 - params['miu1']))
sigma[0] = sig1
# Growth rate of sigma (decarbonization rate)
gsig = np.minimum(params['gsigma1'] * params['delgsig']**np.arange(T), params['asymgsig'])
# Loop to calculate sigma for each time period
for t in range(1, T):
    sigma[t] = sigma[t-1] * np.exp(5 * gsig[t-1])

# Emissions limits
# Initial limits on emission control rate
miuup[0] = 0.05
miuup[1] = 0.10
# Loop to set upper bounds on emission control rate for each time period
miuup[2:] = params['delmiumax'] * np.arange(2, T)
miuup[8:] = np.minimum(miuup[8:], 0.85 + 0.05 * np.arange(T-8))
miuup[11:] = np.minimum(miuup[11:], params['limmiu2070'])
miuup[20:] = np.minimum(miuup[20:], params['limmiu2120'])
miuup[37:] = np.minimum(miuup[37:], params['limmiu2200'])
miuup[57:] = np.minimum(miuup[57:], params['limmiu2300'])
