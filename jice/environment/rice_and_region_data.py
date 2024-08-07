import chex 

@chex.dataclass(frozen=True)
class RegionRiceConstants:
    """ 
        No default values, fill with region yaml data 
        each property should be an array of shape (num_regions,)
    """
    xA_0: chex.Array
    xK_0: chex.Array
    xL_0: chex.Array
    xL_a: chex.Array
    xa_1: chex.Array
    xa_2: chex.Array
    xa_3: chex.Array
    xdelta_A: chex.Array
    xg_A: chex.Array
    xgamma: chex.Array
    xl_g: chex.Array
    xmitigation_0: chex.Array
    xsaving_0: chex.Array
    xsigma_0: chex.Array
    xtax: chex.Array

    ximport: chex.Array # will be 2d (num_regions, num_regions)
    # '1': 0.3999044818
    # '2': 0.0
    # '3': 0.3968398882
    xexport: chex.Array

@chex.dataclass(frozen=True)
class DiceEnvConstants:
    xt_0: int = 2015 # starting year of the whole model
    xDelta: int = 5 # the time interval (year)
    xN: int = 20 # total time steps
    
    # Climate diffusion parameters
    xPhi_T: tuple = ((0.8718, 0.0088), (0.025, 0.975))
    xB_T: tuple = (0.1005, 0)
    # xB_T: [0.03, 0]

    # Carbon cycle diffusion parameters (the zeta matrix in the paper)
    xPhi_M: tuple = ((0.88, 0.196, 0), (0.12, 0.797, 0.001465), (0, 0.007, 0.99853488))
    # xB_M: tuple = (0.2727272727272727, 0, 0) # 12/44
    xB_M: tuple = (1.36388, 0, 0) # 12/44
    xeta: float = 3.6813 #?? I don't find where it's used

    xM_AT_1750: int = 588 # atmospheric mass of carbon in the year of 1750
    xf_0: float = 0.5 # in Eq 3  param to effect of greenhouse gases other than carbon dioxide
    xf_1: int = 1 # in Eq 3  param to effect of greenhouse gases other than carbon dioxide
    xt_f: int = 20 # in Eq 3  time step param to effect of greenhouse gases other than carbon dioxide
    xE_L0: float = 2.6 # 2.6 # in Eq 4 param to the emissions due to land use changes
    xdelta_EL: float = 0.001 # 0.115 # 0.115 # in Eq 4 param to the emissions due to land use changes

    xM_AT_0: int = 851 # in CAP the atmospheric mass of carbon in the year t
    xM_UP_0: int = 460 # in CAP the atmospheric upper bound of mass of carbon in the year t
    xM_LO_0: int = 1740 # in CAP the atmospheric lower bound of mass of carbon in the year t
    xe_0: float = 35.85 # in EI define the initial simga_0: e0/(q0(1-mu0))
    xq_0: float = 105.5 # in EI define the initial simga_0: e0/(q0(1-mu0))
    xmu_0: float = 0.03 # in EI define the initial simga_0: e0/(q0(1-mu0))

    # From Python implementation PyDICE
    xF_2x: float = 3.6813 # 3.6813 # Forcing that doubles equilibrium carbon.
    xT_2x: float = 3.1 # 3.1 # Equilibrium temperature increase at double carbon eq.

@chex.dataclass(frozen=True)
class RiceEnvConstants:
    """
        Commented out variables are region specific and part of the region yamls
    """
    # xgamma: float = 0.3 # in CAP Eq 5 the capital elasticty
    xtheta_2: float = 2.6 # in CAP Eq 6
    # xa_1: float = 0
    # xa_2: float = 0.00236 # in CAP Eq 6   # DFAIR: 0.003467
    # xa_3: float = 2 # in CAP Eq 6
    xdelta_K: float = 0.1 # in CAP Eq 9 param discribe the depreciate of the capital
    xalpha: float = 1.45 # Utility function param
    xrho: float = 0.015 # discount factor of the utility
    # xL_0: float = 7403 # in POP population at the staring point
    # xL_a: float = 11500 # in POP the expected population at convergence
    # xl_g: float = 0.134 # in POP control the rate to converge

    # xmitigation_0: float = 0.0
    # xsaving_0: float = 0.25
    # xtax: float = 0.0

    # xexport: float = 0.0
    # ximport: float =
    # "1": float = 0.0
    # "2": float = 0.0
    # "3": float = 0.0

    # xA_0: float = 5.115 # in TFP technology at starting point
    # xg_A: float = 0.076 # in TFP control the rate of increasing of tech larger->faster
    # xdelta_A: float = 0.005 # in TFP control the rate of increasing of tech smaller->faster
    # xsigma_0: float = 0.3503 # e0/(q0(1-mu0)) in EI emission intensity at the starting point
    xg_sigma: float = 0.0025 # 0.0152 # 0.0025 in EI control the rate of mitigation larger->reduce more emission
    xdelta_sigma: float = 0.1 # 0.01 in EI control the rate of mitigation larger->reduce less emission
    xp_b: float = 550 # 550 # in Eq 2 (estimate of the cost of mitigation)  represents the price of a backstop technology that can remove carbon dioxide from the atmosphere
    xdelta_pb: float = 0.001 # 0.025 # in Eq 2 control the how the cost of mitigation change through time larger->cost less as time goes by
    xscale_1: float = 0.030245527 # in Eq 29 Nordhaus scaled cost function param
    xscale_2: float = 10993.704 # in Eq 29 Nordhaus scaled cost function param

    xT_AT_0: float = 0.85 # in CAP a part of damage function initial condition
    xT_LO_0: float = 0.0068 # in CAP a part of damage function initial condition
    # xK_0: float = 223 # in CAP initial condition for capital

    # FAIR model
    xM_R1_0: float = 127.159
    xM_R2_0: float = 93.313
    xM_R3_0: float = 37.840
    xM_R4_0: float = 7.721

    xM_a0: float = 0.217
    xM_a1: float = 0.224
    xM_a2: float = 0.282
    xM_a3: float = 0.276

    xM_t0: float = 10000000
    xM_t1: float = 394.4
    xM_t2: float = 36.54
    xM_t3: float = 4.304

    xEcum_0: float = 400
    xEcumL_0: float = 197
    xEInd_0: float = 35.85
    xEL_0: float = 2.6
    xalpha_0: float = 1

    irf0: float = 35  # Fair: 35, DFAIR: 32.4
    irC: float = 0.019
    irT: float = 4.165

    # DFAIR model
    xT_LO_tq: float = 0.324  # Thermal equilibration parameter for box 1 (m^2 per KW), 0.33 in FaIR
    xT_UO_tq: float = 0.44  # Thermal equilibration parameter for box 2 (m^2 per KW), 0.41 in FaIR

    xT_LO_rt: float= 236  #  Thermal response timescale for box 1 (lower/deep ocean) (year), 239 in FaIR 
    xT_UO_rt: float= 4.07  # Thermal response timescale for box 2 (upper ocean) (year), 4.1 in FaIR
    
    xT_LO_0: float = 0.11541  # Initial temperature box 1 change, Mean of DFaIR (2020): 0.1477 (C from 1765) and Fair-DICE implementation (2010): 0.08312
    xT_UO_0: float = 0.934127  # Initial temperature box 2 change, Mean of DFaIR (2020): 1.099454 (C from 1765) and Fair-DICE implementation for (2010): 0.7688

    # Temperature model
    xT_AT_0_FaIR: float = 1.243
    xT_LO_0_FaIR: float = 0.324
    xT_1: float = 7.3
    #  xT_2y: float = 3.6813/3.1 #forcoings of eq co2 doubling/ETS
    xT_3: float = 0.73
    xT_4: float = 106
    xT_kappa: float = 3.6813

    # bounds
    b_savings_min: float = 0.0
    b_savings_max: float = 1.0
    b_mitigation_rate_min: float = 0.0
    b_mitigation_rate_max: float = 1.0
    b_export_limit_min: float = 0.0
    b_export_limit_max: float = 1.0
    b_import_bids_min: float = 0.0
    b_import_bids_max: float = 1.0
    b_import_tariffs_min: float = 0.0
    b_import_tariffs_max: float = 1.0
    b_proposal_min: float = 0.0
    b_proposal_max: float = 1.0
    b_proposal_decisions_min: float = 0.0
    b_proposal_decisions_max: float = 1.0


