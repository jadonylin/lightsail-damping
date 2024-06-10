# IMPORTS ###########################################################################################################################################################################
import grcwa
grcwa.set_backend('autograd')
import autograd.numpy as np



# REFLECTION/TRANSMISSION ###########################################################################################################################################################################
def RT(params
       , struc_geom: list=[1.,1.]
       , Nx: float=100
       , nG: int=20
       , eps_substrate: float=-1e6
       , inc_angle: float=0.
       , Qabs: float=1e5):
    """
    Calculate reflection/transmission at a given incident angle for a given geometry.

    Parameters
    ----------
    params         :   [laser wavelength, grating depth, eps for every grid in the unit cell] e.g. [0.7, 0.5, 1,9,8,9,1]
    struc_geom     :   [grating period, substrate depth]
    Nx             :   Number of grid points in the unit cell
    nG             :   Number of Fourier components
    eps_substrate  :   Permittiivty of the substrate
    inc_angle      :   Incoming plane wave angle in radians
    Qabs           :   Relaxation parameter
    """

    # Extract parameters
    inc_wavelength = params[0]
    grating_depth = params[1]
    grating_grid = params[2:]
    
    # Grating
    dy = 1e-4 # Small slice in the y-direction to simulate 2D grating
    grating_period = struc_geom[0] # grating period
    L1 = [grating_period,0]
    L2 = [0,dy]

    freq = 1/inc_wavelength # freq = 1/wavelength when c = 1
    freqcmp = freq*(1+1j/2/Qabs)

    # Incoming wave
    theta = inc_angle # radians
    phi = 0.

    # setup RCWA
    obj = grcwa.obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)


    ## CREATE LAYERS ##
    # Grid resolution
    Ny = 1 # number of grid points along y

    # Layer depth
    eps_vacuum = 1
    vacuum_depth = grating_period
    substrate_depth = struc_geom[1] # Substrate thickness/height

    # Construct layers
    obj.Add_LayerUniform(vacuum_depth,eps_vacuum)
    obj.Add_LayerGrid(grating_depth,Nx,Ny)
    obj.Add_LayerUniform(substrate_depth,eps_substrate)
    obj.Add_LayerUniform(vacuum_depth,eps_vacuum)
    obj.Init_Setup()

    # Construct patterns
    obj.GridLayer_geteps(grating_grid)


    ## INCIDENT WAVE ##
    planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

    # solve for R and T
    R_byorder,T_byorder = obj.RT_Solve(normalize=1, byorder=1)
    Fourier_orders = obj.G

    Rs = []
    Ts = []
    RT_orders = [-1,0,1]
    # IMPORTANT: have to use append method to a list rather than index assignment
    # Else, autograd will throw a TypeError with float() argument being an ArrayBox
    for j in range(0,len(RT_orders)):
        Rs.append( np.sum(R_byorder[ Fourier_orders[:,0]==RT_orders[j] ]) )
        Ts.append( np.sum(T_byorder[ Fourier_orders[:,0]==RT_orders[j] ]) )

    return Rs,Ts