# IMPORTS ########################################################################################################################
import adaptive as adp
import autograd.numpy as np
from autograd import grad
from effderivs import RT



# RADIATION PRESSURE CROSS-SECTION ########################################################################################################################
def Qpr1_RG(params
            , struc_geom: list=[1.,1.]
            , Nx: float=100
            , nG: int=20
            , eps_substrate: float=-1e6
            , Qabs: float=1e5) -> float:
    """
    Calculate Qpr1 for the reflection grating at zero incident angle

    Parameters
    ----------
    params        :   [laser wavelength, grating depth, eps for every slice in the grating] e.g. [0.7, 0.5, 1,9,8,9,1]
    struc_geom    :   [grating period, substrate depth]
    Nx            :   Number of grid points in the unit cell
    nG            :   Number of Fourier components
    eps_substrate :   Permittivity of the mirror substrate
    Qabs          :   Relaxation parameter
    """
    
    inc_angle = 0.

    # Calculate reflection
    RT_orders = RT(params, struc_geom, Nx, nG, eps_substrate, inc_angle, Qabs)

    rNeg1, r0, r1 = RT_orders[0]
    tNeg1, t0, t1 = RT_orders[1]
    
    # Calculate Qpr1
    Qpr1 = 2*r0 + (rNeg1 + r1)*(1 + np.sqrt(1-params[0]**2)) + (tNeg1 + t1)*(1 - np.sqrt(1-params[0]**2))

    return Qpr1



# DAMPING FOM ########################################################################################################################
def FD_RG(params
          , struc_geom: list=[1.,1.]
          , Nx: float=100
          , nG: int=20
          , eps_substrate: float=-1e6
          , Qabs: float=1e5) -> float:
    """
    Calculate the damping figure of merit for the reflection grating at a specific wavelength.

    Parameters
    ----------
    params        :   [laser wavelength, grating depth, eps for every slice in the grating] e.g. [0.7, 0.5, 1,9,8,9,1]
    struc_geom    :   [grating period, substrate depth]
    Nx            :   Number of grid points in the unit cell
    nG            :   Number of Fourier components
    eps_substrate :   Permittivity of the mirror substrate
    Qabs          :   Relaxation parameter
    """

    # Calculate Qpr1
    Qpr1 = Qpr1_RG(params, struc_geom, Nx, nG, eps_substrate, Qabs)
    
    # Calculate damping term PDQpr2 + Qpr1
    Delta_theta = 1e-4
    R_plustheta,T_plustheta = RT(params, struc_geom, Nx, nG, eps_substrate, Delta_theta, Qabs)
    rNeg1_plustheta, r0_plustheta, r1_plustheta = R_plustheta
    tNeg1_plustheta, t0_plustheta, t1_plustheta = T_plustheta

    R_negtheta,T_negtheta = RT(params, struc_geom, Nx, nG, eps_substrate, -Delta_theta, Qabs)
    rNeg1_negtheta, r0_negtheta, r1_negtheta = R_negtheta
    tNeg1_negtheta, t0_negtheta, t1_negtheta = T_negtheta

    PDrNeg1 = (rNeg1_plustheta - rNeg1_negtheta)/(2*Delta_theta)
    PDtNeg1 = (tNeg1_plustheta - tNeg1_negtheta)/(2*Delta_theta)
    PDr1 = (r1_plustheta - r1_negtheta)/(2*Delta_theta)
    PDt1 = (t1_plustheta - t1_negtheta)/(2*Delta_theta)

    damp = params[0]*(PDrNeg1 + PDtNeg1 - PDr1 - PDt1)
    FD_RG = damp/Qpr1

    return FD_RG


def mean_FD_RG(params
               , struc_geom: list=[1.,1.]
               , Nx: float=100
               , nG: int=20
               , eps_substrate: float=-1e6
               , perc_Dshift: float=5
               , n_avg: int=20
               , return_grad: bool=True
               , Qabs: float=1e5) -> float:
    """
    Calculate the mean of FD for the reflection grating over a given wavelength range.

    Parameters
    ----------
    params        :   [laser wavelength, grating depth, eps for every slice in the grating] e.g. [0.7, 0.5, 1,9,8,9,1]
    struc_geom    :   [grating period, substrate depth]
    Nx            :   Number of grid points in the unit cell
    nG            :   Number of Fourier components
    eps_substrate :   Permittivity of the mirror substrate
    perc_Dshift   :   Percentage Doppler shift of wavelength
    n_avg         :   Number of points to average the FOM over the given range
    return_grad   :   Return gradient along with Fbal
    Qabs          :   Relaxation parameter
    """

    lambda_min = params[0]
    lambda_max = lambda_min*(1 + perc_Dshift/100.0)    
    wavelength_range = (lambda_min, lambda_max)

    
    # Define a one argument function to pass to learner
    params_tmp = [*params] # need to make copy to avoid modifying params input
    del params_tmp[0]
    
    def onearg_FD(lam):
        params_tmp2 = [lam] + params_tmp
        return FD_RG(params_tmp2, struc_geom, Nx, nG, eps_substrate, Qabs)
    
    # Adaptive sample FD
    FD_learner = adp.Learner1D(onearg_FD, bounds=wavelength_range)
    FD_runner = adp.runner.simple(FD_learner, npoints_goal=int(n_avg))
    
    FD_data = FD_learner.to_numpy()
    FD_wavelengths = FD_data[:,0]
    FDs = FD_data[:,1]

    mean_FD = np.trapz(FDs,FD_wavelengths)/(lambda_max-lambda_min)

    if return_grad:
        # Define a one argument function to pass to learner
        FD_grad = grad(FD_RG)
        def onearg_FD_grad(lam):
            params_tmp2 = [lam] + params_tmp
            return FD_grad(params_tmp2, struc_geom, Nx, nG, eps_substrate, Qabs)
        
        # Adaptive sample FD_grad
        FD_grad_learner = adp.Learner1D(onearg_FD_grad, bounds=wavelength_range)
        FD_grad_runner = adp.runner.simple(FD_grad_learner, npoints_goal=int(n_avg))

        FD_grad_data = FD_grad_learner.to_numpy()
        FD_grad_wavelengths = FD_grad_data[:,0]
        FD_grads = FD_grad_data[:,1:]
        
        mean_FD_grad = np.trapz(FD_grads,FD_grad_wavelengths,axis=0)/(lambda_max-lambda_min)
        
        return [mean_FD,mean_FD_grad]
    else:
        return mean_FD