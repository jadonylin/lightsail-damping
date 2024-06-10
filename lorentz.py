# IMPORTS ################################################################################################################
import numpy as np



# FUNCTIONS ################################################################################################################
# Lorentz gamma-factor
def gamma_ND(v):
    """
    Calculate the Lorentz gamma factor with an input non-dimensionalised speed/velocity.

    Parameters
    ----------
    v     : ND speed or two/three-velocity or list of two/three-velocities

    Returns
    -------
    gamma : Lorentz gamma factor
    """
    if not isinstance(v,(list,np.ndarray)):
        v = [v]
    v = np.array(v)
    
    if any(isinstance(i, np.ndarray) for i in v):
        vnorm = np.linalg.norm(v,axis=1)
    else:
        vnorm = np.linalg.norm(v)
    
    gamma = 1/np.sqrt(1-np.power(vnorm,2))
    return gamma

# Relativistic Doppler-factor
def D1_ND(v):
    """
    Calculate the D_1 Doppler factor with an input non-dimensionalised velocity.

    Parameters
    ----------
    v  : ND two/three-velocity of the moving frame or list of two/three-velocities

    Returns
    -------
    D1 : Doppler factor
    """
    if not isinstance(v,(list,np.ndarray)):
        v = [v]
    v = np.array(v)
    
    if any(isinstance(i, np.ndarray) for i in v):
        vx = np.array([i[0] for i in v])
    else:
        vx = np.array(v[0])

    D1 = gamma_ND(v)*(1-vx)
    return D1