# IMPORTS ################################################################################################################################################
import os
## Limit number of numpy threads (MUST GO BEFORE NUMPY IMPORT) ##
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import numpy as np
import nlopt
from numpy import *

from FOM import mean_FD_RG
from lorentz import D1_ND

import time
from datetime import datetime, timedelta

from multiprocessing import Pool

import pickle



# GLOBAL OPTIMISATION ###########################################################################
## FIXED PARAMETERS ##
grating_period = 1 
substrate_depth = 1*grating_period 
struc_geom = [grating_period,substrate_depth]

Nx = 30 # Number of grid points
nG = 20 # Number of Fourier components
n_avg = 20 # Number of wavelength points to average F_D 

eps_vac = 1 # Minimum allowed grid permittivity
eps_max = 3.5**2 # Maximum allowed grid permittivity
eps_substrate = -1e6 # Substrate permittivity

vf = [0.2,0] # final velocity
Dinv = 1/D1_ND(vf)-1
perc_Dshift = 100*Dinv
lambda_range_max = np.round(0.998/(1+Dinv),3) # stay away from the cutoff divergence

return_grad = True # Return Fdmp and gradient of Fdmp
Qabs = 1e5 # Large but finite relaxation to avoid singular matrix

fixed_args = (struc_geom, Nx, nG, eps_substrate, perc_Dshift, n_avg, return_grad, Qabs)


## OPTIMISATION ##
# Set up NLOPT
seed = 20240305 # LDS seed
sampling = 'sobol' # 'sobol' or 'random'
n_sample_exp = 3
n_sample = 2**n_sample_exp
ndof = Nx+2

# Parameter bounds
lambda_min = 0.5*grating_period
lambda_max = lambda_range_max*grating_period
h1_min = 0*grating_period # h1 = grating depth
h1_max = 1*grating_period

# Stopping criteria
xtol_rel = 1e-4 # local
ftol_rel = 1e-8 # local
num_proc = 36 # number of processors to run parallel optimisation
maxfev = 32000 # global



# OBJECTIVE FUNCTION ###########################################################################
fun = lambda params: mean_FD_RG(params,*fixed_args)

# Set up objective function for nlopt
def fun_nlopt(params,gradn):
    if gradn.size > 0:
        # Even for gradient methods, in some calls gradn will be empty []
        gradn[:] = fun(params)[1]
    y = fun(params)[0]
    return y



# RECORDING RESULTS ###########################################################################
## Converting non-h1 parameter dicts to strings ##
# Fixed parameters
fixed_par_dict = {'struc_geom': struc_geom
                  , 'Nx': Nx
                  , 'nG': nG
                  , 'eps_substrate': f"{eps_substrate:.0E}"
                  , 'perc_Dshift': perc_Dshift
                  , 'n_avg': n_avg
                  , 'Qabs': Qabs}
fixed_par_line = str(fixed_par_dict)

# Bounded parameters
bounds_dict = {'lambda_min': lambda_min, 'lambda_max': lambda_max
               , 'eps_min': eps_vac, 'eps_max': eps_max}
bounds_line = str(bounds_dict)

# Optimiser options
sampling_dict = {'Sampling method': sampling, 'n_sample': f'2E+{n_sample_exp}', 'seed': seed}
sampling_line = str(sampling_dict)
LO_dict = {'xtol_rel': f"{xtol_rel:.1E}", 'ftol_rel': f"{ftol_rel:.1E}"}
LO_line = str(LO_dict)
GO_dict = {'number of cores': num_proc, 'maxfev per core': maxfev}
GO_line = str(GO_dict)

# Date and time at beginning of run
time_at_execution = str(datetime.now())

# Strings to write to file
lines_to_file = ["\n\n------------------------------------------------------------------------------------------------------------------------------------\n"
                , f"Date & time      | {time_at_execution}\n"
                ,  "\n"
                , f"Fixed parameters | {fixed_par_line}\n"
                , f"Non-h1 bounds    | {bounds_line}\n"
                ,  "\n"
                , f"Sampling options | {sampling_line}\n"
                , f"LO options       | {LO_line}\n"
                , f"GO options       | {GO_line}\n"
                , "------------------------------------------------------------------------------------------------------------------------------------\n"]


## Writing to file ##
# The if __name__ is necessary so that the non-h1 parameters are printed only once
txt_fname = f'./final_data/optimisation.txt'
if __name__ == "__main__":
    with open(txt_fname, "a") as result_file:
        result_file.writelines(lines_to_file)



# FUNCTION TO RETURN OPTIMISATION RESULT ###########################################################################
# To speed up computation, run optimisation in parallel over num_proc cores
def mean_optimise(h1_min, h1_max):    
    # Choose GO and LO
    if sampling == 'sobol':
        opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
    elif sampling == 'random':
        opt = nlopt.opt(nlopt.G_MLSL, ndof)
    else:
        opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
    local_opt = nlopt.opt(nlopt.LD_MMA, ndof)

    # Set LDS seed
    nlopt.srand(seed) 

    # Initial guess
    init = [0.7, random.uniform(h1_min,h1_max)]\
        + [1]*5 + [eps_max]*7 + [1]*5 + [eps_max]*8 + [1]*5

    # Set options for optimiser
    opt.set_population(n_sample) # initial sampling points

    # Set options for local optimiser
    local_opt.set_xtol_rel(xtol_rel)
    local_opt.set_ftol_rel(ftol_rel)
    opt.set_local_optimizer(local_opt)

    # Set objective function
    opt.set_max_objective(fun_nlopt)
    opt.set_maxeval(maxfev)
    
    lb = [lambda_min, h1_min] + [eps_vac]*Nx 
    ub = [lambda_max, h1_max] + [eps_max]*Nx
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # Time at start of optimisation
    start_time = time.monotonic()

    opt_params = opt.optimize(init)
    result = opt.last_optimize_result()
    
    # Time at end of optimisation
    end_time = time.monotonic()
    run_time = str(timedelta(seconds=end_time - start_time))
    

    ## Converting result dicts to strings ##
    # h1 bounds #
    h1_dict = {'h1_min': h1_min, 'h1_max': h1_max}
    h1_line = str(h1_dict)

    # Results
    success_line = str({'success': result})
    fmax = fun(opt_params)[0]
    fun_line = str({'fun': fmax})
    params_line = str({'params': opt_params})

    # Strings to write to file
    lines_to_file = ["------------------------------------------------------------------------------------------------------------------------------------\n"
                    , f"h1 bounds          | {h1_line}\n"
                    , f"Initial guess      | {init}\n"
                    , f"Runtime            | {run_time}\n"
                    , f"Result (success)   | {success_line}\n"
                    , f"      Function max | {fun_line}\n"
                    , f"      Maximiser    | {params_line}\n"
                    , "------------------------------------------------------------------------------------------------------------------------------------\n"]


    ## Writing to file ##
    with open(txt_fname, "a") as result_file:
        result_file.writelines(lines_to_file)
    
    return (fmax,opt_params)



# RUN OPTIMISATION IN PARALLEL ###########################################################################
# Create list of range tuples
h1_bounds = []
h1s = np.linspace(h1_min,h1_max,num_proc+1)
for p in range(0,num_proc):
    interval = (h1s[p], h1s[p+1])
    h1_bounds.append(interval)

# Run parallel optimisation
pkl_fname = f'./final_data/optimisation.pkl'
if __name__ == '__main__':
    with Pool(processes=num_proc) as pool:
        res = pool.starmap(mean_optimise, h1_bounds)

        # Save result
        with open(pkl_fname, 'wb') as data_file:
            pickle.dump(res, data_file)