import os
import sys
import yaml
import pickle as pickle
import numpy as np
import xarray as xr
from src.inversion_scripts.utils import load_obj, calculate_superobservation_error

from functools import partial
print = partial(print, flush = True)

def merge_partial_k(satdat_dir, lat_bounds, lon_bounds, obs_err, precomp_K):
    """
    Description:
        This function is used to generate the full jacobian matrix (K), observations (y),
        background vector (y_bkgd), and observational error (So) for the lognormal inversion.

        The normal inversion script of the IMI reads in the jacobian matrix, observations,
        and observational error piece by piece in order to avoid loading the full jacobian
        matrix into memory (which can be quite large). The lognormal inversion script
        requires the full form of these variables to iteratively solve for the posterior.
        Here we load in the partial jacobian matrices and observations from each satellite
        data file and concatenate them into the full jacobian matrix and observation vector,
        for use in the lognormal inversion script. We also calculate the observational error
        and background vector.

    Parameters:
        satdat_dir    [str]: path to directory containing satellite data files
        lat_bounds   [list]: list of latitude bounds to consider each bound is a tuple
        lon_bounds   [list]: list of longitude bounds to consider each bound is a tuple
        obs_err     [float]: default observational error value
        precomp_K [boolean]: whether or not to use precomputed jacobian matrices
    """
    # Get observed and GEOS-Chem-simulated TROPOMI columns
    files = [f for f in np.sort(os.listdir(satdat_dir)) if "TROPOMI" in f]
    # lat = np.array([])
    # lon = np.array([])
    tropomi = np.array([])
    geos_prior = np.array([])
    so = np.array([])

    print('Merging K')
    for i, f in enumerate(files):
        print(f'\r{f}',end='')
        obs_error = obs_err  # reset obs_error to original value
        # Get paths
        pth = os.path.join(satdat_dir, f)
        # Get same file from bc folder
        # Load TROPOMI/GEOS-Chem and Jacobian matrix data from the .pkl file
        obj = load_obj(pth)
        # If there aren't any TROPOMI observations on this day, skip
        if obj["obs_GC"].shape[0] == 0:
            continue
        # Otherwise, grab the TROPOMI/GEOS-Chem data
        obs_GC = obj["obs_GC"]
        # Only consider data within latitude and longitude bounds
        ind = np.where(
            (obs_GC[:, 2] >= lon_bounds[0])
            & (obs_GC[:, 2] <= lon_bounds[1])
            & (obs_GC[:, 3] >= lat_bounds[0])
            & (obs_GC[:, 3] <= lat_bounds[1])
        )
        if len(ind[0]) == 0:  # Skip if no data in bounds
            continue
        obs_GC = obs_GC[ind[0], :]  # TROPOMI and GEOS-Chem data within bounds

        # concatenate full jacobian, obs, so, and prior
        tropomi = np.concatenate((tropomi, obs_GC[:, 0]))
        geos_prior = np.concatenate((geos_prior, obs_GC[:, 1]))

        # read K from reference dir if precomp_K is true
        if precomp_K:
            pass
            print('Using precomputed K so not merging')
            # Get Jacobian from reference inversion
            ## fi_ref = pth.replace("data_converted", "data_converted_reference")
            ## dat_ref = load_obj(fi_ref)
            ## K_temp = dat_ref["K"][ind[0]]
        else:
            K_temp = obj["K"][ind[0]]

        # append partial Ks to build full jacobian
        if not precomp_K:
            if i == 0:
                K = K_temp
            else:
                K = np.append(K, K_temp, axis=0)

        # calculate superobservation error
        s_superO_1 = calculate_superobservation_error(obs_error, 1)
        s_superO_p = np.array(
            [
                calculate_superobservation_error(obs_error, p) if p >= 1 else s_superO_1
                for p in obs_GC[:, 4]
            ]
        )
        # scale error variance by gP value following Chen et al. 2023
        gP = s_superO_p**2 / s_superO_1**2
        obs_error = obs_error ** 2
        obs_error = gP * obs_error

        # check to make sure obs_err isn't negative, set 1 as default value
        obs_error = [obs if obs > 0 else 1 for obs in obs_error]
        so = np.concatenate((so, obs_error))

    if not precomp_K:
        gc_ch4_prior = np.asmatrix(geos_prior)

    if not precomp_K:
        obs_tropomi = np.asmatrix(tropomi)

    print('Done merging K')

    if precomp_K:
        return None, None, None, so
    else:
        return gc_ch4_prior, obs_tropomi, K, so
    


if __name__ == "__main__":
    # read in arguments
    satdat_dir = sys.argv[1]
    state_vector_filepath = sys.argv[2]
    obs_error = float(sys.argv[3])
    precomputed_jacobian = sys.argv[4] == "true"
    config_path = sys.argv[5]

    # directory containing partial K matrices
    # Get observed and GEOS-Chem-simulated TROPOMI columns
    files = np.sort(os.listdir(satdat_dir))
    files = [f for f in files if "TROPOMI" in f]

    state_vector = xr.load_dataset(state_vector_filepath)
    state_vector_labels = state_vector["StateVector"]
    lon_bounds = [np.min(state_vector.lon.values), np.max(state_vector.lon.values)]
    lat_bounds = [np.min(state_vector.lat.values), np.max(state_vector.lat.values)]

    # jde ensemble
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if 'EnsembleDirName' in config.keys():
        ens_dir = config['EnsembleDirName']
    else:
        ense_dir = 'base_inv'

    # Paths to GEOS/satellite data
    gc_ch4_bkgd, obs_tropomi, jacobian_K, so = merge_partial_k(
        satdat_dir, lat_bounds, lon_bounds, obs_error, precomputed_jacobian
    )

    if not precomputed_jacobian:
        # only save if computing jacobian for the first time
        np.savez("full_jacobian_K.npz", K=jacobian_K)
        np.savez("obs_ch4_tropomi.npz", obs_tropomi=obs_tropomi)
        np.savez("gc_ch4_bkgd.npz", gc_ch4_bkgd=gc_ch4_bkgd)

    # always saved out
    os.makedirs(ens_dir, exist_ok = True)
    np.savez(f'{ens_dir}/so_super.npz', so=so)



