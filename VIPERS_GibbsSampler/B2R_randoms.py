import mask
import numpy as np
from scipy import interpolate

def sample_redshift(zdist, zbin, n):
    z_min, z_max = zbin

    z = zdist[:,0]
    p = zdist[:,1] # p(z) NOT THE CUMULATIVE

    func = interpolate.interp1d(z, p, bounds_error=False, fill_value=(0, 0))

    zout = np.random.uniform(z_min, z_max, n)
    zweight = func(zout)

    zweight /= np.max(zweight)

    return zout, zweight

def sample_sky(mask, density):

    ra, dec, area = mask.random_sample(density=density)
    print("sampled", len(ra))
    print("area", area)
    return ra, dec

def randoms(paths,tag, dens_randoms,zbin,z_mock,mocks=False):

    M = mask.Mask()

    if mocks==False:
        zdist_arg = np.loadtxt(paths['path_zdist'])
        #print('z_distVIPERS from RANDOMS is ',zdist_arg)
    elif mocks==True:
        zdist_arg = z_mock
        #print('z_mockMOCK from RANDOMS is ',zdist_arg)

    if tag == 'w1':

        M.add_mask_file(paths['path_spec_mask_w1'])
        M.add_mask_file(paths['path_phot_mask_w1'], holes=True)

    elif tag == 'w4':
        M.add_mask_file(paths['path_spec_mask_w4'])
        M.add_mask_file(paths['path_phot_mask_w4'], holes=True)

    ra, dec = sample_sky(M, dens_randoms)

    zout, zweight = sample_redshift(zdist_arg, zbin, len(ra))

    assert len(zout) == len(ra)

    return ra, dec, zout, zweight
