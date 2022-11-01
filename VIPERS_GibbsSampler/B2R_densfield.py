# LOADING LIBRARIES
import pylab
import scipy
import sphere
import logging
import B2R_randoms
import numpy as np
from scipy import ndimage
from astropy.io import fits
from matplotlib import pyplot as plt
logging.basicConfig(level=logging.INFO)
from astropy.cosmology import FlatLambdaCDM as cosmo


### B2R IMPROVEMENTS:
# Box function that calculates the box size depending on the prior
# Redshift selection of the galaxies
# Mask for TSR and SSR
# Smarter use of the data variables
# Omega matter as a parameter for the density field (new comoving distance on each WF iteration)
def box(z, prior, cell_size):
    """Generates a 3d array in comoving coordinates"""
    z_min, z_max   = z
    Om_inf, Om_sup = prior

    C_inf = cosmo(H0=100,Om0=Om_inf)
    C_sup = cosmo(H0=100,Om0=Om_sup)

    dec_vipers = 0.0316
    # with w1 ---> np.pi*(vipers[:,1].max() - vipers[:,1].min())/180
    ra_vipers  = 0.1504
    # with w1 ---> np.pi*(vipers[:,0].max() - vipers[:,0].min())/180

    x = C_inf.comoving_distance(z_max).value - C_inf.comoving_distance(z_min).value + 5
    z = C_inf.comoving_transverse_distance(z_max)* dec_vipers/2
    y = C_inf.comoving_transverse_distance(z_max)* ra_vipers /2
    bins  = (np.arange(0,x,cell_size),np.arange(-z.value,z.value,cell_size),np.arange(-y.value,y.value,cell_size))
    return bins

def zbin(z_min, z_max, z_array):
    """Selects the galaxies inside a redshift interval"""
    sel_min  = z_array >= z_min
    sel_max  = z_array <= z_max
    z_bin = np.all([sel_min, sel_max],axis=0)
    return z_bin

def cell_weight(alpha, dec, mask_wt, edges_alpha, edges_dec): # Put the edges of the bins as input and compute the step from it
    """Applies the TSR/SSR mask to a given number of ra, dec values"""
    step_alpha  = edges_alpha[1] -edges_alpha[0]
    step_dec    = edges_dec[1]   -edges_dec[0]

    alpha_0 = edges_alpha[0]
    dec_0   = edges_dec[0]

    p_alpha  = ((alpha-alpha_0)/step_alpha).astype(int)
    p_dec    = ((dec-dec_0)/step_dec).astype(int)

    weight   = mask_wt[p_alpha,p_dec]

    return weight



def densfield(config, paths, tag, z_bin, prior, cell_size, realizations, dens_randoms, mocknum, Om0=0.31361, com_random=False, com_vipers=False,mocks=False):

    threshold = config['threshold']
    z_min, z_max   = z_bin
    Om_inf, Om_sup = prior

    C = cosmo(H0=100,Om0=Om0)
    x0    = C.comoving_distance(z_min).value - 4

    # VIPERS data
    vipers  = np.loadtxt(paths['path_vipers_'+tag]) # W1 = 49220 galaxies     W4 = 24353 galaxies

    center =  (vipers[:,0].max() + vipers[:,0].min())/2
    delta  = -(vipers[:,1].max() + vipers[:,1].min())/2
    z_mock = []
    if mocks==True:
        vipers, z_mock  = mockfield(paths, tag, mocknum)
        print('z_mock from densfield is ',z_mock)
    ##### Selecting redshift #####
    sel_z  = zbin(z_min, z_max, vipers[:,2])
    vipers = vipers[sel_z]

    ##### VIPERS Variables #####
    alpha= vipers[:,0] - center
    dec  = vipers[:,1]
    z    = vipers[:,2]
    if mocks==False:
        tsr  = vipers[:,3]
        ssr  = vipers[:,4]
    elif mocks==True:
        tsr  = np.ones(len(z))
        ssr  = np.ones(len(z))

    ##### Bins #####
    bins = box(z_bin, prior, cell_size)                 # Box size dependent on the prior. Calculated only once.

    ##### Weight mask #####
    """Generating a TSR/SSR mask from VIPERS data"""
    print ("building weight grid")

    c_wt    = tsr * ssr

    edges_alpha  = np.arange(alpha.min()-0.1, alpha.max()+0.1, 5./60)
    edges_dec    = np.arange(dec.min()-0.1, dec.max()+0.1, 5./60)

    grid_dec,grid_ra = np.meshgrid(edges_dec, edges_alpha)
    shape = grid_ra.shape
    grid_ra = grid_ra.flatten()
    grid_dec = grid_dec.flatten()

    # construct a KDTree to find the points nearest to each grid point
    tree = scipy.spatial.KDTree(np.transpose([alpha, dec]))
    matches = tree.query_ball_point(np.transpose([grid_ra, grid_dec]), 10./60)
    distance, matches_knn = tree.query(np.transpose([grid_ra, grid_dec]), 10)

    mask_wt                  = np.zeros(shape)
    for i, m in enumerate(matches):
        # m is a list of indices of galaxies within the radius of the ith grid point
        if len(m)>10:
            mask_wt.flat[i] = np.mean(c_wt[m])
        else:
            m2 = matches_knn[i]
            if len(m2)>0:
                mask_wt.flat[i] = np.mean(c_wt[m2])

    # set unasigned cells to the mean weight
    mask_wt[mask_wt==0] = np.mean(mask_wt[mask_wt>0])

    np.save("weight_%s.npy"%tag, mask_wt)

    ##### Random grid #####
    rand_all  = []
    histr = 0
    histr_flat = 0
    for i in range(realizations):

        rar,decr,zr,zweight = B2R_randoms.randoms(paths,tag, dens_randoms,z_bin,z_mock,mocks=mocks)

        alphar = rar - center

        sel_zr  = zbin(z_min, z_max, zr)
        alphar  = alphar[sel_zr]

        rar, decr, zr = rar[sel_zr], decr[sel_zr], zr[sel_zr]

        random_wt = cell_weight(alphar, decr, mask_wt, edges_alpha, edges_dec)     # A weigth value for each galaxy in the random
        random_wt *= zweight

        # From (zr,decr,rar) to (xr_axis,zr_axis,yr_axis)
        comr_dist = C.comoving_distance(zr)                                                     # From redshift to com distance
        xr_axis, yr_axis, zr_axis = sphere.lonlat2xyz(alphar, decr, r=comr_dist)                # From spherical to cartesian
        xr_axis, yr_axis, zr_axis = sphere.rotate_xyz(xr_axis,yr_axis,zr_axis,[(0,0,delta)])    # Latitude rotation
        xr_axis   = xr_axis - x0

        datar             = (xr_axis,zr_axis,yr_axis)                                           # Instead of (zr,decr,rar)
        histr_temp, edges = np.histogramdd(datar,bins,weights=random_wt)                        # Grid making
        histr_temp_flat, edges = np.histogramdd(datar,bins)
        rand_all.append([alphar, decr, zr, random_wt ])    #

        histr = histr + histr_temp
        histr_flat = histr_flat + histr_temp_flat

    rand_all = np.hstack(rand_all)
    num_randoms = np.sum(histr)

    ##### VIPERS grid #####
    # From (z,dec,ra) to (x_axis,z_axis,y_axis)
    vipers_all= [alpha,   dec,  z, bins, delta]
    com_dist  = C.comoving_distance(z)                                                     # From redshift to comoving distance
    x_axis, y_axis, z_axis = sphere.lonlat2xyz(alpha, dec, r=com_dist)                    # From spherical to cartesian
    x_axis, y_axis, z_axis = sphere.rotate_xyz(x_axis,y_axis,z_axis,[(0,0,delta)])        # Latitude rotation
    x_axis   = x_axis - x0

    data       = (x_axis,z_axis,y_axis)                                                   # Instead of (zr,decr,rar)
    hist,edges = np.histogramdd(data,bins)                                                # Grid making
    num_vipers = np.sum(hist)

    #### NORMALIZATION #####
    # Selecting cells correctly filled
    print("applying mask threshold", threshold)
    bad = histr_flat < threshold*histr_flat.max()
    histr[bad] = 0

    # Normalizing
    histr      = histr/np.sum(histr)
    histr_norm = histr * np.sum(hist)
    hist_real  = hist
    hist       = hist/np.sum(hist)                                                        # Normalization

    ##### Density field #####
    dens       = np.zeros(hist.shape)
    mask       = histr>0
    dens[mask] = ( (hist[mask]) / (histr[mask]) ) - 1


    if   com_random==True and com_vipers==False:      # Return randoms data in comoving coord.
        return dens , histr_norm, vipers_all, rand_all

    elif com_random==False and com_vipers==True:      # Return VIPERS data in comoving coord.
        return dens , hist_real, vipers_all, rand_all

    elif com_random==True  and com_vipers==True:      # Return randoms and VIPERS in comoving coord.
        return dens , histr_norm , hist_real, vipers_all, rand_all

    return dens, vipers_all, rand_all


def new_densfield(config, z_bin, vipers_all, rand_all, Om0=0.31361):
    threshold                         = config['threshold']
    z_min, z_max                      = z_bin
    alphar, decr, zr, random_wt       = rand_all
    alpha,   dec,  z, bins, delta     = vipers_all

    C     = cosmo(H0=100,Om0=Om0)
    x0    = C.comoving_distance(z_min).value - 4

    ##### RANDOM #####
    comr_dist = C.comoving_distance(zr)                                                     # From redshift to com distance
    xr_axis, yr_axis, zr_axis = sphere.lonlat2xyz(alphar, decr, r=comr_dist)                # From spherical to cartesian
    xr_axis, yr_axis, zr_axis = sphere.rotate_xyz(xr_axis,yr_axis,zr_axis,[(0,0,delta)])    # Latitude rotation
    xr_axis   = xr_axis - x0

    datar        = (xr_axis,zr_axis,yr_axis)                                           # Instead of (zr,decr,rar)
    histr, edges = np.histogramdd(datar,bins,weights=random_wt)                        # Grid making
    histr_flat, edges = np.histogramdd(datar,bins)
    num_randoms  = np.sum(histr)


    ##### VIPERS #####
    com_dist  = C.comoving_distance(z)                                                     # From redshift to comoving distance
    x_axis, y_axis, z_axis = sphere.lonlat2xyz(alpha, dec, r=com_dist)                    # From spherical to cartesian
    x_axis, y_axis, z_axis = sphere.rotate_xyz(x_axis,y_axis,z_axis,[(0,0,delta)])        # Latitude rotation
    x_axis   = x_axis - x0

    data       = (x_axis,z_axis,y_axis)                                                   # Instead of (zr,decr,rar)
    hist,edges = np.histogramdd(data,bins)                                                # Grid making
    num_vipers = np.sum(hist)

    ##### NORMALIZATION #####
    # Selecting full cells
    bad = histr_flat < threshold*histr_flat.max()
    histr[bad] = 0

    # Normalizing
    histr        = histr/np.sum(histr)
    histr_norm = histr * np.sum(hist)                                                     # Random normalized to galaxy number
    hist       = hist/np.sum(hist)                                                        # Normalization


    ##### Density field #####
    dens       = np.zeros(hist.shape)
    mask       = histr>0
    dens[mask] = ( (hist[mask]) / (histr[mask]) ) - 1

    return dens, histr_norm

def mockfield(paths,tag,mocknum):

    ra  = []
    dec = []
    z   = []

    mockpers=np.loadtxt(paths['path_mocks']+tag+"/error_mock"+tag+"_"+str(mocknum)+".txt")

    if tag == 'w1':
        mockpers_aux=np.loadtxt(paths['path_mocks']+"w4/error_mockw4_"+str(mocknum)+".txt")
    elif tag == 'w4':
        mockpers_aux=np.loadtxt(paths['path_mocks']+"w1/error_mockw1_"+str(mocknum)+".txt")

    ra      = mockpers[:,0]
    dec     = mockpers[:,1]
    z       = mockpers[:,2]
    z_aux   = mockpers_aux[:,2]

    mockpers = np.transpose([ra,dec,z])

    # Computing the redshift ditribution of the considered mock
    w1_hist,edges=np.histogram(z,bins=np.linspace(0.4,1.2,81))
    w4_hist,edges=np.histogram(z_aux,bins=np.linspace(0.4,1.2,81))
    hist=(w1_hist+w4_hist)/(z.sum()+z_aux.sum())

    z_dist = ndimage.filters.gaussian_filter(hist, 5,order=0)

    z_mock = np.transpose(np.array([edges[:80],z_dist]))

    return mockpers, z_mock
