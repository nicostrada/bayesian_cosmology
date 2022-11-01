import sys
import bins
import time
import yaml
import pylab
import scipy
import emcee
import corner
import logging
import fftutils
import progressbar
import numpy as np
import B2R_densfield
from classy import Class
from pprint import pprint
from scipy  import special
logging.basicConfig(level=logging.INFO)
from scipy.interpolate import UnivariateSpline

pprint(sys.argv)

conf_path = sys.argv[1]
data_path = sys.argv[2]
nameid    = sys.argv[3]
field     = sys.argv[4]
z_min     = float(sys.argv[5])
z_max     = float(sys.argv[6])
wf_iter   = int(sys.argv[7])
mocknum   = int(sys.argv[8])

if z_min == 0.6 and z_max== 0.9:
    z_id = 'z1'
elif z_min == 0.9 and z_max== 1.1:
    z_id = 'z2'
elif z_min == 0.6 and z_max== 1.1:
    z_id = 'z3'
elif z_min == 0.6 and z_max== 1.0:
    z_id = 'z4'
elif z_min == 0.6 and z_max== 0.8:
    z_id = 'lowz'
elif z_min == 0.8 and z_max== 1.0:
    z_id = 'highz'
else:
    z_id = 'z'

run_id    = str(nameid+str(mocknum)+'_'+field+'_'+z_id)

with open(conf_path) as fid:
    config = yaml.load(fid, Loader=yaml.Loader)
with open(data_path) as fid:
    paths = yaml.load(fid, Loader=yaml.Loader)
pprint(config)

if mocknum == 0:
    mocks= False
elif mocknum != 0:
    mocks= True


B2Rprior       = config['prior']

dens_rand      = config['dens_rand']          # Number of random points to generate the mean density
realizations   = config['realizations']       # Nuber of random catalogues used for the mean density
phys_cell_size = config['phys_cell_size']     # Physical dimensions of each cell in h-1 Mpc
threshold      = config['threshold']          # Threshold for cancelling empty cells

num_kbins      = config['num_kbins']          # Number of bins in k space
k_max          = config['k_max']              # Percent of the Nyquist frequency
k_class        = config['k_class']            # OPTIONS:  'k_lin' 'k_log' 'k_equal'

ndim           = config['ndim']               # Number of dimensions on which EMCEE works
nwalkers       = config['nwalkers']           # Number of walkers (min 2*ndim)
emcee_run      = config['emcee_run']          # Number of steps for each walker
emcee_burniter   = config['emcee_burniter']   # Number of iterations with more steps
emcee_burnstep   = config['emcee_burnstep']   # Number of steps for the firs


z_bin          = [z_min, z_max]
z_eff          = (z_max+z_min)/2

params = config['class_params']

prior_Om       = [B2Rprior['Omega_Mh_min']/params['h'],B2Rprior['Omega_Mh_max']/params['h']]

labels= config['labels']
start = config['start']
sig   = config['sig']
OM_0  = start[0]/params['h']
fb = start[1]

params['Omega_b'] = OM_0*fb
params['Omega_cdm'] = OM_0*(1-fb)
params['z_pk'] = z_eff

cosmo = Class()
cosmo.set(params)
cosmo.compute()

with open(run_id + '_start.txt','w') as n:
    print('field          = ', field ,file=n)
    print('z_bin          = ', z_bin ,file=n)
    print('phys_cell_size = ', phys_cell_size , ' [h-1 Mpc]',file=n)
    print('randoms dens   = ', dens_rand, ' realizations = ', realizations, file=n)
    print('threshold      = ', threshold,file=n)
    print('wf_iter        = ', wf_iter ,file=n)
    print('ndim           = ', ndim , '                              nwalkers = ', nwalkers, file=n)
    print('emcee_burniter = ', emcee_burniter ,file=n)
    print('emcee_step     = ', emcee_burnstep ,file=n)
    print('emcee_run      = ', emcee_run ,file=n)
    print('num_kbins      = ', num_kbins, '       k_max [percent of K_Nyquist]= ', k_max, '    k_class = ', k_class, file=n)
    print('start          = ', start,file=n)
    print('PRIOR',file=n)
    print(B2Rprior,file=n)
    print('params', file=n)
    print(params ,file=n)

class WF:
    def __init__(self, S, N):
        self.N=N
        self.S=S
        self.S.flat[0] = 1
        self.shape=(S.size,S.size)
        self.dtype=None

    def matvec(self,xhat):      # Input: S(Fourier space, 1d), N(Conf. space, 1d), x(Conf. space)
        xhat  = np.reshape(xhat,self.S.shape)
        xhatk = fftutils.gofft_numpy(xhat)
        q = (1/self.S)*xhatk
        q = fftutils.gofftinv_numpy(q).real
        z = self.N*xhat

        return q + z            # Output: Conf space

def k_bins(k_big, lenght, k_max, nbins, cell_size=1, type=k_class):

    k_inf = np.pi/length.max()        # Larger k mode computed on the survey - max dimensions og the grid
    k_sup = (np.pi*k_max)/cell_size   # Smallest k mode - be careful of aliasing effects

    sel_inf = k_big >= k_inf
    sel_sup = k_big <= k_sup
    sel_k   = np.all([sel_inf,sel_sup],axis=0)
    k_true  = k_big[sel_k]                                   # Selecting the apropriate k values
    k_true.sort()

    if type == 'k_lin':
        kbins      = np.linspace(k_inf,k_sup,nbins)
        numk_each, bins_new = np.histogram(k_true, bins=kbins)
        for i in range (len(numk_each)):
            if numk_each[i] == 0:
                numk_each[i]= 1

    if type == 'k_log':
        #kbins     = np.logspace(np.log10(k_true.min()),np.log10(k_true.max()),nbins)
        kbins      = np.logspace(np.log10(0.25)        ,np.log10(k_true.max()),nbins)   # The correction on large scales is fair, otherwise there are many empty bins

        numk_each, bins_new = np.histogram(k_true, bins=kbins)

        for i in range (len(numk_each)):
            if numk_each[i] == 0:
                numk_each[i]= 1

    if type == 'k_equal':
        numk_each = int(k_true.size/nbins)                       # Number of modes on each bin
        kbins   = []
        count_k = 0
        for j in range (nbins):
            kbins.append(k_true[count_k])
            count_k += numk_each
        kbins = np.array(kbins)                                  # Cell units

    kcent = (kbins[1:]+kbins[:-1])/2.                           # Cell units

    return kbins, kcent, numk_each

def Wiener(c,inv_noise_variance,b,densw1):

    # Wiener Filtering
    wf = WF(c,inv_noise_variance)
    A  = scipy.sparse.linalg.LinearOperator(wf.shape,wf.matvec)
    solution,info=scipy.sparse.linalg.cg(A, b,tol=config['cg_tol'], maxiter=config['cg_maxiter'])            # SOLUTION = X_WF
    solution     = np.reshape(solution,densw1.shape)

    # Xi field
    w1 = np.random.normal(0,1,densw1.size)
    w2 = np.random.normal(0,1,densw1.size)
    w1 = np.reshape(w1,densw1.shape)
    w2 = np.reshape(w2,densw1.shape)

    w2_f  = fftutils.gofft_numpy(w2)                              # DO NO COMPUTE K_GRID AT EACH ITER
    temp1 = fftutils.gofftinv_numpy(np.sqrt(1/c)*w2_f)

    b_xi  = np.sqrt(inv_noise_variance)*w1 + temp1
    b_xi  = b_xi.flatten().real

    xi,info_xi = scipy.sparse.linalg.cg(A, b_xi,tol=config['cg_tol'], maxiter=config['cg_maxiter'])
    xi         = np.reshape(xi,densw1.shape)                      # XI

    # Reconstruction of the field (Wiener + Xi)
    best  = solution.real + xi.real
    best -= best.mean()

    return best

def prior(cosm_params, B2Rprior):
    Omega_Mh, fb, bias, sigma_v, fsig8 = cosm_params
    h         = params['h']
    Omega_b   = (Omega_Mh *fb)/h
    Omega_cdm = (Omega_Mh/h)*(1-fb)

    if Omega_cdm < Omega_b:                     return -np.inf
    if Omega_Mh  < B2Rprior['Omega_Mh_min']:    return -np.inf
    if Omega_Mh  > B2Rprior['Omega_Mh_max']:    return -np.inf
    if fb        < B2Rprior['fb_min']:          return -np.inf
    if fb        > B2Rprior['fb_max']:          return -np.inf
    if bias      < B2Rprior['bias_min']:        return -np.inf
    if bias      > B2Rprior['bias_max']:        return -np.inf
    if sigma_v   < B2Rprior['sigma_v_min']:     return -np.inf
    if sigma_v   > B2Rprior['sigma_v_max']:     return -np.inf
    if fsig8     < B2Rprior['fsig8_min']:       return -np.inf
    if fsig8     > B2Rprior['fsig8_max']:       return -np.inf

    inside_prior = 1
    return inside_prior

def check_walkers(p0,B2Rprior):
    inf_prior = [B2Rprior['Omega_Mh_min'],B2Rprior['fb_min'],B2Rprior['bias_min'],B2Rprior['sigma_v_min'], B2Rprior['fsig8_min']]
    sup_prior = [B2Rprior['Omega_Mh_max'],B2Rprior['fb_max'],B2Rprior['bias_max'],B2Rprior['sigma_v_max'], B2Rprior['fsig8_max']]

    #fixing    = [0.1, 0.1, 0.3, 0.3, 0.3]
    fixing    = [0.1, 0.1, 0.1, 0.1, 0.1]
    new_p0    = []

    for p in p0:
        for i in range(len(p)):
            if   p[i] < inf_prior[i]:
                p[i]  = inf_prior[i] + np.random.uniform(0,fixing[i])
                print('We had to fix walker '+str(i)+' because it went to the INFERIOR limit of the prior')
            elif p[i] > sup_prior[i]:
                p[i]  = sup_prior[i] - np.random.uniform(0,fixing[i])
                print('We had to fix walker '+str(i)+' because it went to the SUPERIOR limit of the prior')
        new_p0.append(p)
    p0 = np.array(new_p0)

    return p0

class PkInterp:
    logkmin = -3
    logkmax = np.log10(5)
    nbins = 1000
    def __init__(self, cosmo, params):
        """ """
        self.cosmo = cosmo
        self.params = params
        self.cosmo.set(self.params)
        self.cosmo.compute()
        k = np.logspace(self.logkmin, self.logkmax, self.nbins)
        logk = np.log10(k)
        pk = self.run_class(k)
        self._interp = scipy.interpolate.interp1d(logk, np.log10(pk), bounds_error=True, fill_value=0)

    def run_class(self, k):
        """ """
        kk = np.zeros((len(k),1,1), dtype='d')
        kk[:,0,0] = k*self.params['h']
        zz = np.array([self.params['z_pk']])
        pk_CLASS = self.cosmo.get_pk(kk, zz, len(k), 1, 1).reshape((len(k),))
        pk_CLASS *= self.params['h']**3
        return pk_CLASS

    def get_pk(self, k):
        """ """
        return 10**self._interp(np.log10(k))

def model(cosm_params, k_x, k_y, k_z, cosmo, phys_cell_size, B2Rprior, z=z_eff,h=params['h'], aliasing_nmax=1):
    """Calculates the model power spectrum"""
    Omega_Mh, fb, bias, sigma_v, fsig8 = cosm_params

    prior_ok = prior(cosm_params,B2Rprior)
    if prior_ok == -np.inf: return -np.inf

    Omega_b   = (Omega_Mh *fb)/h
    Omega_cdm = (Omega_Mh/h)*(1-fb)

    params = {
        'non linear': 'halofit',
        'output': 'mPk',
        'P_k_max_h/Mpc': 5,
        'A_s': 2.0968e-9,
        'YHe': 0.2454,
        'n_s': 0.9652,
        'z_pk': z,
        'h': h,
        'Omega_b': Omega_b,
        'Omega_cdm': Omega_cdm}


    pk = PkInterp(cosmo, params)

    pk_model =np.zeros(k_x.size)


    sig8 = cosmo.sigma8()*cosmo.scale_independent_growth_factor(params['z_pk'])

    knyquist = np.pi / phys_cell_size

    nx,ny,nz=np.mgrid[:aliasing_nmax+1,:aliasing_nmax+1,:aliasing_nmax+1]
    nx = nx.flatten()*2*knyquist
    ny = ny.flatten()*2*knyquist
    nz = nz.flatten()*2*knyquist

    #print(f"Computing aliasing harmonics {len(nx)}")

    for i in range(len(nx)):

        k      = np.sqrt((k_x+nx[i])**2 + (k_y+ny[i])**2 + (k_z+nz[i])**2)                                # Length of the k vector
        mu     = (k_x+nx[i])/k

        k      = k.flatten()
        mu     = mu.flatten()

        pk_CLASS = pk.get_pk(k)

        w    = 1.
        step = phys_cell_size
        kgrid = [k_x+nx[i], k_y+ny[i], k_z+nz[i]]

        for j in range(len(kgrid)):
            x       = kgrid[j]*step/2.
            bad     = x==0
            x[bad]  = 1
            w       = w*np.sin(x)/x
            w[bad]  = 1
        w = w.flatten()

        a=2
        if i==0:
            a=1
        pk_model += a*pk_CLASS*((bias+(fsig8/sig8)*mu**2)**2)*np.exp(-(k*mu*sigma_v)**2)*(w**2)

    cosmo.struct_cleanup()

    return pk_model                                                           # Physical units

def lnprob(cosm_params, cosmo, phys_cell_size, B2Rprior, k_x, k_y, k_z, pk_true_phys, comb=False):
    Omega_Mh, fb, bias, sigma_v, fsig8 = cosm_params
    h = params['h']

    Omega_b   = (Omega_Mh *fb)/h
    Omega_cdm = (Omega_Mh/h)*(1-fb)

    prior_ok = prior(cosm_params, B2Rprior)
    if prior_ok == -np.inf: return -np.inf

    try:
        pk_model = model(cosm_params, k_x, k_y, k_z, cosmo, phys_cell_size, B2Rprior)      # Physical units
    except:
        return -np.inf
    # pk_true is delta square...
    likeA = 0.5*np.sum(pk_true_phys/pk_model)
    likeB = 0.5*np.sum(np.log(pk_model))
    if comb==True:
        likeB *= 2

    likelihood = -likeA - likeB

    return likelihood

t0 = time.time()
print('RUNNING!')

##### Generating the field #####
if   field == 'w1'or field == 'w4':
    combined = False
    dens   , rand , vipers_all, rand_all      = B2R_densfield.densfield(config,paths,field,z_bin,prior_Om,phys_cell_size,realizations,dens_rand,mocknum,Om0=OM_0,com_random=True,mocks=mocks)
    t1 = time.time()
    print('Time employed on generating one VIPERS density field: ', t1-t0)

elif field == 'w1w4':
    combined = True
    dens   , rand, vipers_all, rand_all       = B2R_densfield.densfield(config,paths,'w1',z_bin,prior_Om,phys_cell_size,realizations,dens_rand,mocknum,Om0=OM_0,com_random=True,mocks=mocks)
    t1A = time.time()
    print('Time employed on generating VIPERS W1 density field: ', t1A-t0)
    densw4 , randw4, vipers_allw4, rand_allw4 = B2R_densfield.densfield(config,paths,'w4',z_bin,prior_Om,phys_cell_size,realizations,dens_rand,mocknum,Om0=OM_0,com_random=True,mocks=mocks)
    t1 = time.time()
    print('Time employed on generating VIPERS W4 density field: ', t1-t1A)

alpha_xxx,   dec_xxx,  z_xxx, bins_xxx, delta_xxx     = vipers_all
##### Setting parameters #####
shape          = dens.shape                          # ( , , ) (cell units)
length         = np.array(shape)                     # Dimensionless (cell units)

with open(run_id + '_start.txt','a') as n:
    print('',file=n)
    print('shape          = ', shape ,file=n)

with open(run_id + '_bins.txt','w') as n:
    print('# bins[0] = ',file=n)
    print(bins_xxx[0],file=n)
    print('',file=n)
    print('# bins[1] = ',file=n)
    print(bins_xxx[1],file=n)
    print('',file=n)
    print('# bins[2] = ',file=n)
    print(bins_xxx[2],file=n)


k_big, pk_obs  = fftutils.powerspectrum(dens, length)    # Cell units (Useful only for the k value)
kbins, kcent, numk_each   = k_bins(k_big, length, k_max, num_kbins, type=k_class)

t2 = time.time()
print('Time for computing the FFT of the first Wiener solution', t2-t1)

### NEW PART ###
k_x, k_y, k_z        = fftutils.kgrid3d(shape, length)
k_x = k_x.flatten()
k_y = k_y.flatten()
k_z = k_z.flatten()

### Boolean for selecting appropiate values
k_tot                = np.sqrt(k_x**2 + k_y**2 + k_z**2)

k_inf = np.pi/length.max()         # Cell units
k_sup = np.pi*k_max                # Cell units (In physical units is divided by cell size)

sel_inf = k_tot >= k_inf
sel_sup = k_tot <= k_sup
sel_k   = sel_inf*sel_sup

### K in cartesian reference and physical units
k_x    = k_x/ phys_cell_size
k_x[0] = k_x[1]

k_y    = k_y/ phys_cell_size
k_y[0] = k_y[1]

k_z    = k_z/ phys_cell_size
k_z[0] = k_z[1]

t3 = time.time()
print('Time for computing kgrid', t3-t2)

inv_noise_variance   = rand                                          # WF INPUT ---> dens shape & Cell units
b = dens*inv_noise_variance
b = b.flatten()

if combined==True:
    inv_noise_variancew4 = randw4                    # WF INPUT ---> shape should be ( , , ) Cell units
    bw4 = densw4*inv_noise_variancew4
    bw4 = bw4.flatten()

##### Generating the correct input for cycle #####
c      = model(start,k_x, k_y, k_z, cosmo, phys_cell_size, B2Rprior)           # Physical units
print(c.shape)
print(c)
c      = c/(phys_cell_size**3)                   # Cell units
c      = np.reshape(c,dens.shape)                # Cell units

reg_mod     = []
reg_cwf     = []

##### Generating the starting position of the walkers #####
p0    = np.random.normal(0,sig,(nwalkers,ndim)) + start
p0    = check_walkers(p0, B2Rprior)

with open(run_id + '_params.txt','w') as n:
    print('# OMh','# fb','# h','# bias','# sigma_v', '# fsig8',file=n)
    for k in range(len(start)-1):
        print(start[k],end=" ", file=n)
    print(start[-1],file=n)

with open(run_id + '_walkers.txt','w') as n:
    print('# OMh','# fb','# h','# bias','# sigma_v', '# fsig8',file=n)
    for k in range(len(start)-1):
        print(start[k],end=" ", file=n)
    print(start[-1],file=n)

    for j in range(nwalkers):
        for k in range(ndim-1):
            print(p0[-j,k],end=" ", file=n)
        print(p0[-j,-1],file=n)

with open(run_id + '_model.txt','w') as n:
    print('# model pk at each iteration',file=n)

with open(run_id + '_wiener.txt','w') as n:
    print('# measured pk at each iteration',file=n)

with open(run_id + '_kcent.txt','w') as n:
    for i in range(len(kcent)-1):
        print(kcent[i],end=" ",file=n)
    print(kcent[-1],file=n)

with open(run_id + '_numk_each.txt','w') as n:
    for i in range(len(numk_each)-1):
        print(numk_each[i],end=" ",file=n)
    print(numk_each[-1],file=n)

t4 = time.time()
print('Time computing c (Model pk) and b(ShotNoise) for the Wiener filter and p0 for EMCEE', t4-t3)
print_array=np.arange(3000,3100,10)
bar      = progressbar.ProgressBar()
for i in bar(range(wf_iter)):
    print('WF iteration = ',i)
    t5 = time.time()
    # Wiener filter and its power spectrum
    best       = Wiener(c,inv_noise_variance,b,dens)

    t6 = time.time()
    print('Time doing the Wiener Filter (Conjugate gradient) of one filed: ', t6-t5)

    k, c_best  = fftutils.powerspectrum(best, length)                          # Giving ps on cell units
    c_best     = c_best.flatten()
    c_b        = bins.binit(k, c_best, kbins)

    pk_true          = c_best[sel_k]
    pk_true_phys     = pk_true*(phys_cell_size**3)

    t7 = time.time()
    print('Time computing FFT of the Wiener solution for one field: ', t7-t6)
    print('BEST SHAPEEEEEEEEE',best.shape)
    p1,p2,p3 = best.shape # Should be kind of 184,16,77

    if combined==True:
        bestw4       = Wiener(c,inv_noise_variancew4,bw4,densw4)
        boop = np.isin(i,print_array)
        if boop == True:
        #if i == 200:
            with open(run_id + '_bestw1_'+str(i)+'.txt','w') as n:
                print('# reshape using this array = ', best.shape ,file=n)
                for i_plot in range(p1):
                    for j_plot in range(p2):
                        for k_plot in range(p3-1):
                            print(best[i_plot,j_plot,k_plot],end=" ",file=n)
                        print(best[i_plot,j_plot,-1],file=n)
                        # use np.loadtxt and reshape!!!
                        #bestw1 = np.loadtxt("").reshape((184,16,77))
                        #bestw4 = np.loadtxt("").reshape((184,16,77))

            with open(run_id + '_bestw4_'+str(i)+'.txt','w') as n:
                print('# reshape using this array = ', best.shape ,file=n)
                for i_plot in range(p1):
                    for j_plot in range(p2):
                        for k_plot in range(p3-1):
                            print(bestw4[i_plot,j_plot,k_plot],end=" ",file=n)
                        print(bestw4[i_plot,j_plot,-1],file=n)

        t7B = time.time()
        print('Time doing the Wiener Filter (Conjugate gradient) of VIPERS W4: ', t7B-t7)

        k, c_bestw4  = fftutils.powerspectrum(bestw4, length)                  # Giving ps on cell units

        c_bestw4     = c_bestw4.flatten()
        c_bw4        = bins.binit(k, c_bestw4, kbins)                               # Selecting the apropriate k values
        pk_truew4    = c_bestw4[sel_k]
        pk_truew4_phys    = pk_truew4*(phys_cell_size**3)

        c_b          = ( c_b + c_bw4 ) / 2
        pk_true_phys = pk_true_phys + pk_truew4_phys

        t7 = time.time()
        print('Time computing FFT of VIPERS W4: ', t7-t7B)

    if k_class=='k_log':
        sel_cb  = c_b >= 1
        c_b     = np.interp(kcent,kcent[sel_cb],c_b[sel_cb])

    reg_cwf.append(c_b)
    if i < emcee_burniter:
        emcee_run = emcee_burnstep
    elif i >= emcee_burniter:
        emcee_run = config['emcee_run']

    # Emcee: Calling the instance
    cosmo_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[cosmo, phys_cell_size, B2Rprior, k_x[sel_k], k_y[sel_k], k_z[sel_k], pk_true_phys, combined],threads=1)
    pos, prob, state = cosmo_sampler.run_mcmc(p0, emcee_run)
    pos = check_walkers(pos, B2Rprior)

    t8 = time.time()
    print('Time computing EMCEE: ', t8 - t7)

    # Emcee: saving results and reset the chain
    samples = cosmo_sampler.chain[:,:,:].reshape((-1, ndim))
    like    = cosmo_sampler.lnprobability
    cosmo_sampler.reset()

    np.random.shuffle(pos)
    p0 = pos

    start       = samples[-1,:]
    mean_params = np.mean(samples,axis=0)

    check = np.isfinite(like[-1,-1])
    if check == False:
        start = mean_params
        print('The likelihood of the last walker was -inf at iter ', i , ' but we go ahead...')
        print(samples[-1,:])

    c     = model(start, k_x, k_y, k_z, cosmo, phys_cell_size, B2Rprior)         # Physical units

    if np.array(c).size == 1:
        print('That should never happend!!! ---> ', start)
        start = mean_params
        c     = model(start, k_x, k_y, k_z, cosmo, phys_cell_size, B2Rprior)          # Physical units

    c_new = bins.binit(k, c, kbins)
    c_new = c_new/(phys_cell_size**3)                  # Cell units
    c     = c/(phys_cell_size**3)                      # Cell units
    c     = np.reshape(c,dens.shape)

    reg_mod.append(c_new)                              # Cell units

    t9 = time.time()
    print('Time saving results and computing new c: ', t9 - t8)

    ##### Generating again the field with the new matter density #####
    new_Om0 = start[0]/params['h']

    if   field == 'w1'or field == 'w4':
        combined = False
        dens, rand = B2R_densfield.new_densfield(config,z_bin, vipers_all, rand_all, Om0=new_Om0)
    elif field == 'w1w4':
        combined = True
        dens, rand     = B2R_densfield.new_densfield(config,z_bin, vipers_all,   rand_all,   Om0=new_Om0)
        densw4, randw4 = B2R_densfield.new_densfield(config,z_bin, vipers_allw4, rand_allw4, Om0=new_Om0)

    inv_noise_variance   = rand                          # WF INPUT ---> dens shape & Cell units
    b = dens*inv_noise_variance
    b = b.flatten()

    if combined==True:
        inv_noise_variancew4 = randw4                    # WF INPUT ---> shape should be (74,74,74) Cell units
        bw4 = densw4*inv_noise_variancew4
        bw4 = bw4.flatten()

    t10 = time.time()
    print('Time calculating the new density field: ', t10 - t9)

    with open(run_id + '_params.txt','a') as n:
        for k in range(len(start)-1):
            print(start[k],end=" ", file=n)
        print(start[-1],file=n)

    with open(run_id + '_walkers.txt','a') as n:
        for j in range(nwalkers):
            for k in range(ndim-1):
                print(p0[j,k],end=" ", file=n)
            print(p0[j,-1],file=n)

    with open(run_id + '_model.txt','a') as n:
        for k in range(len(c_new)-1):
            print(c_new[k],end=" ",file=n)
        print(c_new[-1],file=n)

    with open(run_id + '_wiener.txt','a') as n:
        for k in range(len(c_b)-1):
            print(c_b[k],end=" ",file=n)
        print(c_b[-1],file=n)
