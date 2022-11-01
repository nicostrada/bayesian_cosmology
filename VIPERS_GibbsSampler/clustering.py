import copy
import logging
import numpy as np
import scipy.special
import scipy.integrate
from classy import Class
from scipy  import interpolate
from hankel import HankelTransform
N = np

def legendre_poly(l, x):
    """return the legendre polynomials """
    if l == 0:  # monopole
        return N.ones(len(x))
    elif l == 2: # quadrupole
        return 0.5*(3*x*x-1)
    elif l == 4: # hex
        x2 = x*x
        return 1./8 * (35*x2*x2 - 30*x2 + 3)
    elif l == 6:
        x2 = x*x
        return 1./16 * (231*x2**3 - 315*x2**2 + 105*x2 - 5)
    else:
        raise Exception("multipole %s is not implemented!"%str(l))


#dictionaries for spectrum parameters
class_params = {
    'output': 'mPk',
    'non linear': 'halofit',
    'P_k_max_h/Mpc': 300.,
    'z_pk': 0.7546,
    'A_s': 2.0968e-9,
    'n_s': 0.9652,
    'h': 0.6732,
    'Omega_b': 0.04927,
    'Omega_cdm': 0.26434,
    }

params = {
    'sigma8': 0.8,
    'bias1': 1.,
    'bias2': 1.,
    'sigmav': 0,
    'noiseb': 0,
    'f': 0,
    'n_mubins': 6,
    'interp_rmin': 1e-1,
    'interp_rmax': 1000,
    'interp_steps': 1000,
    'auto': True,
    'hankel_nmax': 10000,
    'hankel_n': 10000,
    'hankel_h': 1./4096,
    'dispersion_mode': 'gaussian',
    'q1halo':1.,
    }


class Clustering(object):
    """ """
    logger = logging.getLogger(__name__)

    def __init__(self, k=None, cell_radius=1, cell_shape='sphere', **param_dict):
        """ """
        if k is None:
            k = N.logspace(-4,2,10000)
        self.k = k

        self.multipoles = {}
        self.Hankel = {}
        self.cf_interpolator = {}

        self.cosmo = Class() #initialization for Class package used for compute theoretical power spectrum

        self.params = params.copy()
        self.class_params = class_params.copy()
        self.set(**param_dict) #other params for CLASS

        self.cosmo.set(self.class_params)

        self.logger.info("Computing power spectrum with Class code")
        self.cosmo.compute()

        sig8 = self.cosmo.sigma8() #Get sigma8
        self.logger.info("class sig8: %f"%sig8)

        A_s = self.cosmo.pars['A_s']
        self.cosmo.struct_cleanup()

        self.cosmo.set(A_s=A_s*(self.params['sigma8']*1./sig8)**2)

        self.logger.debug(self.cosmo.pars)

        self.cosmo.compute()
        sig8 = self.cosmo.sigma8()

        self.logger.info("renormalized class sig8: %f"%sig8)

        self.pk = N.zeros(len(self.k))
        for i in range(len(self.k)):
            self.pk[i] = self.cosmo.pk(self.k[i]*self.class_params['h'], self.class_params['z_pk'])*self.class_params['h']**3 ### Physical units: P(k*h)*h^3 Returns (Mpc/h)^3

        self.xi0 = None
        self.cell_radius = cell_radius
        self.cell_volume = 4./3.*N.pi*self.cell_radius**3

        # self.pk /= self.cell_volume #Normalize Pk
        #self.logger.debug("Cell volume: %f (Mpc/h)^3", self.cell_volume)
        #self.logger.debug("Cell radius: %f Mpc/h", self.cell_radius)
        #self.logger.info("Nyquist frequency: %f", N.pi/self.cell_radius/2)

    def get_pk(self,k,applywindow=False):

        self.logger.warning("Called get_pk()!")
        pk=N.zeros(len(k))
        for i in range(len(k)):
            pk[i]=self.cosmo.pk(k[i]*self.class_params['h'], self.class_params['z_pk'])*self.class_params['h']**3

        if applywindow:
            pk *= self.get_window(k)
        return pk

    def get_window(self, k=None, fudge=1.): ###Limited-volume survey window function: top-hat fourier transform
        """ """
        if k is None:
            k = self.k
        x = N.array(k) * self.cell_radius
        window = 3*(N.sin(x)/x - N.cos(x))/(x*x)
        return window * window

    def copy(self):
        """ """
        return copy.deepcopy(self)

    def set(self, **param_dict):
        """ """
        for key,value in param_dict.items():
            if key == 'non_linear':
                key = 'non linear'
            self.logger.debug("Set %s=%s",key,value)
            found = False
            if self.class_params.has_key(key):
                self.class_params[key] = value
                found = True
            if self.params.has_key(key):
                self.params[key] = value
                found = True
            if not found:
                raise Exception("Unknown parameter %s: %s"%(key,value))

        assert(self.params['dispersion_mode'] in ('gaussian','lorentzian','lorentziansquare'))

        self.reset()

    def reset(self):
        self.multipoles = {}
        self.Hankel = {}
        self.cf_interpolator = {}

    # #Compute Pk multipoles
    # def pk_multipole(self, l=0, deriv=None):
    #     """ """
    #     self.logger.debug("pk_multipole bias1: %s, bias2: %s, f: %s, sigmav: %s, noiseb: %s", self.params['bias1'],self.params['bias2'],self.params['f'],self.params['sigmav'],self.params['noiseb'])
    #     if deriv is not None:
    #         self.logger.debug("pk_mulitpole %i deriv %s",l,deriv)

    #     l = int(l)
    #     key = l
    #     if deriv is not None:
    #         key = "%s%i"%(deriv, l)
    #     if self.multipoles.has_key(key):
    #         return self.multipoles[key]

    #     self.logger.debug("computing pk interpolator")

    #     pout = self.pk * self.get_window(self.k)

    #     self.multipoles[key] = pout

    #     return pout

    #Compute Correlation Function Interpolation
    def _cf_multipole_interpolator(self, l=0, deriv=None):
        """ """
        l = int(l)
        assert l == 0

        key = l
        if deriv is not None:
            key = "%s%i"%(deriv, l)

        if self.cf_interpolator.has_key(key):
            return self.cf_interpolator[key]

        self.logger.debug("initializing cf interpolator l=%i, deriv=%s",l,deriv)

        r = N.logspace(N.log10(self.params['interp_rmin']),
                       N.log10(self.params['interp_rmax']),
                       self.params['interp_steps'])


        pk = self.pk

        # pk = self.pk_multipole(l, deriv=deriv)

        #pk = self.pk * self.get_window(self.k)
        #self.logger.debug("clustering applying window")

        func = scipy.interpolate.interp1d(self.k, pk, bounds_error=False, fill_value=0.,assume_sorted=True) #

        if not self.Hankel.has_key(l):
            self.Hankel[l] = HankelTransform(nu=0.5, N=1000, h=.001)

        xi = np.zeros(len(r), dtype='d')
        for i in range(len(r)):
            xi[i] = self.Hankel[l].integrate(lambda x: func(x/r[i]) * x**1.5, ret_err=False)

        norm = r**-3 * 1.0/(2.0*np.pi)**1.5
        xi *= norm

        s = (-1)**(l//2)

        xi *= s


        self.cf_interpolator[key] = scipy.interpolate.interp1d(r, xi, bounds_error=False, fill_value=0.,assume_sorted=True) #Interpolate corr_func values for given l


        return self.cf_interpolator[key]

    def cf_multipole(self, r, l=0, deriv=None):
        """ """
        interp = self._cf_multipole_interpolator(l=l, deriv=deriv)

        #self.logger.debug("computing cf %i",len(r))

        ii = r>0
        f = 0
        if l==0:
            if self.xi0 is None:
                d = self.params['interp_rmin']
                f = interp(d)
            else:
                f = self.xi0

        #if deriv is not None:
        #    f = 0.

        y = N.ones(r.shape)*f
        y[ii] = interp(r[ii])

        #self.logger.debug("done computing cf %i",len(r))

        return y

    def set_zero_lag(self, xi0):
        """ """
        self.xi0 = xi0

    def cf(self, r, mu, lmax=0, deriv=None):
        """ Compute correlation functino

        Inputs
        ------
        r     - separation
        mu    - cos angle to line of sight
        lmax  - maximum multipole to use
        deriv - key word for computing derivative with respect to a parameter

        Outputs
        -------
        covariance
        """
        xi = self.cf_multipole(r, l=0, deriv=deriv)

        if lmax >= 2:
            for l in range(2,int(lmax)+1,2):
                pl = legendre_poly(l, mu)
                xi = xi + pl * self.cf_multipole(r, l, deriv=deriv)

        return xi

    cov = cf

    def projected_cf(self, rp, pi_max=50, lmax=0, pi_step=0.1):
        """ """
        pi = np.arange(0, pi_max, pi_step)

        x, y = np.meshgrid(rp, pi)

        shape = x.shape

        x = x.flatten()
        y = y.flatten()

        r = np.sqrt(x*x + y*y)
        mu = np.zeros(r.shape, dtype='d')
        pos = r > 0
        mu[pos] = y[pos] * 1./ r[pos]

        xi = self.cf(r, mu, lmax=lmax).reshape(shape)

        wp = xi.sum(axis=0) * pi_step * 2

        return wp


def demo_cf():
    logging.basicConfig(level=logging.DEBUG)
    import pylab

    C = Clustering()
    x = N.arange(0,200,5)
    x,y = N.meshgrid(x,x)
    shape = x.shape
    x = x.flatten()
    y = y.flatten()
    r = N.sqrt(x**2+y**2)
    r[0] = 1
    mu = y/r
    xi = C.cov(r,mu, lmax=0, deriv=None)
    xi = xi.reshape(shape)
    r = r.reshape(shape)

    pylab.imshow(N.log10(N.abs(xi)), origin='lower')
    pylab.colorbar()
    pylab.contour(N.log10(N.abs(xi)), colors='k', origin='lower')
    pylab.show()

if __name__=="__main__":
    import time
    demo_cf()
    exit()

    logging.basicConfig(level=logging.DEBUG)

    r = N.logspace(0,3.5,2000)

    t0 = time.time()
    C1 = Clustering(hankel_nmax=500,hankel_n=500,hankel_h=1./256)
    xi1 = C1.cov(r,N.zeros(len(r)))

    t1 = time.time()

    C2 = Clustering(hankel_nmax=10000,hankel_n=10000,hankel_h=1./512)
    xi2 = C2.cov(r,N.zeros(len(r)))

    t2 = time.time()

    #p0 = C.pk_multipole(2)
    #p2 = C.pk_multipole(2, deriv='f')
    #p4 = C.cf_multipole(r, 4)


    import pylab
    pylab.semilogx(r,r**2*xi1)
    pylab.semilogx(r,r**2*xi2)

    print("times",t1-t0,t2-t1)
    #pylab.plot(C.k,N.abs(p2))
    #pylab.plot(r,N.abs(p4))

    pylab.show()
