import sys
import time
import numpy as N

nthreads = 1

def powerspectrum(grid, length, mask=None, zeropad=None, norm=1, getdelta=False,computek=True, nthreads=1):
    """ Compute the power spectrum <--- MEASURES THE POWER SPECTRUM OF THE INPUT!!!
    Inputs:
      grid -- delta values 1, 2 or 3 dimensions
      length -- physical dimensions of box
    Outputs:
      k, pk
    """
    shape = grid.shape
    dim = len(shape)

    if not zeropad==None:
        bigbox = N.zeros(N.array(grid.shape)*zeropad)
        if dim==3: bigbox[:grid.shape[0],:grid.shape[1],:grid.shape[2]] = grid
        if dim==2: bigbox[:grid.shape[0],:grid.shape[1]] = grid

        bigmask = None
        if not mask is None:
            bigmask = N.zeros(N.array(grid.shape)*zeropad)
            bigmask[:grid.shape[0],:grid.shape[1],:grid.shape[2]] = mask

        return powerspectrum(bigbox,N.array(length)*zeropad, mask=bigmask, zeropad=None, getdelta=getdelta, norm=zeropad**3, nthreads=nthreads,computek=computek)

    if N.shape(length)==():          # True if length is a number
        length = N.ones(dim)*length  # create a list

    assert(len(length)==dim)

    t0 = time.time()
    dk = gofft(grid, nthreads=nthreads)

    dk *= N.sqrt(N.prod(length)*norm)

    if not mask is None:
        print ("no use of a mask is implemented!")

    pk = N.abs(dk**2)
    pk = pk.flatten()

    # save significant time if we dont need to recompute k
    if not computek:
        #print "skipping k comptuation"
        if getdelta:
            return pk, dk
        return pk
    if dim==3:
        kgrid = kgrid3d(shape, length)
    elif dim==2:
        kgrid = kgrid2d(shape, length)
    elif dim==1:
        kgrid = kgrid1d(shape, length)
    else:
        print("fftutils: bad grid dimension:",dim, file=sys.stderr)
        raise

    #print kgrid[0].max()
    s = 0
    for i in range(dim):
        s += kgrid[i]**2
    k = s.flatten()**.5

    #pk = pk[1:]
    #k = k[1:]
    pk = pk[0:]                # Because the correct dimension of the output is with all the values of pk
    k[0] = k[1]                # Because k[0] is zero and it's not possible

    assert(N.all(k>0))

    #print "kmax",k.max(),shape,length

    # sorting is pretty slow
    # order = k.argsort()
    # k = k[order][1:]
    # pk = pk[order][1:]

    if getdelta:
        return k, pk, (kgrid, dk)
    return k, pk


def gofft_numpy(grid, nthreads=1):
    """ Forward FFT """
    n = N.prod(grid.shape)          # Volume of the sample
    dk = 1./n*N.fft.fftn(grid)      # Fast Fourier Transform
    return dk

def gofftinv_numpy(grid, nthreads=1):
    """ inverse FFT """
    n = N.prod(grid.shape)
    d = n*N.fft.ifftn(grid)
    return d

def gofft_fftw(grid, nthreads=1):
    """ Forward FFT """
    #print "gofft_fftw"
    n = N.prod(grid.shape)

    #print "size",grid.shape,grid.nbytes*1./1024**3
    if grid.dtype=='complex':
        print ("ok complex")
        g = grid
    else:
        g = grid.astype('complex')
    dk = 1./n*fftw.fft(g, grid.shape, nthreads=nthreads)
    return dk

def gofftinv_fftw(grid, nthreads=1):
    """ inverse FFT """
    #print "WARNING!     gofftinv_fftw has not been tested!!!!"
    n = N.prod(grid.shape)
    d = n*fftw.ifft(grid, grid.shape,nthreads=nthreads)
    return d

gofft = gofft_numpy
gofftinv = gofftinv_numpy
kgridcache = {}

def kgrid3d(shape, length):
    """ Return the array of frequencies """
    key = '%s %s'%(shape[0],length[0])
    if key in kgridcache:
        #print ("hitting up kgrid cache")
        return kgridcache[key]

    a = N.fromfunction(lambda x,y,z:x, shape)
    a[N.where(a > shape[0]//2)] -= shape[0]
    b = N.fromfunction(lambda x,y,z:y, shape)
    b[N.where(b > shape[1]//2)] -= shape[1]
    c = N.fromfunction(lambda x,y,z:z, shape)
    c[N.where(c > shape[2]//2)] -= shape[2]

    norm = 2*N.pi
    a = a*norm*1./length[0]
    b = b*norm*1./length[1]
    c = c*norm*1./length[2]

    kgridcache[key] = (a,b,c)

    return a,b,c

def kgrid2d(shape, length):
    """ Return the array of frequencies """

    a = N.fromfunction(lambda x,y:x, shape)
    a[N.where(a > shape[0]//2)] -= shape[0]
    b = N.fromfunction(lambda x,y:y, shape)
    b[N.where(b > shape[1]//2)] -= shape[1]

    norm = 2*N.pi
    a = a*norm*1./(length[0])
    b = b*norm*1./(length[1])

    return a,b

def kgrid1d(shape,length):
    """ Return the array of frequencies """

    a = N.arange(shape[0])
    a[N.where(a > shape[0]//2)] -= shape[0]
    a = a*2*N.pi*1./(length[0])

    return N.array([a])

def testpoisson(mu=9, shape=(30,30,30)):

    print ("-------- Poisson test mu=%g, shape=%s ---------"%(mu,str(shape)))
    grid = N.random.poisson(mu,shape)*1./mu - 1

    # forward and backward transform
    gk = gofft(grid)
    g2 = gofftinv(gk)
    assert(N.allclose(grid,g2))

    # test power spectrum
    k,pk = powerspectrum(grid, shape)

    print ("  pk mean", pk.mean(), "should be", 1./mu)
    print ("  kmin, kmax", k.min(), k.max())
    print ("  error",pk.mean()*mu-1.)
    assert(N.abs(pk.mean()*mu-1.) < .05)
    print (" pass :)")


def testmask(mu=100, shape=(30,30,30)):
    import pylab,bins
    grid = N.random.poisson(mu,shape)*1./mu - 1

    mask = N.fromfunction(lambda x,y,z: x+y+z<45, shape)*1.

    f = N.sum(mask)*1./mask.size

    k,p1 = powerspectrum(grid, shape, zeropad=2)
    k,p2 = powerspectrum(grid*mask, shape, zeropad=2)
    k,p3 = powerspectrum(grid, shape, zeropad=None)

    print ("f",f)
    p2 /= f

    kbins = bins.logbins(k.min(),k.max(),100)
    x = kbins[1:]
    p1b = bins.binit(k,p1,kbins)
    p2b = bins.binit(k,p2,kbins)
    p3b = bins.binit(k,p3,kbins)

    pylab.plot(x,p1b)
    pylab.plot(x,p2b)
    pylab.semilogy(x,p3b)

    pylab.axhline(1./mu)
    pylab.show()


if __name__=="__main__":
    x = N.random.normal(0,1,10000)
    kx = gofft(x)
    x2 = gofftinv(kx)
    print(x2)
    assert(N.allclose(x,x2))
    print ("pass")

    #testmask()
    #sys.exit()
    testpoisson(shape=(30,30,50))
    #testpoisson(shape=(200,200))
    #testpoisson(shape=4e4)
