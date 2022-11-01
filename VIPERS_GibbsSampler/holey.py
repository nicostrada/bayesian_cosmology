""" Mask processor """
import logging
import numpy as N
from   scipy.spatial   import cKDTree
import time
from   matplotlib.path import Path

def distance(a,b):
    """The haversine formula to compute angular separation on the sphere"""
    x1,y1 = a
    x2,y2 = b

    dx = (x1-x2)*N.pi/180
    dy = (y1-y2)*N.pi/180
    a = N.sin(dx/2.)**2
    b = N.sin(dy/2.)**2
    c = N.sqrt(b + N.cos(y1*N.pi/180)*N.cos(y2*N.pi/180)*a)
    return 2*N.arcsin(c)*180/N.pi

class holey:

    def __init__(self, regionfile=None):
        """ """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mask = {}
        if not regionfile is None:
            self.loadregion(regionfile)


    def loadregion(self, path):
        tag = 'None'
        mask = {}
        linenu = 0

        f = open(path, 'r')
        #for line in file(path):
        for line in f:
            linenu += 1
            a = line.find("(")
            if a<0: continue
            b = line.find(")")
            if b<0: continue

            label = line[:a]
            label = label.lower()

            if label.endswith("circle"):
                tag = 'circle'
            elif label.endswith("box"):
                tag = 'box'
            elif label.endswith("polygon"):
                tag = 'polygon'
            else:
                print("unknown label!",label)
                continue

            coords = line[a+1:b]
            coords = coords.replace(","," ")

            try:
                c = [float(v) for v in coords.split()]
            except:
                print("can't parse line %i"%linenu)
                print(line)
                print(coords)
                raise
            #if not mask.has_key(tag):
            #    mask[tag] = []
            if not tag in mask:
                mask[tag] = []

            mask[tag].append(c)
        self.mask = mask


    def getRadius(self, tag, c):
        """ """
        if tag=='circle':
            x,y,r = c
            return x,y,r
        elif tag=='box':
            x,y,dx,dy = c
            return x,y,N.sqrt(dx**2+dy**2)
        elif tag=='polygon':
            poly = c.reshape((len(c)//2,2))
            a = poly.min(axis=0)
            b = poly.max(axis=0)
            r = N.sqrt(N.sum((b-a)**2))/2
            x,y = (a+b)/2.

            return x,y,r


    def checkinside(self, tag, maskcoord, cc):
        """ """
        tx,ty = N.transpose(cc)
        if tag=='circle':
            x,y,r = maskcoord
            return distance((x,y),(tx,ty))<r
        elif tag=='box':
            x,y,dx,dy = maskcoord
            hx = dx/2./N.cos(y*N.pi/180)
            hy = dy/2.
            return N.logical_and(N.abs(tx-x)<hx, N.abs(ty-y)<hy)
        elif tag=='polygon':
            poly = maskcoord.reshape((len(maskcoord)//2,2))
            path = Path(poly)
            return path.contains_points(cc)
        else:
            print("oh hell")


    def check(self,lon=None,lat=None, density=5e6):
        """ """
        t0 = time.time()

        if not lon is None:
            data = N.transpose((lon,lat))
            assert len(data)>0
            tree = cKDTree(data)
            coords = N.transpose((lon,lat))

            self.tree = tree
            self.coords = coords
        else:
            tree = self.tree
            coords = self.coords

        counter = 0
        inside = []
        for tag in ['circle','box','polygon']:
            #if not self.mask.has_key(tag): continue
            if not tag in self.mask: continue

            for maskcoord in self.mask[tag]:
                maskcoord = N.array(maskcoord)
                counter += 1
                x,y,r = self.getRadius(tag, maskcoord)
                cos = N.cos(y*N.pi/180)

                nmax = int(max(100,r**2*density))

                matches = tree.query_ball_point((x,y), r/cos*1.1)
                cc = coords[matches,:]

                ii = self.checkinside(tag,maskcoord,cc)

                inside.append(N.array(matches)[ii])

        inside = N.concatenate(inside).astype(int)

        s = N.zeros(len(coords))
        s[inside] = 1

        self.logger.debug("Holey time (n=%i, m=%i): %g"%(len(s),counter,time.time()-t0))

        return s>0


def test(n=1e5):
    import pylab
    x = N.random.uniform(0.,1.,int(n))
    y = N.random.uniform(0.,1.,int(n))
    mask = {'circle':[(.50,.50,.10),
                      (.6,.8,.2)],
            'box':[(0.5,0.1,.2,.1)],
            'polygon':[(0.2,0.1,0.3,0.1,0.15,0.8)]}

    H = holey()
    H.mask = mask
    ii = H.check(x,y)==False

    #pylab.plot(x,y,",")

    # This is for plotting on Jupyter
    pylab.plot(x[ii],y[ii],",")
    pylab.show()

    # This is for printing a file on DORAEMON
    #chain = N.vstack((x[ii],y[ii])).T
    #N.savetxt('holey_prove.txt',chain)


def testw1(n=1e6):
    import pylab

    x = N.random.uniform(30.,39.,int(n))
    y = N.random.uniform(-4.,-6.,int(n))

    H1= holey('/home/nicolas/MEGAsync/thesis/VIPERS/data/vipers_photo/photo_W1.reg')
    H2= holey('/home/nicolas/MEGAsync/thesis/VIPERS/data/vipers_spectromask/vipers_W1.reg')

    #H1= holey('/home/nestrada/VIPERS_mask/photo_W1.reg')
    #H2= holey('/home/nestrada/VIPERS_mask/vipers_W1.reg')

    ii = H1.check(x,y,density=3e6)==False
    jj = H2.check(x,y,density=3e6)==True

    kk = N.all([ii,jj],axis=0)

    pylab.figure(figsize=(20,12))
    pylab.plot(x[kk],y[kk],",")
    pylab.show()

    # This is for printing a file on DORAEMON
    #chain = N.vstack((x[kk],y[kk])).T
    #N.savetxt('holey_provew1.txt',chain)


def testw4(n=1e6):
    import pylab

    x = N.random.uniform(330.,336.,int(n))
    y = N.random.uniform(0.8,2.5,int(n))

    H1= holey('/users/nestrada/MEGA/MEGAsync/thesis/VIPERS/data/vipers_photo/photo_W4.reg')
    H2= holey('/users/nestrada/MEGA/MEGAsync/thesis/VIPERS/data/vipers_spectromask/vipers_W4.reg')

    ii = H1.check(x,y,density=3e6)==False
    jj = H2.check(x,y,density=3e6)==True

    kk = N.all([ii,jj],axis=0)

    pylab.figure(figsize=(20,12))
    pylab.plot(x[kk],y[kk],",")
    pylab.show()

    # This is for printing a file on DORAEMON
    #chain = N.vstack((x[kk],y[kk])).T
    #N.savetxt('holey_provew4.txt',chain)


def spheretest(n=1e7):
    import pylab
    lon = N.random.uniform(-180,180,n)
    z = N.random.uniform(0,1,n)
    lat = 90-N.arccos(z)*180/N.pi

    H = holey()

    i = 0
    print("go")
    for l in N.arange(5,100,20):
        i+=1
        mask = {'circle':[]}
        mask['circle'].append((170,l,2))

        H.mask = mask

        if i==1:
            ii = H.check(lon,lat)
        else:
            ii = H.check()

        pylab.plot(lon[ii],lat[ii],",")

        print(l,len(lon[ii]))


    for l in N.arange(5,100,20):
        mask = {'box':[]}
        mask['box'].append((175,l,2,2))

        H.mask = mask

        ii = H.check()

        pylab.plot(lon[ii],lat[ii],",")

        print(l,len(lon[ii]))

    pylab.show()


if __name__=="__main__":
    #spheretest()
    test()
    #testw1()
    #testw4()
