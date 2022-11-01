import os
import logging

def write_txt(filename, data, count=False, boundbox=False, rows=False):
    """ Write data with optional header readable by DTFE. """
    if rows:
        c = len(data)
    else:
        c = len(data[0])

    if boundbox:
        if rows:
            x, y = zip(*data)[-2:]
        else:
            x, y = data[-2:]

    with file(filename, "w") as out:
        if count:
            print >>out, c
        if boundbox:
            assert(len(x)>0)
            print >>out, x.min(), x.max(), y.min(), y.max()
            logging.debug("BB: %s",(x.min(), x.max(), y.min(), y.max()))

        if rows:
            for row in data:
                print >>out, " ".join([str(v) for v in row])
        else:
            for i in range(c):
                for j in range(len(data)):
                    print >>out, data[j][i],
                print >>out, ""

    logging.debug("wrote out %s",filename)

def tempname(dir="tmp", prefix=""):
    """ """
    if not os.path.exists(dir):
        os.mkdir(dir)
    return os.tempnam(dir, prefix)


import numpy as N

def unzip(a):
    """ """
    ncol = len(a[0])

    r = zip(*a)

    for i in range(ncol):
        r[i] = N.array(r[i])

    return r


def rank_transform1(data):
    """ Compute ranks along columns. """
    order = N.argsort(data)
    r = N.arange(len(order))
    r[order] = N.arange(len(order))
    return r

def rank_transform(data, ref=None, axis=1):
    """ """
    if ref is None:
        ref = data

    ncol = ref.shape[axis]
    ranks = []
    for i in range(ncol):
        col = ref.take(i, axis=axis).copy()
        col.sort()
        r = col.searchsorted(data.take(i, axis=axis))
        ranks.append(r)
    return N.array(ranks)


def stddev_transform(data, stats=None, axis=1):
    ncol = data.shape[axis]
    pval = []
    stats_tmp = []
    for i in range(ncol):
        y = data.take(i, axis=axis)

        if stats is None:
            mu = N.mean(y)
            sig = N.std(y)
            stats_tmp.append((mu,sig))
        else:
            mu,sig = stats[i]

        r = (y - mu) / sig

        pval.append(r)

    if stats is None:
        stats = stats_tmp
    return N.array(pval), stats



def increment_path(filename, format=".%02i"):
        """ Create a unique filename by adding an index to the path. """
        path,ext = os.path.splitext(filename)
        new_name = filename
        i = 0
        while os.path.exists(new_name):
                i += 1
                tag = format%i
                new_name = path+tag+ext
        return new_name

def dumptxt(data, savefile, **header):
    """ """
    with file(savefile, 'w') as out:
        print >>out, "#",time.asctime()
        print >>out, "# commit",githeader.gitcommit()
        for key,value in header.items():
            print >>out, "# %s %s"%(key,value)
        for i in range(len(data)):
            for j in range(len(data[i])):
                print >>out, data[i,j],
            print >>out, ""
    logging.info("Wrote data to file: ",savefile)



def test_ranks(shape=(10,2)):
    x = N.random.uniform(0,100,shape).astype(int)
    a = rank(x)
    b = rank2(x,x)
    for i in range(len(x)):
        print(i,x[i,0],a[0,i],b[0,i])


def lowerwater(z, plevels,alpha=0.99):
    """ """
    if N.max(z)==0:
        return [0]*len(plevels)
    tot = N.sum(z)
    levels = []
    for l in plevels:
        h = z.max()
        t = tot*l
        ll = 0
        if z.sum()<t:
            levels.append(0)
        while z[z>h].sum()<t:
            h*=alpha
            ll+=1
            if ll>1e6:
                print("uhoh!",ll,h,t)
                break

        levels.append(h)

    return levels

if __name__=="__main__":
    test_ranks()
