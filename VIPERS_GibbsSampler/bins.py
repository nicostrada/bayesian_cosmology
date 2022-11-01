import numpy as N

def binit(x, y, bands, count=False):
    """ """
    out = N.zeros(len(bands)-1,dtype='d')
    n = out.copy()
    sig = out.copy()
    for i in range(len(out)):
        ii = (x >= bands[i]) & (x < bands[i+1])           # true/false array
        sub = y[ii]                                       # array that selects the true values
        if sub.size > 0:
            out[i] = N.mean(sub)
            sig[i] = N.std(sub)
            n[i] += sub.size

    if count:
        return out,sig,n
    return out

def binit_lnorm(x, y, bands):
    """ """
    norm = x*(x+1)
    out = N.zeros(len(bands)-1,dtype='d')
    for i in range(len(out)):
        ii = N.where(x >= bands[i])
        jj = N.where(x[ii] < bands[i+1])
        sub = y[ii][jj]
        n = norm[ii][jj]
        if sub.size > 0:
            out[i] = N.sum(sub*n)/N.sum(n)

    return out


def logbins(min,max, n, integer = False):
    """integer log bins"""
    l = N.exp(N.arange(N.log(min),N.log(max),N.log(max/min)/n))
    if not integer:
        return l

    d = {}
    for a in l:
        d[int(a)] = 1
    out = d.keys()
    out.sort()
    return N.array(out)



def thetabins(f="cache/cross.cor",n=20):

    theta = []
    for line in file(f):
        if line.startswith("#"): continue
        theta.append(float(line.split()[0]))
    theta = N.array(theta)
    theta.sort()

    i = N.arange(len(theta))
    bins = logbins(1,len(i),n,integer=True)

    x = binit(i,theta,bins)

    if x[0] != theta[0]:
        x = N.concatenate([[theta[0]],x])

    return x


if __name__=="__main__":
    test()
