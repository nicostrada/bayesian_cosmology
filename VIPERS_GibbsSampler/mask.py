""" Routines to deal with the survey geometry.
"""
import logging
import numpy as np
import sphere
import holey
import healpy
import utils

class Mask:
    """ The survey mask """

    fudge = 1. # 1 deg

    def __init__(self, center=None, radius=None, verbose=True, usevenice=False, venicecmd="venice"):
        """ """
        self.logger = logging.getLogger(self.__class__.__name__)
        #if verbose:
        #    logging.basicConfig(level=logging.DEBUG)

        self.verbose   = verbose
        self.center    = center
        self.radius    = radius
        self.usevenice = usevenice
        self.venicecmd = venicecmd

        self.field_masks = []
        self.hole_masks  = []
        self.field_paths = []
        self.hole_paths  = []

    def add_mask_file(self, maskfile, holes=False, format="holey"):
        """
        Inputs
        maskfile - string
        holes - bool
        format - holey or mangle
        """
        if self.usevenice:
            m = None
        else:
            format = format.lower()
            if format.startswith("h"):
                m = holey.holey(maskfile)
            elif format.startswith("m"):
                m = mangle.Mangle(maskfile)
            else:
                self.logger.critical("WARNING~~~~~~~ unknown mask format! %s", format)

        if holes:
            self.hole_masks.append(m)
            self.hole_paths.append(maskfile)
        else:
            self.field_masks.append(m)
            self.field_paths.append(maskfile)
        self.logger.debug("loaded %s",maskfile)
        self.logger.debug("format %s",format)

    def check_venice(self, ra, dec):
        """ """
        import os
        venice_input  = os.tempnam()
        venice_output = os.tempnam()

        ind = np.arange(len(ra))
        utils.write_txt(venice_input, (ind,ra,dec))

        venice_args = {'venice': self.venicecmd}
        for maski in range(len(self.field_paths)):
            venice_args['mask'] = self.field_paths[maski]
            venice_args['cat'] = venice_input
            venice_args['outfile'] = venice_output
            cmd = "{venice} -m {mask} -cat {cat} -xcol 2 -ycol 3 -f inside > {outfile}".format(**venice_args)
            self.logger.debug(cmd)
            os.system(cmd)
            venice_input, venice_output = venice_output, venice_input

        # apply photometric mask
        for maski in range(len(self.hole_paths)):
            venice_args['mask'] = self.hole_paths[maski]
            venice_args['cat'] = venice_input
            venice_args['outfile'] = venice_output
            cmd = "{venice} -m {mask} -cat {cat} -xcol 2 -ycol 3 -f outside{outfile}".format(**venice_args)
            self.logger.debug(cmd)
            os.system(cmd)
            venice_input, venice_output = venice_output, venice_input


        sel = np.loadtxt(venice_input,unpack=True)[0]

        if len(sel) == 0:
            self.logger.warning("No objects found inside the mask!")

        sel = sel.astype(int)
        out = np.zeros(len(ra), dtype=bool)
        out[sel] = True

        os.remove(venice_input)
        os.remove(venice_output)

        return out

    def check_inside(self, ra, dec):
        """ Check if points are contained in the mask and outside the holes.

        Inputs
        ra - float array of longitude angles (Degrees)
        dec - float array of latitude angles (Degrees)

        Outputs
        bool array - Boolean values indicating inside (True) and outside (False)

        """
        if self.usevenice:
            return self.check_venice(ra, dec)

        ra[ra<0] += 360

        field_sel = True
        for m in self.field_masks:
            ii = m.check(ra.tolist(),dec.tolist())
            field_sel = np.logical_and(field_sel, ii)

        rat = ra[field_sel]
        dect = dec[field_sel]

        #assert len(rat) > 0

        hole_sel = None
        for m in self.hole_masks:
            ii = m.check(rat,dect)
            if hole_sel is not None:
                hole_sel = np.logical_or(hole_sel, ii)
            else:
                hole_sel = ii

        if hole_sel is not None:
            hole_sel = np.logical_not(hole_sel)

        # logging.debug("hole_sel: %s",hole_sel)

        # do some index transformations
        ind = np.arange(len(ra))[field_sel]
        if hole_sel is not None:
            ind = ind[hole_sel]

        sel = np.zeros(len(ra),dtype='bool')
        sel[ind] = True

        return sel

    def generate_grid(self):
        """ """
        pass

    def pixelize(self,nside=512):
        """ """
        pix = np.arange(12*nside**2)
        theta,phi = healpy.pix2ang(nside, pix)
        ra = 180/np.pi*phi
        dec = 90 - 180./np.pi*theta

        sel = self.check_inside(ra,dec)
        map = np.zeros(len(pix))
        map[sel] = 1
        return map



    def determine_bounds(self, nside=256):
        """ """
        map = self.pixelize(nside=nside)
        sel = map>0
        pix = np.arange(12*nside**2)[sel]

        xyz = healpy.pix2vec(nside,pix)
        xyz = np.array(xyz)

        center = np.sum(xyz,axis=1)

        norm = np.sum(xyz**2,axis=0)**.5 * np.sum(center**2)**.5
        costheta = np.min(np.dot(center,xyz)/norm)
        radius = np.arccos(costheta)*180/np.pi

        theta,phi = healpy.vec2dir(*center)
        center_radec = 180/np.pi*phi, 90-180/np.pi*theta

        self.logger.debug("> center %s",center_radec)
        self.logger.debug("> radius %s",radius)

        self.center = center_radec
        self.radius = radius

        return center_radec, radius


    def random_sample(self, n=1000, batchfrac = 0.2, density=None):
        """ Draw random points from the mask area.

        Inputs
        n - int minimum number of samples to generate.
        batchfrac - On each iteration n*batchfrac points are tested.
        density - number density (per square degree)

        Outputs
        longitude
        latitude
        area

        """

        if self.center is None:
            self.determine_bounds()

        if density is not None:
            batchfrac = 1
            # area = sphere.cap_area(self.radius + self.fudge)
            area = 2*np.pi*(1-np.cos((self.radius+self.fudge)*np.pi/180))
            n = np.round(density * area * sphere.steradian_to_degree)

        self.logger.debug("n %s",n)

        nstep = int(n*batchfrac)
        assert(nstep>0)

        lon = []
        lat = []
        count = 0
        count_tot = 0

        while count<n:
            ra,dec = sphere.sample_cap(nstep, lon=self.center[0], lat=self.center[1], theta=self.radius + self.fudge)
            count_tot += len(ra)
            assert len(ra)>0
            assert len(dec)>0
            sel = self.check_inside(ra,dec)
            count += np.sum(sel)
            lon.append(ra[sel])
            lat.append(dec[sel])

            if density is not None:
                break

        lon = np.concatenate(lon)
        lat = np.concatenate(lat)

        area = 2*np.pi*(1-np.cos((self.radius+self.fudge)*np.pi/180))
#         print("cap area", area)
        eff_area = count*1./count_tot * area * sphere.steradian_to_degree

        self.logger.debug("> Drew %i randoms (tested %i).  Mask area = %f deg sqr"%(count,count_tot,eff_area))

        return lon, lat, eff_area

if __name__=="__main__":
    M = Mask(center=(30,45),radius=45.)
    ra,dec,area = M.random_sample(n=1e6)
    import pylab
    pylab.plot(ra,dec,",",alpha=0.2)
    pylab.show()
