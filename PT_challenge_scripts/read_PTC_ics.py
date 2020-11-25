from __future__ import print_function, division
import numpy as np

#from nbodykit.lab import *
#from nbodykit.source.catalog import HDFCatalog
#from nbodykit import CurrentMPIComm

#from lsstools.nbkit03_utils import get_cmean, get_cstats_string



def main():
    """
    Read PT challenge ICs
    """

    ng = 512
    fname = '/data/mschmittfull/lss/takahiro/PTChallenge/R001_delta_conf_ng%d.dat' % ng
    print('Will read %s' % fname)

    ng3d = ng**3
    f = open(fname, "rb")
    delta_lin = np.fromfile(f, np.float32, ng3d)
    f.close()
    delta_lin = delta_lin.reshape(ng,ng,ng)

    print('Done reading deltalin')


    if False:
        # save to bigfile
        out_fname = '%s_PtcleDensity_z%d_Ng%d' % (fname, int(catmesh.attrs['Redshift']), Nmesh)
        if comm.rank==0:
            print('Writing to %s' % out_fname)
        catmesh.save(out_fname)
        if comm.rank==0:
            print('Wrote %s' % out_fname)

if __name__ == '__main__':
    main()


