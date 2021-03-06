#!/bin/bash -l

set -x
export HDF5_USE_FILE_LOCKING=FALSE


MinMass=10.8
MaxMass=11.8
RSD=1

source activate nbodykit-0.3.7-env

myscript=main_convert_fof_bigfile_to_hdf5.py

#for SimSeed in 400 401 402 403 404
for SimSeed in 403
do
    python $myscript --SimSeed=$SimSeed --MinMass=$MinMass --MaxMass=$MaxMass --RSD=$RSD
done
        
source deactivate                

