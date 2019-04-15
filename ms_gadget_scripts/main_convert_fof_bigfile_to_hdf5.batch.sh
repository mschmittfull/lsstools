#!/bin/bash -l

set -x
export HDF5_USE_FILE_LOCKING=FALSE


MinMass=13.8
MaxMass=15.1
RSD=1

source activate nbodykit-0.3.7-env

myscript=main_convert_fof_bigfile_to_hdf5.py

for SimSeed in 400 401 402 403 404
do
    python $myscript --SimSeed=$SimSeed --MinMass=$MinMass --MaxMass=$MaxMass --RSD=$RSD
done
        
source deactivate                

