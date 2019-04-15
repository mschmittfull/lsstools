#!/bin/bash

# Examples:
# run.sh python main_quick_calc_Pk.py "{'sim_seed': 300}"
# run.sh mpiexec -n 3 python main_quick_calc_Pk.py "{'sim_seed': 300}"
# run.sh srun -n 3 python main_quick_calc_Pk.py "{'sim_seed': 300}"

#set -x

# Code file is $2 or $5

nbkit_version=0.3.7

if [[ "$1" =~ ^(main_quick_plot_avg_Pk_v0dot5.py)$ ]]; then
  echo "Do not load any env to run $1"
  python "$@"
  exit 0
fi

echo "$0: Run with nbkit_version=$nbkit_version"
sleep 1

if [ "$MS_HOST" == "MBPmsl15" ] ; then
  if [ $nbkit_version == 0.3 ] ; then
    envname="nbodykit-0.3-env"
  elif [ $nbkit_version == 0.3.4 ] ; then
    envname="nbodykit-0.3.4-env"
  elif [ $nbkit_version == 0.3.7 ] ; then
    envname="nbodykit-0.3.7-env"
  else
    envname="nbodykit-0.1_for_psiRec"
  fi

elif [ "$MS_HOST" == "helios" ]
then
  if [ $nbkit_version == 0.3 ] ; then
    envname="nbodykit-0.3-env"
  elif [ $nbkit_version == 0.3.4 ] ; then
    envname="nbodykit-0.3.4-env"
  elif [ $nbkit_version == 0.3.7 ] ; then
    envname="nbodykit-0.3.7-env"
  else
    envname="nbodykit-0.1.5_with_marcel_edits"
  fi
  tmp_hdf5_use_file_locking=$HDF5_USE_FILE_LOCKING
  export HDF5_USE_FILE_LOCKING=FALSE

else
    echo "$0: MS_HOST variable not valid: $MS_HOST"
    exit 1
fi


# activate environment with nbodykit
source activate $envname
echo "$0: Activated virtualenv $envname"

# run code, copy arguments
"$@"

# change environment back
source deactivate
echo "$0: Deactivated virtualenv $envname"

if [ "$MS_HOST" == "helios" ]
then
  export HDF5_USE_FILE_LOCKING=$tmp_hdf5_use_file_locking
fi

exit 0

