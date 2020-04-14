# install
cd ~/CODE/EXTERNAL/rockstar_marcel/
make with_hdf5

# edit config file ms_gadget.cfg

# start job
salloc --nodes=3 --time 01:00:00 --exclusive
ssh nodeXY.hpc.sns.ias.edu

# start server. This generates OUTBASE/auto-rockstar.cfg
./rockstar -c ms_gadget.cfg 

# start processes (use the generated auto-rockstar.cfg)
# login to all 3 nodes we have and run (REPLACE SEED!)
~/CODE/EXTERNAL/rockstar_marcel/rockstar -c /scratch/mschmittfull/lss/ms_gadget/run4/00000404-01536-500.0-wig/snap_0.6250.gadget3/auto-rockstar.cfg

# watch output on master node
