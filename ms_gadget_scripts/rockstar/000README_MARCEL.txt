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
# should run on scratch b/c faster than data
~/CODE/EXTERNAL/rockstar_marcel/rockstar -c /data/mschmittfull/lss/ms_gadget/run4/00000404-01536-500.0-wig/snap_0.6250.gadget3/auto-rockstar.cfg

# watch output on master node

mv out_0.list rockstar_out_0.list

# also get parent/subhalo information
make parents
cd /scratch/mschmittfull/lss/ms_gadget/run4/00000400-01536-500.0-wig/snap_0.6250.gadget3/
/home/mschmittfull/CODE/EXTERNAL/rockstar_marcel/util/find_parents rockstar_out_0.list 500.0 > rockstar_out_0.list.parents
