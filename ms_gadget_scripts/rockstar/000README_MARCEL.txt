# install
cd ~/CODE/EXTERNAL/rockstar_marcel/
make with_hdf5

# edit config file ms_gadget.cfg

# start job
salloc --nodes=3 --time 01:00:00 --exclusive
ssh nodeXY.hpc.sns.ias.edu

# start server. This generates OUTBASE/auto-rockstar.cfg
cd ~/CODE/EXTERNAL/rockstar_marcel
./rockstar -c 000_marcel_scripts/ms_gadget.cfg_L1500

# start processes (use the generated auto-rockstar.cfg)
# login to all 3 nodes we have and run (REPLACE SEED!)
# should run on scratch b/c faster than data
~/CODE/EXTERNAL/rockstar_marcel/rockstar -c /scratch/mschmittfull/lss/ms_gadget/run4/00000401-01536-1500.0-wig/snap_0.6250.gadget3/auto-rockstar.cfg

# watch output on master node

mv out_0.list rockstar_out_0.list

# also get parent/subhalo information
make parents
cd /scratch/mschmittfull/lss/ms_gadget/run4/00000400-01536-1500.0-wig/snap_0.6250.gadget3/
/home/mschmittfull/CODE/EXTERNAL/rockstar_marcel/util/find_parents rockstar_out_0.list 1500.0 > rockstar_out_0.list.parents

# convert to bigfile
for SEED in {0..4}; do python main_rockstar_catalog_to_bigfile.py --rockstar_halos "/scratch/mschmittfull/lss/ms_gadget/run4/0000040${SEED}-01536-500.0-wig/snap_0.6250.gadget3/rockstar_out_0.list" --max_rows 5 --include_parent_ID; done
