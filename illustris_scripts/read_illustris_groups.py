import illustris_python as il

#basePath = '/data/mschmittfull/lss/IllustrisTNG/L205n2500TNG/output/'
basePath = '/Users/mschmittfull/scratch_data/lss/IllustrisTNG/Illustris-3/output'
fields = ['SubhaloMass','SubhaloMassType','SubhaloSFRinRad']
subhalos = il.groupcat.loadSubhalos(basePath,67,fields=fields)
print subhalos
print subhalos.keys()
