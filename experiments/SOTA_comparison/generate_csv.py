import numpy as np
from glob import glob
import pickle
import os

maindir = 'csv_data/'
filelist = sorted(glob(maindir + '*.h5'))

vgg_dense_idx = [96, 102, 108]
deepdta_dense_idx = [14, 16, 18, 20]

for f in filelist:
	outname = maindir + f.split(sep=os.sep)[-1][:-3]
    with open(f, "rb") as a:
        m = pickle.load(a)
    np.savetxt(outname + '_096.csv', m[96],  delimiter=',', fmt='%.5e')
    np.savetxt(outname + '_102.csv', m[102], delimiter=',', fmt='%.5e')
    np.savetxt(outname + '_108.csv', m[108], delimiter=',', fmt='%.5e')
