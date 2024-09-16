import pyfftw
import functools
import numpy as np
from __w90_files import CheckPoint
from time import time
from __utility import FortranFileR,iterate3dpm, real_recip_lattice, fourier_q_to_R,fourier_R_to_q
#import multiprocessing
#from itertools import islice
import gc
import os
import sys

def wigner_seitz(mp_grid,real_lattice):
    ws_search_size = np.array([1] * 3)
    dist_dim = np.prod((ws_search_size + 1) * 2 + 1)
    origin = divmod((dist_dim + 1), 2)[0] - 1
    real_metric = real_lattice.dot(real_lattice.T)
    mp_grid = np.array(mp_grid)
    irvec = []
    ndegen = []
    for n in iterate3dpm(mp_grid * ws_search_size):
        dist = []
        for i in iterate3dpm((1, 1, 1) + ws_search_size):
            ndiff = n - i * mp_grid
            dist.append(ndiff.dot(real_metric.dot(ndiff)))
        dist_min = np.min(dist)
        if abs(dist[origin] - dist_min) < 1.e-7:
            irvec.append(n)
            ndegen.append(np.sum(abs(dist - dist_min) < 1.e-7))

    return np.array(irvec), np.array(ndegen)

arg_list = sys.argv
name = arg_list[1]
seedname = name+'.chk'
chk = CheckPoint(seedname)
win_min = chk.win_min
win_max = chk.win_max
kpts = chk.num_kpts
v_matrix = np.array(chk.v_matrix)
mp_grid = chk.mp_grid
nwann = chk.num_wann
kvec = chk.kpt_latt
real_lattice = chk.real_lattice
iRvec, Ndegen = wigner_seitz(mp_grid,real_lattice)
rpts = len(iRvec)

seedname1 = name+'up.chk'
chk1 = CheckPoint(seedname1)
win_min1 = chk1.win_min
win_max1 = chk1.win_max
v_matrix1 = np.array(chk1.v_matrix)
nwannup = chk1.num_wann

seedname2 = name+'dn.chk'
chk2 = CheckPoint(seedname2)
win_min2 = chk2.win_min
win_max2 = chk2.win_max
v_matrix2 = np.array(chk2.v_matrix)
nwanndn = chk2.num_wann

diff_so = win_max - win_min
diffup = win_max1 - win_min1
diffdn = win_max2 - win_min2
diff_sp = diffup + diffdn
bloch = []

for i in range(kpts):
    os.chdir('../wannier/'+name+'.save/')
    wa = np.load("wfc{i}.npy".format(i=i+1))
    #wa = np.load("/home/robin/gilbertdamping/examples/Fe/wannier/Fe.save/wfc{i}.npy".format(i=i+1))
    os.chdir('../../spinpol/wannierup/'+name+'.save/')
    wu = np.load("wfcup{i}.npy".format(i=i+1))
    #wu = np.load("/home/robin/gilbertdamping/examples/Fe/spinpol/wannierup/Fe.save/wfcup{i}.npy".format(i=i+1))
    #wd = np.load("/home/robin/gilbertdamping/examples/Fe/spinpol/wannierup/Fe.save/wfcdw{i}.npy".format(i=i+1))
    wd = np.load("wfcdw{i}.npy".format(i=i+1))
    os.chdir('../../../damping/')
    bloch_k = np.zeros((diff_sp[i],diff_so[i]),complex)
    for j in range(diff_sp[i]):
        for k in range(diff_so[i]):
            lenth = int(len(wa[k+win_min[i]]))
            if j<int(diffup[i]):
                bloch_k[j][k]= np.vdot(wu[j+win_min1[i]],wa[k+win_min[i]][0:int(lenth/2)])
            else:
                bloch_k[j][k]= np.vdot(wd[j-diffup[i]+ win_min2[i]],wa[k+win_min[i]][int(lenth/2):lenth])
    bloch.append(bloch_k.tolist())

k_mult =  np.zeros((kpts,nwann,nwann),complex)
for i in range(kpts):
    zeroup = np.zeros((nwannup,diffup[i]),complex)
    zerodn = np.zeros((nwanndn,diffdn[i]),complex)
    v_sp = np.block([[v_matrix1[i],zeroup],[zerodn,v_matrix2[i]]])
    bloch_k = np.array(bloch[i])
    k_mult[i] = ((v_sp.conj())@(bloch_k))@(v_matrix[i].T)


kpt_mp_grid = [
            tuple(k) for k in np.array(np.round(chk.kpt_latt * np.array(chk.mp_grid)[None, :]), dtype=int) % chk.mp_grid
        ]
if (0, 0, 0) not in kpt_mp_grid:
        raise ValueError(
                "the grid of k-points read from .chk file is not Gamma-centered. Please, use Gamma-centered grids in the ab initio calculation"
                )

#npar=multiprocessing.cpu_count()
fourier_q_to_R_loc = functools.partial(
           fourier_q_to_R,
            mp_grid=chk.mp_grid,
            kpt_mp_grid=kpt_mp_grid,
            iRvec=iRvec,
            ndegen=Ndegen,
            numthreads=1,
            fft='fftw')

print('starting fft ')
timeFFT = 0
#GGq = chk.get_GG_q(eig,0.0,0.0)
t0 = time()
wann_t = fourier_q_to_R_loc(k_mult)
#print(wann_t[0])
timeFFT += time() - t0
print(timeFFT)

#print(wann_t[0])
np.save("wanniertrans.npy",wann_t)
