import pyfftw
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import numpy as np
import functools
from __utility import FortranFileR,iterate3dpm, real_recip_lattice, fourier_q_to_R,fourier_R_to_q
from numpy import linalg as LA
from __w90_files import EIG, MMN, CheckPoint, SPN, UHU, SIU, SHU
import multiprocessing
from time import time
from itertools import islice
import gc
import sys

arg_list = sys.argv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

inputfile = arg_list[1]
#name = arg_list[1]
file = open(inputfile,"r")
inputparam = file.readlines()
file.close()

name = inputparam[0].split(" ")[1]
T = float(inputparam[1].split(" ")[1])
eta = float(inputparam[2].split(" ")[1])
kgrid = int(inputparam[3].split(" ")[1])
m = float(inputparam[4].split(" ")[1])
ef = float(inputparam[5].split(" ")[1])


start = time()
seedname = name+'.chk'
chk = CheckPoint(seedname)
real_lattice,recip_lattice = real_recip_lattice(chk.real_lattice,chk.recip_lattice)
mp_grid = chk.mp_grid
kpt_latt = chk.kpt_latt
nwann = np.zeros(1)
nwann[0] = chk.num_wann 
nbnd = np.zeros(1)
nbnd[0] = chk.num_bands
kpts = np.zeros(1)
kpts[0] = chk.num_kpts
seedname = name+'up.chk'
chk1 = CheckPoint(seedname)
    
seedname = name+'dn.chk'
chk2 = CheckPoint(seedname)
    
seedname = name+'.eig'
eig = EIG(seedname)
    
seedname = name+'up.eig'
eig1 = EIG(seedname)
    
seedname = name+'dn.eig'
eig2 = EIG(seedname)
    
seedname = name+'.spn'
spn = SPN(seedname)

HHq = chk.get_HH_q(eig)
SSq = chk.get_SS_q(spn)
HHq1 = chk1.get_HH_q(eig1)
HHq2 = chk2.get_HH_q(eig2)

del eig
del eig1
del eig2
del spn


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

iRvec, Ndegen = wigner_seitz(mp_grid,real_lattice)

for i,nr in enumerate(iRvec):
    if nr[0]==0 and nr[1] == 0 and nr[2] == 0:
        index=i

nRvec0 = len(iRvec)
kpt_mp_grid = [
            tuple(k) for k in np.array(np.round(kpt_latt * np.array(mp_grid)[None, :]), dtype=int) % mp_grid
        ]
if (0, 0, 0) not in kpt_mp_grid:
        raise ValueError(
                "the grid of k-points read from .chk file is not Gamma-centered. Please, use Gamma-centered grids in the ab initio calculation"
                )

################### OPERATORS IN WANNIER BASIS ############################
npar=1
fourier_q_to_R_loc = functools.partial(
           fourier_q_to_R,
            mp_grid=mp_grid,
            kpt_mp_grid=kpt_mp_grid,
            iRvec=iRvec,
            ndegen=Ndegen,
            numthreads=npar,
            fft='fftw')


Ham_R = fourier_q_to_R_loc(HHq)
SS_R = fourier_q_to_R_loc(SSq)
Ham_R1 = fourier_q_to_R_loc(HHq1)
Ham_R2 = fourier_q_to_R_loc(HHq2)


zero = np.zeros((int(int(nwann[0])/2),int(int(nwann[0])/2)),complex)
H_sp = np.zeros((nRvec0,int(nwann[0]),int(nwann[0])),complex)
for i in range(nRvec0):
    H_sp[i] = np.block([[Ham_R1[i],zero],[zero,Ham_R2[i]]])

wann_t = np.load("wanniertrans.npy")

H_spt = np.zeros((nRvec0,int(nwann[0]),int(nwann[0])),complex)
for i in range(nRvec0):
    H_spt[i] = ((wann_t[index].conj().T)@(H_sp[i]))@(wann_t[index])


H_so = Ham_R - H_spt
np.save("hso.npy",H_so[index])

del wann_t
del H_spt
del H_sp
del Ham_R1
del Ham_R2

########### HAMILTONIAN GAUGE #################
def ham_gauge(ham):
    w,v = LA.eig(ham)
    return v

############### INTERPOLATION BACK TO k GRID###############
def get_K_list(div):
    dK = 1./div
    t0 = time()
    npts = np.prod(div)
    K_arr = np.zeros((npts,3))
    init = 0
    for i in range(div[0]):
        for j in range(div[1]):
            for k in range(div[2]):
                K_arr[init][0] = dK[0]*i
                K_arr[init][1] = dK[1]*j
                K_arr[init][2] = dK[2]*k
                init += 1
    return K_arr

#kgrid = int(arg_list[4])
div = np.array([kgrid,kgrid,kgrid])
K_int_list = get_K_list(div)
k_int = [
            tuple(k) for k in np.array(np.round(K_int_list * np.array(div)[None, :]), dtype=int) % div
        ]
if (0, 0, 0) not in k_int:
        raise ValueError(
                "the grid of k-points read from .chk file is not Gamma-centered. Please, use Gamma-centered grids in the ab initio calculation"
            )
fourier_R_to_q_loc = functools.partial(
           fourier_R_to_q,
            mp_grid1=div,
            kpt_mp_grid1=k_int,
            iRvec1=iRvec,
            numthreads=npar,
            fft='fftw')

######### HAM IFFT ######
timeFFT = 0
Ham_k = fourier_R_to_q_loc(Ham_R)

##### TORQUE MATRIX ELEMENTS ######
def commu(A,B):
    return np.matmul(A,B) - np.matmul(B,A)

###### Tx = [sigma_x,H_so] #####
Tx = np.zeros((nRvec0,int(nwann[0]),int(nwann[0])),complex)
for i in range(nRvec0):
    Tx[i] = commu(SS_R[i,:,:,0],H_so[index])

###### Ty = [sigma_y,H_so] #####
Ty = np.zeros((nRvec0,int(nwann[0]),int(nwann[0])),complex)
for i in range(nRvec0):
    Ty[i] = commu(SS_R[i,:,:,1],H_so[index])

###### Tz = [sigma_z,H_so] #####
Tz = np.zeros((nRvec0,int(nwann[0]),int(nwann[0])),complex)
for i in range(nRvec0):
    Tz[i] = commu(SS_R[i,:,:,2],H_so[index])

######## factors########
g = 2.0
#m = float(arg_list[5])
fac = g/(m*np.pi*np.prod(div))
kb = 8.6173E-5
#T = float(arg_list[2])
beta = 1./(kb*T)
#ef = float(arg_list[6]) ###inev
sigma = kb*T
h = sigma
energy = ef - (size//2)*h +rank*h
#eta = float(arg_list[3])    ### eV
ninter = np.prod(div)

def fermi(beta,ener,ef):
    return beta*np.exp(beta*(ener-ef))/(1+np.exp(beta*(ener-ef)))**2

def trapezoidal(h, f, n):
    integral = -(f[0] + f[n-1])/2.0
    for i in range(n):
        integral = integral + f[i]
    integral = integral* h
    return integral

############ CALCULATION OF TRACE xx ##########
alpha_xx = 0.0
Txk = fourier_R_to_q_loc(Tx)
I = np.identity(18,complex)
for i in range(ninter):
    umat = ham_gauge(Ham_k[i])
    Hmat = (umat.T.conj())@(Ham_k[i])@umat
    GGk = np.linalg.inv((energy+1j*eta)*I-Hmat)
    GGk = np.imag(GGk)
    Tmat = (umat.T.conj())@(Txk[i])@umat
    fin_arr1 = np.matmul(Tmat,GGk)
    fin1 = np.matmul((Tmat.T.conj()),GGk)
    fin_arr = np.matmul(fin_arr1,fin1)
    trace = np.trace(fin_arr,dtype=complex)
    alpha_xx += trace
    del umat,Hmat,GGk,Tmat,fin_arr1,fin1,fin_arr,trace

del Txk

alpha_yy=0.0
Tyk = fourier_R_to_q_loc(Ty)
for i in range(ninter):
    umat = ham_gauge(Ham_k[i])
    Hmat = (umat.T.conj())@(Ham_k[i])@umat
    GGk = np.linalg.inv((energy+1j*eta)*I-Hmat)
    GGk = np.imag(GGk)
    Tmat = (umat.T.conj())@(Tyk[i])@umat
    fin_arr1 = np.matmul(Tmat,GGk)
    fin1 = np.matmul((Tmat.T.conj()),GGk)
    fin_arr = np.matmul(fin_arr1,fin1)
    trace = np.trace(fin_arr,dtype=complex)
    alpha_yy += trace
    del umat,Hmat,GGk,Tmat,fin_arr1,fin1,fin_arr,trace

del Tyk

alpha_xx = alpha_xx*fermi(beta,energy,ef)
alpha_yy = alpha_yy*fermi(beta,energy,ef)

end = time()
timet = end-start
########### final alpha = alphaxx + alphayy #############
gilbx = np.zeros(1,complex)
gilby = np.zeros(1,complex)
timearr = np.zeros(1)
timearr[0] = timet
gilbx[0] = alpha_xx*fac
gilby[0] = alpha_yy*fac

comm.Barrier()
if rank == 0:
    xgather = np.zeros(size,complex)
    ygather = np.zeros(size,complex)
else:
    xgather = None
    ygather = None

comm.Gather(gilbx,xgather,root=0)
comm.Gather(gilby,ygather,root=0)

if rank == 0:
    integralx = trapezoidal(h,xgather,size)
    integraly = trapezoidal(h,ygather,size)
    print("The alpha xx, alpha yy and alpha are:",integralx/4.0,integraly/4.0,(integralx+integraly)/4.0)

