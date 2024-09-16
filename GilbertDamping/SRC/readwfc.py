from scipy.io import FortranFile
import numpy as np
import sys

arg_list = sys.argv
print(len(arg_list))
print(arg_list[2])
nk = int(arg_list[2])
seedname = arg_list[1]

for i in range(nk):
    f = FortranFile(seedname+'{i}.dat'.format(i=i+1),'r')
    a = f.read_reals('i4')
    a = f.read_ints('i4')
    nbnd = a[3]
    #print(nbnd)
    ng = a[1]
    #print(ng)
    nspin = a[2]
    #print(nspin)
    length = int(nspin*ng)
    #print(length)
    arr = np.zeros((nbnd,length),complex)
    a = f.read_reals('f8')
    a = f.read_ints('i4')
    for j in range(nbnd):
        a = f.read_reals('f8')
        arr[j] = a[::2]+1j*a[1::2]
    #print(arr.shape)
    np.save(seedname+'{i}.npy'.format(i=i+1),arr)
