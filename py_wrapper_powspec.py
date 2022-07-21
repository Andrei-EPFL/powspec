from ctypes import *
import numpy as np
from cffi import FFI


class Data(Structure):
    _fields_ = [
        ("x", c_double * 3),
        ("w", c_double)
    ]


class Cata(Structure):
    _fields_ = [
        ("num", c_int), # number of data catalogs to be read
        ("data", POINTER(POINTER(Data))),
        ("rand", POINTER(POINTER(Data))),
        ("ndata", POINTER(c_size_t)),    
        ("nrand", POINTER(c_size_t)),
        ("wdata", POINTER(c_double)),
        ("wrand", POINTER(c_double)),
        ("alpha", POINTER(c_double)),
        ("shot", POINTER(c_double)),
        ("norm", POINTER(c_double)),
    ]


def compute_catalog():
    x, y, z = np.loadtxt("/global/homes/a/avariu/desi_avariu/FastPM_SLICS/slics_galaxy_rsd/1.041halo.dat_LOS996.gcat", usecols=(0,1,2), unpack=True, dtype=np.float32)
    # x, y, z = (x + 505) % 505, (y + 505) % 505, (z + 505) % 505
    print(len(x))
    range_ = (0 <= x) & ( x < 505) & (0 <= y) & ( y < 505) & (0 <= z) & ( z < 505)
    print(np.max(x))
    print(np.max(y))
    print(np.max(z))
    x, y, z = x[range_], y[range_], z[range_]
    
    print(len(x))
    num = 1
    ndata = len(x)
    nrand = 0
    wdata = ndata
    wrand = 0.
    alpha = 0.
    shot = 0. # 1/n n(density of the box) n = len(x) / (505)**3
    norm = 0.


    data_instance = Data * ndata
    data_array = data_instance()

    cata_instance = Cata
    ca = cata_instance()

    ca.num = c_int(num)
    
    ca.data.contents = pointer(data_array)
    ca.rand= None
    
    ca.ndata.contents = c_size_t(ndata)
    ca.nrand.contents = c_double(nrand)

    ca.wdata.contents = c_double(wdata)
    ca.wrand.contents = c_double(wrand)
    
    ca.alpha.contents = c_double(alpha)
    ca.shot.contents = c_double(shot)
    ca.norm.contents = c_double(norm)

    for i in range(ndata):
        da = ca.data[0][i]

        da.w = c_double(1.)
        da.x[0] = c_double(x[i])
        da.x[1] = c_double(y[i])
        da.x[2] = c_double(z[i])
   
   
    return ca

def main():
    lib = cdll.LoadLibrary("/global/homes/a/avariu/phd/chengscodes/powspec_cffi/libpowspec.so")

    config_file = b'/global/homes/a/avariu/phd/chengscodes/powspec_cffi/etc/powspec_HODFIT.conf'
    output_file = b'/global/homes/a/avariu/phd/1.041halo.dat_LOS996.gcat_wrapper_py.pspec'
    input_file = b'/global/homes/a/avariu/desi_avariu/FastPM_SLICS/slics_galaxy_rsd/1.041halo.dat_LOS996.gcat'

    ### Arguments
    size_text = len(input_file)
    arg0  = create_string_buffer(b'test', size_text)
    arg1  = create_string_buffer(b'-c', size_text)
    arg2 = create_string_buffer(config_file)

    arg3  = create_string_buffer(b'-d', size_text)
    arg4  = create_string_buffer(input_file, size_text)

    arg5  = create_string_buffer(b'-a', size_text)
    arg6  = create_string_buffer(output_file, size_text)

    N_args = 7

    args_pointer_instance = POINTER(c_char) * N_args
    args_pointer_arr_to_send = args_pointer_instance(arg0, arg1, arg2, arg3, arg4, arg5, arg6)

    ### Compute catalog
    cata = compute_catalog()

    ### Declaration and initialization of parameters
    nkbin = c_int(0)

    #### Method 1
    nbin = 1000
    pk_instance = c_double * (4 * nbin)
    pk_array = pk_instance()
    t = lib.compute_pk(pointer(cata), pointer(nkbin), pointer(pk_array), c_int(N_args), args_pointer_arr_to_send)
    if t == 0:
        print("ERROR: the pk code crashed")
    #####

    #### Method 2

    # lib.compute_pk.restype = c_void_p
    # address_pk = lib.compute_pk(pointer(cata), pointer(nkbin), c_int(N_args), args_pointer_arr_to_send)
    # nkbin_value = nkbin.value

    # pk_instance = c_double * (4 * nkbin_value)

    # print(address_pk)
    # pk_array = pk_instance.from_address(address_pk)



    # print(pointer(pk_array))
    nkbin_value = nkbin.value

    nbin = nkbin_value
    ### From C to NumPy Array
    k = np.zeros(nbin)
    pk0 = np.zeros(nbin)
    pk2 = np.zeros(nbin)
    pk4 = np.zeros(nbin)
    
    for i in range(nbin):
        k[i] = pk_array[i]
        pk0[i] = pk_array[nbin + i]
        pk2[i] = pk_array[2 * nbin + i]
        pk4[i] = pk_array[3 * nbin + i]

    # lib.free_pk_array(pointer(pk_array))
    # print(pk_array[0: nbin])
    # print(k)
    np.savetxt("/global/homes/a/avariu/phd/1.041halo.dat_LOS996.gcat_wrapper_py.pspec", np.array([k, pk0, pk2, pk4]).T, fmt=("%f %f %f %f"))

def plot():
    import matplotlib.pyplot as pt
    fig, ax = pt.subplots(1, 3)
    k1, pk01, pk21, pk41 = np.loadtxt("/global/homes/a/avariu/phd/1.041halo.dat_LOS996.gcat.pspec", usecols=(0, 5, 6, 7), unpack=True)
    # ax[0].plot(k, k*pk0)
    # ax[1].plot(k, k*pk2)
    # ax[2].plot(k, k*pk4)

    k, pk0, pk2, pk4 = np.loadtxt("/global/homes/a/avariu/phd/1.041halo.dat_LOS996.gcat_wrapper_py.pspec", usecols=(0, 1, 2, 3), unpack=True)
    ax[0].plot(k[1:], (pk0[1:] - pk01[1:]))
    ax[1].plot(k[1:], (pk2[1:] - pk21[1:]))
    ax[2].plot(k[1:], (pk4[1:] - pk41[1:]))

    ax[0].axhline(0, color="grey")
    ax[1].axhline(0, color="grey")
    ax[2].axhline(0, color="grey")
    # ax[0].set_ylim([-0.1, 0.1])
    # ax[1].set_ylim([-0.1, 0.1])
    # ax[2].set_ylim([-0.1, 0.1])
    fig.savefig("test.png")


# def compute_ffi():
#     ffibuilder = FFI()
#     ffibuilder.cdef("double *compute_pk(CATA *cata, int *nkbin, int argc, char *argv[]);")
#     ffibuilder.set_source("_mypowspec_cffi",
#     """
#      #include "powspec.h"   // the C header of the library
#     """, libraries=['libpowspec.so'])   # library name, for the linker

if __name__ == '__main__':
    # compute_ffi()
    main()
    plot()




# pk_array = c_double * (4*nbin)

# lib.compute_pk.restype = c_void_p
# address_pk = lib.compute_pk(pointer(cata), pointer(nkbin), pointer(pk_array), c_int(N), s_to_send)

# print(address_pk)
# pk_instance = pk_array.from_address(address_pk)

