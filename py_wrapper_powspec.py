from ctypes import *
import numpy as np


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
    x, y, z = np.loadtxt("/global/homes/a/avariu/phd/chengscodes/test.test", usecols=(0,1,2), unpack=True)

    num = 1
    ndata = len(x)
    nrand = 0
    wdata = ndata
    wrand = 0.
    alpha = 0.
    shot = 0.
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
    lib = cdll.LoadLibrary("/global/homes/a/avariu/phd/chengscodes/powspec_lib/libpowspec.so")

    config_file = b'/global/homes/a/avariu/phd/chengscodes/powspec_lib/etc/powspec_HODFIT.conf'
    output_file = b'/global/homes/a/avariu/phd/test.pspec_py'
    input_file = b'/global/homes/a/avariu/phd/chengscodes/test.test'

    ### Arguments
    size_text = len(config_file)
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
    # pk_instance = c_double * (4 * nbin)
    # pk_array = pk_instance()

    # if (lib.compute_pk(pointer(cata), pointer(nkbin), pointer(pk_array), c_int(N_args), args_pointer_arr_to_send)):
    #     print("ERROR: the pk code crashed")
    #####

    #### Method 2

    lib.compute_pk.restype = c_void_p
    address_pk = lib.compute_pk(pointer(cata), pointer(nkbin), c_int(N_args), args_pointer_arr_to_send)
    nkbin_value = nkbin.value

    pk_instance = c_double * (4 * nkbin_value)

    print(address_pk)
    pk_array = pk_instance.from_address(address_pk)



    print(pointer(pk_array))
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

    lib.free_pk_array(pointer(pk_array))
    print(pk_array[0: nbin])
    print(k)
    np.savetxt("test_test_2.dat", np.array([k, pk0, pk2, pk4]).T, fmt=("%f %f %f %f"))

def plot():
    import matplotlib.pyplot as pt
    fig, ax = pt.subplots(1, 3)
    k1, pk01, pk21, pk41 = np.loadtxt("test.pspec_CHENG", usecols=(0, 5, 6, 7), unpack=True)
    # ax[0].plot(k, k*pk0)
    # ax[1].plot(k, k*pk2)
    # ax[2].plot(k, k*pk4)

    k, pk0, pk2, pk4 = np.loadtxt("test_test_2.dat", usecols=(0, 1, 2, 3), unpack=True)
    ax[0].plot(k, (pk0 -pk01)/pk01)
    ax[1].plot(k, (pk2 - pk21)/ pk21)
    ax[2].plot(k, (pk4 - pk41)/pk41)
    
    k2, pk02, pk22, pk42 = np.loadtxt("test.pspec_py", usecols=(0, 5, 6, 7), unpack=True)
    ax[0].plot(k, (pk02 -pk01)/pk01)
    ax[1].plot(k, (pk22 - pk21)/ pk21)
    ax[2].plot(k, (pk42 - pk41)/pk41)
    
    fig.savefig("test.png")

if __name__ == '__main__':
    main()
    plot()




# pk_array = c_double * (4*nbin)

# lib.compute_pk.restype = c_void_p
# address_pk = lib.compute_pk(pointer(cata), pointer(nkbin), pointer(pk_array), c_int(N), s_to_send)

# print(address_pk)
# pk_instance = pk_array.from_address(address_pk)

