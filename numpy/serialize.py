import numpy as np

def try_tobytes():
    a = np.random.randn(20, 5).astype(np.float32)
    data = a.tobytes()
    # b = np.frombytes(data) - no frombytes
    b = np.frombuffer(data)
    # Wont work due to loss of info like shape, dtype
    print("np tobytes/frombuffer works?", np.array_equal(a, b))

from io import BytesIO

def np_dumps(arr):
    bio = BytesIO()
    np.save(bio, arr)
    return bio.getvalue()

def np_loads(data):
    bio = BytesIO(data)
    return np.load(bio)

def try_np_save_load():
    a = np.random.randn(20, 5).astype(np.float32)
    data = np_dumps(a)
    b = np_loads(data)
    print("np save/load works?", np.array_equal(a, b))

import pickle

def try_pickle():
    a = np.random.randn(20, 5).astype(np.float32)
    data = pickle.dumps(a)
    b = pickle.loads(data)
    print("pickle works?", np.array_equal(a, b))

import h5py

def hdf5_dumps(arr):
    bio = BytesIO()
    with h5py.File(bio, "w") as fp:
        fp.create_dataset("array", data=arr)
    return bio.getvalue()

def hdf5_loads(data):
    bio = BytesIO(data)
    with h5py.File(bio, "r") as fp:
        dataset = fp["array"]
        arr = dataset[:]
    return arr

def try_hdf5():
    a = np.random.randn(20, 5).astype(np.float32)
    data = hdf5_dumps(a)
    b = hdf5_loads(data)
    print("hdf5 works?", np.array_equal(a, b))

import json
import base64

def np_json_dumps(arr):
    data_b64 = base64.b64encode(arr.tobytes()).decode()
    # Byte-order will be C
    arr_info = {
        "shape": arr.shape,
        "dtype": arr.dtype.name,
        "data_b64": data_b64,
    }
    packed = json.dumps(arr_info).encode()
    return packed

def np_json_loads(data):
    arr_info = json.loads(data.decode())
    data_b64 = arr_info["data_b64"]
    data_bytes = base64.b64decode(data_b64.encode())
    arr = np.frombuffer(data_bytes, dtype=arr_info["dtype"])
    arr = arr.reshape(arr_info["shape"])
    return arr

def try_json():
    # Very limited!!
    a = np.random.randn(20, 5).astype(np.float32)
    data = np_json_dumps(a)
    b = np_json_loads(data)
    print("custom json works?", np.array_equal(a, b))

def main():
    np.random.seed(0)
    try_tobytes()
    try_np_save_load()
    try_pickle()
    try_hdf5()
    try_json()

if __name__ == "__main__":
    main()
