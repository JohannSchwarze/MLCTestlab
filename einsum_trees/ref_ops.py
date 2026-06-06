import numpy as np



da = 64
db = 64
dc = 96
dd = 48
de = 64
df = 64
dg = 96
dh = 48







def main():
    print("Allocating and initializing memory ...")
    bacd = np.random.rand(db, da, dc, dd).astype(np.float32)
    cdef = np.random.rand(dc, dd, de, df).astype(np.float32)
    efgh = np.random.rand(de, df, dg, dh).astype(np.float32)
    abef = np.zeros((da, db, de, df), dtype=np.float32)
    gabh = np.zeros((dg, da, db, dh), dtype=np.float32)

    print("Computing bacd,cdef->abef ...")
    np.einsum("bacd,cdef->abef", bacd, cdef, out=abef)
    print("Computing abef,efgh->gabh ...")
    np.einsum("abef,efgh->gabh", abef, efgh, out=gabh)

    print("Writing tensors to data directory ...")
    bacd.reshape(-1).tofile("data/bacd.bin")
    cdef.reshape(-1).tofile("data/cdef.bin")
    efgh.reshape(-1).tofile("data/efgh.bin")
    abef.reshape(-1).tofile("data/abef.bin")
    gabh.reshape(-1).tofile("data/gabh.bin")

    print("Done.")





def inspect(fp):
    tensor = np.fromfile(fp, dtype=np.float32)
    print(tensor.shape[0])
    print(tensor[0])






if __name__ == "__main__":
    main()












