import numpy as np
from pathlib import Path


da = 64
db = 64
dc = 96
dd = 48
de = 64
df = 64
dg = 96
dh = 48






def dump_bacd_cdab():
    bacd = np.random.rand(db, da, dc, dd).astype(np.float32)
    cdab = np.zeros((dc, dd, da, db), dtype=np.float32)

    np.einsum("bacd->cdab", bacd, out=cdab)
    cdab = np.ascontiguousarray(cdab)

    path = "data/bacd_cdab"
    Path(path).mkdir(parents=True, exist_ok=True)
    bacd.reshape(-1).tofile(f"{path}/bacd.bin")
    cdab.reshape(-1).tofile(f"{path}/cdab.bin")



def dump_cdab_cdef_efab():
    cdab = np.random.rand(dc, dd, da, db).astype(np.float32)
    cdef = np.random.rand(dc, dd, de, df).astype(np.float32)
    efab = np.zeros((de, df, da, db), dtype=np.float32)

    np.einsum("cdab,cdef->efab", cdab, cdef, out=efab)
    efab = np.ascontiguousarray(efab)

    path = "data/cdab_cdef_efab"
    Path(path).mkdir(parents=True, exist_ok=True)
    cdab.reshape(-1).tofile(f"{path}/cdab.bin")
    cdef.reshape(-1).tofile(f"{path}/cdef.bin")
    efab.reshape(-1).tofile(f"{path}/efab.bin")


def dump_efgh_efab_gabh():
    efgh = np.random.rand(de, df, dg, dh).astype(np.float32)
    efab = np.random.rand(de, df, da, db).astype(np.float32)
    gabh = np.zeros((dg, da, db, dh), dtype=np.float32)

    np.einsum("efgh,efab->gabh", efgh, efab, out=gabh)
    gabh = np.ascontiguousarray(gabh)

    path = "data/efgh_efab_gabh"
    Path(path).mkdir(parents=True, exist_ok=True)
    efgh.reshape(-1).tofile(f"{path}/efgh.bin")
    efab.reshape(-1).tofile(f"{path}/efab.bin")
    gabh.reshape(-1).tofile(f"{path}/gabh.bin")


def dump_main():
    print("Allocating and initializing memory ...")
    bacd = np.random.rand(db, da, dc, dd).astype(np.float32)
    cdef = np.random.rand(dc, dd, de, df).astype(np.float32)
    efgh = np.random.rand(de, df, dg, dh).astype(np.float32)
    abef = np.zeros((da, db, de, df), dtype=np.float32)
    gabh = np.zeros((dg, da, db, dh), dtype=np.float32)

    print("Computing bacd,cdef->abef ...")
    np.einsum("bacd,cdef->abef", bacd, cdef, out=abef)
    abef = np.ascontiguousarray(abef)
    print("Computing abef,efgh->gabh ...")
    np.einsum("abef,efgh->gabh", abef, efgh, out=gabh)
    gabh = np.ascontiguousarray(gabh)

    print("Writing tensors to data directory ...")
    path = "data/main"
    Path(path).mkdir(parents=True, exist_ok=True)
    bacd.reshape(-1).tofile(f"{path}/bacd.bin")
    cdef.reshape(-1).tofile(f"{path}/cdef.bin")
    efgh.reshape(-1).tofile(f"{path}/efgh.bin")
    abef.reshape(-1).tofile(f"{path}/abef.bin")
    gabh.reshape(-1).tofile(f"{path}/gabh.bin")

    print("Done.")





def inspect(fp):
    tensor = np.fromfile(fp, dtype=np.float32)
    print(tensor.shape[0])
    print(tensor[0])






if __name__ == "__main__":
    dump_bacd_cdab()
    dump_cdab_cdef_efab()
    dump_efgh_efab_gabh()
    dump_main()












