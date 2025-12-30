import h5py
with h5py.File("../data/processed/landmarks/0-AsociacionCivil/0-5.h5", "r") as f:
    print(f.attrs)