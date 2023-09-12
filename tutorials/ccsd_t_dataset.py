import os
import urllib.request
import zipfile
import shutil

import ase.io
import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator as sp
from aml.data.dataset import InMemoryDataset
from aml.data.datapipes import ASEFileReader, AtomsGraphParser, NeighborListBuilder
from torchdata.datapipes.iter import IterableWrapper
from tqdm import tqdm

# Import InMemoryDataset
from aml.data.dataset import InMemoryDataset

def make_dp(src, neighbor_cutoff):
    dp = IterableWrapper(src)
    dp = AtomsGraphParser(dp)
    dp = NeighborListBuilder(dp, neighbor_cutoff, backend="matscipy")
    return dp


def npz_to_ase(npz):
    images = []
    Es = npz["E"] * units.kcal / units.mol
    Fs = npz["F"] * units.kcal / units.mol
    Z = npz["z"]
    Rs = npz["R"]

    for R, E, F in zip(Rs, Es, Fs, strict=True):
        atoms = Atoms(numbers=Z, positions=R)
        atoms.set_calculator(sp(atoms, energy=E, forces=F))
        images.append(atoms)
    return images

def create_ccsd_t_dataset():
    url = "http://www.quantum-machine.org/gdml/data/npz/ethanol_ccsd_t.zip"
    urllib.request.urlretrieve(url, "ethanol_ccsd_t.zip")
    # Unzip the file
    with zipfile.ZipFile("ethanol_ccsd_t.zip", "r") as zip_ref:
        zip_ref.extractall("ethanol_ccsd_t")
    # Remove the zip file
    os.remove("ethanol_ccsd_t.zip")
    
    # Load the data
    train_images = npz_to_ase(np.load("ethanol_ccsd_t/ethanol_ccsd_t-train.npz"))
    test_images = npz_to_ase(np.load("ethanol_ccsd_t/ethanol_ccsd_t-test.npz"))

    shutil.rmtree("ethanol_ccsd_t")
    
    train_dp = tqdm(make_dp(train_images, 5.0), desc="Processing training dataset")
    test_dp = tqdm(make_dp(test_images, 5.0), desc="Processing test dataset")
    train_dataset = InMemoryDataset(train_dp)
    test_dataset = InMemoryDataset(test_dp)
    
    return train_dataset, test_dataset