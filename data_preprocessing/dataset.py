from astropy.io import fits
import numpy as np
from functools import partial
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp

from utils import load_data_dir, load_tensor
import logging

@staticmethod
def fits_to_tensor(fits_file):
    """Open a FITS file and convert it to a Torch tensor."""
    fits_np = fits.getdata(fits_file, memmap=False)
    return torch.from_numpy(fits_np.astype(np.float32))

# All credit for this dataset loading method goes to Aritra Ghosh. For the paper this corresponds to,
# see Ghosh et al. (2022).
class FITSDataset(Dataset):
    def __init__(self,
                 data_dir,
                 label_col="classes",
                 slug=None,  # a slug is a human readable ID
                 split=None,  # splits are defined in make_split.py file.
                 cutout_size=256,
                 normalize=None,
                 transforms=None,  # Supports a list of transforms or a single transform func.
                 channels=1):

        # Initialize all data paths
        self.data_dir = Path(data_dir)
        self.cutouts_path = self.data_dir / "cutouts"
        self.tensors_path = self.data_dir / "tensors"
        self.tensors_path.mkdir(parents=True, exist_ok=True)

        # Initializing cutout shape, assuming the shape is roughly square-like.
        self.cutout_shape = (channels, cutout_size, cutout_size)

        # Initialize image metadata
        self.channels = channels

        # Transforms
        self.normalize = normalize
        self.transforms = transforms  # Compatible with lists of transformations

        # Load in data info
        self.data_info = load_data_dir(data_dir, slug, split)
        self.filenames = np.asarray(self.data_info["file_name"])
        self.labels = np.asarray(self.data_info[label_col])

        # Load in the tensors using our tensor load method.
        num_tensors = len(self.filenames)
        logging.info(f"Generating PyTorch tensors for {num_tensors} objects.")
        for filename in tqdm(self.filenames):
            filepath = self.tensors_path / (filename + ".pt")
            if not filepath.is_file():
                load_path = self.cutouts_path / filename
                t = FITSDataset.fits_to_tensor(load_path)
                torch.save(t, filepath)

        # Preloading the tensors TODO: double check
        logging.info(f"Preloading {num_tensors} tensors")
        load_fn = partial(load_tensor, tensors_path=self.tensors_path)
        with mp.Pool(mp.cpu_count()) as p:  # Multiplexing the tensor loading process
            self.observations = tqdm(p.imap(load_fn, self.tensor_paths), total=num_tensors)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Check if the instance is a slice of a dataset
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(idx, int):
            # If the index is an integer, we proceed as normal and load up our tensor as a data point.
            # We support wrap around functionality
            pt = self.observations[idx % len(self.observations)]

            # Get image label.
            label = torch.tensor(self.labels[idx % len(self.labels)])

            # Transform the tensor if a transformation is specified.
            if self.transform is not None:
                if hasattr(self.transform, "__len__"):  # If inputted in a list of transforms
                    for transform in self.transform(pt):
                        pt = transform(pt)
                else:  # If inputted a single transform.
                    pt = self.transform(pt)

            if self.normalize is not None:
                pt = self.normalize(pt)

            return pt, label

        raise TypeError(f"Invalid argument specified: {type(idx)}")

