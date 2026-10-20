from glob import glob

import h5py
import numpy as np
from nilearn import image
from scipy.spatial import KDTree

KEYS = {"encoding": "scores", "erp": "evoked", "audioxcorr": "corrs"}


def load_results(datatype: str, desc: str = "*", root: str = "../results"):

    result_files = sorted(glob(f"{root}/sub-*/{datatype}/*_desc-{desc}_result.h5"))
    assert len(result_files), f"no {datatype} results for {desc} in {root}"

    values, coords = [], []
    for result_file in result_files:
        with h5py.File(result_file, "r") as f:
            data = f[KEYS[datatype]][...]
            if datatype == "encoding":
                data = data.mean(0)  # average the two folds
            values.append(data)
            coords.append(f["coords"][...])
            lags = f["lags"][...]

    return np.vstack(values), np.vstack(coords), lags


def get_coord_atlas_labels(
    coords: np.ndarray, atlas_map: str, atlas_labels: list[str]
) -> list[str]:
    atlas_map = image.load_img(atlas_map)
    atlas_image = atlas_map.get_fdata().astype(int)

    # find non-zero labels
    image_label_coords = np.nonzero(atlas_image)

    # transform label indices to MNI space
    atlas_coords = np.vstack(
        image.coord_transform(*image_label_coords, atlas_map.affine)
    ).T

    tree = KDTree(atlas_coords)
    _, nearest_neighbor = tree.query(coords, k=1)

    # look up neighbor index in map
    x = image_label_coords[0][nearest_neighbor]
    y = image_label_coords[1][nearest_neighbor]
    z = image_label_coords[2][nearest_neighbor]

    # convert map index to label, dropping the hemisphere tag
    elec_label_ids = atlas_image[x, y, z]
    elec_labels = [atlas_labels[i].split()[-1] for i in elec_label_ids]

    return elec_labels


def generate_null_dist(n_perms: int = 10000, dim_size: int = 2384, seed: int = 0):
    rng = np.random.default_rng(seed=seed)
    null_dist = np.zeros(n_perms)
    for i in range(n_perms):
        a = rng.normal(size=dim_size)
        b = rng.normal(size=dim_size)
        null_dist[i] = np.corrcoef(a, b)[0, 1]
    return null_dist


def calculate_pvalues(observed: np.ndarray, null_dist: np.ndarray) -> np.ndarray:
    # one sided, with the +1 correction scipy uses for permutation tests
    gamma = np.maximum(1e-14, np.abs(1e-14 * observed))
    n_greater = (null_dist[None, :] >= (observed - gamma)[:, None]).sum(1)
    return (n_greater + 1) / (len(null_dist) + 1)
