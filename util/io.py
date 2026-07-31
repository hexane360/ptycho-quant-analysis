from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray
import h5py
from scipy.io import loadmat


def load_foldslice(path: t.Union[str, Path], crop: bool = True) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
    """
    Load a fold_slice reconstruction.

    Returns: A tuple `(obj, object_sampling)`.

    Arguments:
      crop: Whether to crop to the reconstruction region-of-interest. Defaults to True.
    """
    f = loadmat(path, variable_names=['object', 'p'])
    obj = f['object'].T.astype(numpy.complex64)
    obj_sampling = f['p']['dx_spec'][0, 0].ravel()[::-1]

    if crop:
        roi_x, roi_y = f['p']['object_ROI'][0, 0].ravel()
        yy, xx = numpy.meshgrid(roi_y.ravel(), roi_x.ravel(), indexing='ij')
        obj = obj[:, yy, xx]

    return obj, obj_sampling


def load_phaser(path: t.Union[str, Path], crop: bool = True) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
    """
    Load a phaser reconstruction.

    Returns: A tuple `(obj, object_sampling)`.

    Arguments:
      crop: Whether to crop to the reconstruction region-of-interest. Defaults to True.
    """
    f = h5py.File(path)
    obj = t.cast(NDArray[numpy.floating], f['object/data'])[()]
    obj_sampling = t.cast(NDArray[numpy.floating], f['object/sampling'])[()]

    if crop:
        obj_corner = t.cast(NDArray[numpy.floating], f['object/corner'])[()]
        roi_min, roi_max = (t.cast(NDArray[numpy.floating], f[f'object/region_{k}'])[()] for k in ('min', 'max'))

        min_i, min_j = numpy.ceil((roi_min - obj_corner) / obj_sampling).astype(numpy.int_)
        max_i, max_j = numpy.floor((roi_max - obj_corner) / obj_sampling).astype(numpy.int_)

        obj = obj[..., min_i:max_i, min_j:max_j]

    return obj, obj_sampling