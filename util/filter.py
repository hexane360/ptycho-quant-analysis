import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray


def remove_linear_ramp(
    data: ArrayLike, mask: t.Optional[NDArray[numpy.bool_]] = None
) -> NDArray[numpy.floating]:
    """
    Removes a linear 'ramp' from an image or stack of images.
    """

    data = numpy.array(data)
    output = numpy.empty_like(data)

    (yy, xx) = (arr.flatten() for arr in numpy.indices(data.shape[-2:], dtype=float))
    pts = numpy.stack((numpy.ones_like(xx), xx, yy), axis=-1)

    if mask is None:
        mask = numpy.ones(len(yy), dtype=numpy.bool_)
    else:
        mask = mask.flatten()

    for idx in numpy.ndindex(data.shape[:-2]):
        layer = data[tuple(idx)].astype(numpy.float64)
        p, residues, rank, singular = numpy.linalg.lstsq(pts[mask], layer.flatten()[mask], rcond=None)
        output[idx] = (layer - (p @ pts.T).reshape(layer.shape)).astype(output.dtype)

    return output