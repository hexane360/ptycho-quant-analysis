from itertools import cycle, islice
import typing as t

import numpy
from numpy.typing import NDArray
from scipy.spatial import KDTree
from matplotlib import pyplot
from matplotlib.colors import Normalize

if t.TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.collections import QuadMesh
    from index import FoundLatticeDirections


def plot_found_lattice_directions(result: 'FoundLatticeDirections', *,
                                  fig: t.Optional['Figure'] = None, img_cmap=None) -> 'Figure':
    """
    Plot the [`FoundLatticeDirections`][aci.index.FoundLatticeDirections] returned by
    [`find_lattice_directions`][aci.index.find_lattice_directions]. Useful for debugging.

    # Parameters:

    - `result`: [`FoundLatticeDirections`][aci.index.FoundLatticeDirections] to plot
    - `fig`: [`Figure`][matplotlib.figure.Figure] to plot on. If not specified, a new
      figure is created.
    - `img_cmap`: Colormap to use for image and sinogram.

    Returns the [`Figure`][matplotlib.figure.Figure] used.
    """

    if fig is None:
        fig = pyplot.figure()

    (ax1, ax2) = fig.subplots(ncols=2, gridspec_kw={'width_ratios': (1, 2)})  # type: ignore

    colors = list(islice(cycle(('red', 'yellow', 'green')), len(result.peaks)))

    # plot sinogram fft
    ks = numpy.linspace(0., 0.5, result.fft.shape[1])
    ax1.pcolormesh(ks, result.theta, numpy.log(result.fft), cmap=img_cmap, shading='nearest')
    #ax1.scatter(result.peaks[:, 1], result.peaks[:, 0], c=colors)
    ax1.scatter(result.ks, result.dirs, c=colors)
    #ax1.scatter(result.contrast[result.peaks], result.theta[result.peaks], c=colors)
    ax1.set_xlabel("wavevector")
    ax1.set_ylabel("Angle")
    ax1.invert_yaxis()

    norm = Normalize()
    norm(result.costs)

    # and plot image
    ax2.imshow(result.img, cmap=img_cmap)
    # and found directions
    center = numpy.array([(result.img.shape[0]-1)/2, (result.img.shape[1]-1)/2])
    for (i, (d, cost, color)) in enumerate(zip(result.dirs, result.costs, colors)):
        #d = numpy.where(d > numpy.pi/2., d - numpy.pi, d)
        r = numpy.array([0, 0.2 + 0.8 * (1-norm(cost))]) * numpy.min(center)
        v = numpy.array([numpy.sin(d), numpy.cos(d)])
        pts = (numpy.outer(r, v) + center).T
        ax2.plot(*pts[::-1], '-', color=color,
                 linewidth=2)
        # plot text
        ax2.text(*(40 * v + center)[::-1], f"#{i}: {cost:.2e}", fontsize='medium', color='white',
                 ha='left', va='bottom', rotation=-d * 180/numpy.pi, rotation_mode='anchor')

    return fig


def plot_voronoi(ax: 'Axes', xs, ys, vs, img_shape: t.Tuple[int, ...],
                 max_r: t.Optional[float] = None, norm: t.Optional[Normalize] = None,
                 vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
                 **pcolormesh_kwargs) -> 'QuadMesh':
    """
    Plot a voronoi diagram on `ax` given the points `(xs, ys, vs)`.

    Plots to fit over an image with shape `img_shape`.

    Additional kwargs are passed to [`pcolormesh`][matplotlib.axes.Axes.pcolormesh].
    """
    vs = numpy.concatenate([numpy.asanyarray(vs), [numpy.nan]])

    pts = numpy.moveaxis(numpy.indices(img_shape), 0, -1)
    tree = KDTree(numpy.stack((ys, xs), axis=-1))
    dists, idxs = tree.query(pts, distance_upper_bound=max_r or numpy.inf)

    if norm is None:
        vmin = float(numpy.nanquantile(vs[idxs], 0.01)) if vmin is None else vmin
        vmax = float(numpy.nanquantile(vs[idxs], 0.99)) if vmax is None else vmax
        norm = Normalize(vmin, vmax)
    elif vmin is not None or vmax is not None:
        raise ValueError("Cannot specify both 'norm' and 'vmin'/'vmax'")

    ax.set_aspect('equal')
    ax.yaxis.set_inverted(True)
    return ax.pcolormesh(
        pts[..., 1], pts[..., 0], vs[idxs],
        shading='nearest', rasterized=True, norm=norm, **pcolormesh_kwargs
    )


def distribute(n: int, fig_aspect: t.Optional[float] = None, ax_aspect: t.Optional[float] = None) -> t.Tuple[int, int]:
    """
    Return the optimal number of rows and columns to display `n`
    plots in, which minimizes perimeter.

    #Parameters:

    - `fig_aspect`: The aspect ratio of the overall figure (width/height)
    - `ax_aspect`: The aspect ratio of each plot (width/height)
    """
    if fig_aspect is None:
        fig_aspect = 4.
    if ax_aspect is None:
        ax_aspect = 1.

    if n == 0:
        return (0, 0)
    if n == 1:
        return (1, 1)

    aspect = fig_aspect / ax_aspect

    ncols = numpy.arange(n, 0, -1)
    nrows = numpy.ceil(n / ncols).astype('int')

    # minimize perimeter (after scaling by aspect)
    perim = nrows * aspect + ncols
    i = numpy.argmin(perim)

    return (nrows[i], ncols[i])


def plot_stack(stack: NDArray[numpy.floating], norm: t.Optional[Normalize] = None,
               vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
               colorbar: bool = False, **imshow_kwargs) -> t.Tuple['Figure', NDArray[numpy.object_]]:

    n = len(stack)
    n_rows, n_cols = distribute(n)

    if norm is None:
        vmin = float(numpy.nanmin(stack)) if vmin is None else vmin
        vmax = float(numpy.nanmax(stack)) if vmax is None else vmax
        norm = Normalize(vmin, vmax)
    elif vmin is not None or vmax is not None:
        raise ValueError("Cannot specify both 'norm' and 'vmin'/'vmax'")

    fig, axs = pyplot.subplots(n_rows, n_cols, sharex=True, sharey=True, constrained_layout=True)
    ax: Axes

    for ax in axs.flat:
        ax.set_axis_off()

    for (i, (ax, img)) in enumerate(zip(axs.flat, stack)):
        ax.set_title(f"i={i}")
        sm = ax.imshow(img, norm=norm, **imshow_kwargs)

    if colorbar and len(stack):
        fig.colorbar(sm, ax=axs)

    return fig, axs