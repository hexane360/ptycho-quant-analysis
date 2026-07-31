from dataclasses import dataclass, field
import typing as t

import networkx
import numpy
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree

GraphKey: t.TypeAlias = t.Union[int, t.Literal['start', 'end']]


def project(image: numpy.ndarray,
            theta: t.Union[float, ArrayLike]) -> numpy.ndarray:
    """
    Project the image to a line along the `theta` direction(s), and return a 1d slice.

    For instance, when theta = 0, this projects the image onto the x-axis.

    Returns a ndarray of shape `(len(theta), image.shape[0])`
    """
    from skimage.transform import warp

    if isinstance(theta, float):
        return project(image, [theta])[0]
    else:
        theta = numpy.asanyarray(theta)

    center = image.shape[0] // 2
    projected = numpy.zeros((len(theta), image.shape[0]), dtype=image.dtype)
    for i, angle in enumerate(theta):
        cos_a, sin_a = numpy.cos(angle), numpy.sin(angle)
        R = numpy.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                         [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                         [0, 0, 1]])
        rotated = warp(image, R, clip=False)
        projected[i] = rotated.sum(axis=1)
    return projected


@dataclass
class FoundLatticeDirections:
    img: numpy.ndarray = field()
    """Input image."""
    theta: numpy.ndarray = field()
    """Theta values tested. Shape of `(n_theta,)`."""
    sino: numpy.ndarray = field()
    """Sinogram of the image. Shape of `(n_theta, min(img.shape))`."""
    fft: numpy.ndarray = field()
    """FFT of sinogram. Shape of `(n_theta, min(img.shape)//2)`."""
    contrast: numpy.ndarray = field()
    """Measured contrast. Shape of `(n_theta,)`."""
    peaks: numpy.ndarray = field()
    """
    Found `(theta, k)` peak indices. Shape of `(n_peaks, 2)`.
    Sorted by decreasing cost.
    """
    k_wt: float = field()
    """
    `k_wt` used in cost function.
    """
    dirs: numpy.ndarray = field()
    """
    Found lattice directions. Shape of `(n_peaks,)`.
    Sorted by decreasing cost.
    """
    ks: numpy.ndarray = field()
    """
    Peak wavevectors.
    """
    costs: numpy.ndarray = field()
    """
    Peak costs. Calculated as `k_wt * k / k_max - int / int_max`.
    """


def find_lattice_directions(image: numpy.ndarray, n_theta: t.Optional[int] = None,
                            n_peaks: int = 8,
                            k_wt: float = 10) -> FoundLatticeDirections:
    """
    Find the primary lattice directions in the image.

    #Parameters:

    - `image` is the image to process.
    - `n_theta` is the number of directions to consider.
    - `n_peaks` is the number of peaks to find in the sinogram FFT
    - `k_wt` is the weight to apply to finding a small k-vector, sacrificing FFT intensity.

    A higher value prioritizes highly ordered directions, while a lower value prioritizes
    high contrast directions.

    Return a `FoundLatticeDirections` object containing the results.
    """
    from skimage.feature import peak_local_max

    cutout = circular_cutout(image)
    if n_theta is None:
        n_theta = cutout.shape[0]

    theta = numpy.linspace(-numpy.pi/2., numpy.pi/2., n_theta, endpoint=False)

    sino = project(cutout, -theta)
    contrast = numpy.std(sino, axis=-1)
    fft = numpy.abs(numpy.fft.rfft(sino, axis=-1))

    peaks = peak_local_max(fft, min_distance=5, num_peaks=n_peaks * 3)
    # hack to de-duplicate theta values
    peaks = peaks[numpy.argsort(peaks[:, 1])]  # sort by increasing k
    _, indices = numpy.unique(peaks[:, 0], return_index=True)  # de-duplicate thetas
    peaks = peaks[indices]

    ints = fft[peaks[:, 0], peaks[:, 1]]
    int_max = numpy.max(ints)
    k_max = fft.shape[1]
    costs = k_wt * peaks[:, 1] / k_max - ints / int_max

    #peak_dist = peak_width * n_theta // 180  # peak width in samples
    #peaks, props = scipy.signal.find_peaks(contrast, distance=peak_dist, wlen=peak_dist,
    #                                       prominence=prominence)

    idxs = numpy.argsort(costs)[:n_peaks]  # peaks sorted by cost function
    costs = costs[idxs]

    peaks = peaks[idxs]
    dirs = theta[peaks[:, 0]]
    ks = peaks[:, 1] / k_max / 2.

    return FoundLatticeDirections(image, theta, sino, fft, contrast, peaks, k_wt, dirs, ks, costs)


def calc_lattice_transform(dir1: float, dir2: float) -> t.Tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """
    Calculate the transformation which orthogonalizes `dir1` and `dir2`.

    Return the forward and reverse transformation in a tuple.
    The forward transformation takes lattice coords to image coords, while
    the reverse transformation takes image coords and converts to lattice coords.
    """

    # transforms lattice coords to image coords
    T = numpy.array([[numpy.sin(dir1), numpy.sin(dir2)],
                     [numpy.cos(dir1), numpy.cos(dir2)]])
    # transforms image coords to lattice coords
    U = numpy.linalg.inv(T)

    return (T, U)


def construct_grid_points(grid_peaks: NDArray[numpy.floating], fracs: ArrayLike,
                          mod_i: int = 1, mod_j: int = 1,
                          start_i: int = 0, start_j: int = 0, flatten: bool = True):
    """
    Construct points in the grid given by `xs` and `ys`,
    offset by the displacements `fracs` inside each cell.

    If `mod_i` or `mod_j` is specified, points are
    returned from every `ixj`th cell.
    """

    if isinstance(grid_peaks, numpy.ma.masked_array):
        grid_peaks = grid_peaks.filled(numpy.nan)

    grid_peaks = grid_peaks[start_i::mod_i, start_j::mod_j]

    fracs = numpy.atleast_1d(fracs)

    # final broadcast shape: *fracs.shape[:-1], *xx.shape, 2
    # shape: (*fracs.shape[:-1], 1, 1, 1)
    i_fracs = fracs[..., 0][..., None, None, None]
    j_fracs = fracs[..., 1][..., None, None, None]

    # calculate unit cell vectors in i and j
    # extrapolate last value
    i_vecs = numpy.diff(grid_peaks, axis=0)
    i_vecs = numpy.concatenate([i_vecs, i_vecs[[-1]]], axis=0)
    j_vecs = numpy.diff(grid_peaks, axis=1)
    j_vecs = numpy.concatenate([j_vecs, j_vecs[:, [-1]]], axis=1)

    vals = grid_peaks + (i_fracs * i_vecs + j_fracs * j_vecs)

    mask = numpy.all(numpy.isfinite(vals), axis=-1)
    if flatten or vals.ndim < 4:
        return vals[mask, :]

    mask = numpy.all(mask, axis=tuple(range(vals.ndim - 3)))
    return vals[..., mask, :]


def construct_grid_points_masked(grid_peaks: NDArray[numpy.floating], fracs: ArrayLike,
                                 mod_i: int = 1, mod_j: int = 1,
                                 start_i: int = 0, start_j: int = 0) -> numpy.ma.masked_array:
    """
    Construct points in the grid given by `xs` and `ys`,
    offset by the displacements `fracs` inside each cell.

    If `mod_i` or `mod_j` is specified, points are
    returned from every `ixj`th cell.
    """

    if isinstance(grid_peaks, numpy.ma.masked_array):
        grid_peaks = grid_peaks.filled(numpy.nan)

    grid_peaks = grid_peaks[start_i::mod_i, start_j::mod_j]

    fracs = numpy.atleast_1d(fracs)

    # final broadcast shape: *fracs.shape[:-1], *xx.shape, 2
    # shape: (*fracs.shape[:-1], 1, 1, 1)
    i_fracs = fracs[..., 0][..., None, None, None]
    j_fracs = fracs[..., 1][..., None, None, None]

    # calculate unit cell vectors in i and j
    i_vecs = numpy.diff(grid_peaks, axis=0)[:, :-1]
    #i_vecs = numpy.concatenate([i_vecs, i_vecs[[-1]]], axis=0)
    j_vecs = numpy.diff(grid_peaks, axis=1)[:-1, :]
    #j_vecs = numpy.concatenate([j_vecs, j_vecs[:, [-1]]], axis=1)

    vals = grid_peaks[..., :-1, :-1, :] + (i_fracs * i_vecs + j_fracs * j_vecs)

    mask = numpy.any(~numpy.isfinite(vals), axis=-1, keepdims=True)
    mask = numpy.any(mask, axis=tuple(range(vals.ndim - 3)), keepdims=True)

    return numpy.ma.masked_array(vals, numpy.broadcast_to(mask, vals.shape))


class GraphIndexer:
    def __init__(self, peaks: ArrayLike, dist: float, threshold: float = 1e8,
                 stretch: float = 4.0):
        """
        Create a GraphIndexer, using the given parameters.

        Args:
          peaks: Peaks to index into rows and columns
          dist: Average nearest neighbor distance between peaks
          threshold: Distance above which adding new nodes is penalized.
                     Usually disabled by setting to a large value.

        Returns:
          A `GraphIndexer`
        """
        self.threshold: float = threshold
        """Threshold distance to penalize node inclusion"""
        self.peaks: NDArray[numpy.float64] = numpy.array(peaks, dtype=numpy.float64)
        """Input peak positions to index"""
        self.dist: float = dist
        """Approximate nearest-neighbor distance between peaks"""
        self.stretch: float = stretch

    def weight_of_edge(self, a: GraphKey, b: GraphKey, axis: int) -> float:
        """
        Calculate the weight of the edge `(a, b)`, along axis `axis`.

        Args:
            a: First vertex
            b: Second vertex

        Returns:
          Weight between the two vertices
        """
        if isinstance(a, int) and isinstance(b, int):
            return abs(self.peaks[a][axis] - self.peaks[b][axis]) - self.threshold
        else:
            return 0.

    def remove_repeated(self, edges: t.Iterable[t.Tuple[int, int]], axis: int) -> t.List[t.Tuple[int, int]]:
        """
            given the list of the edges, leaves only edges that goes "down" or "right" based on flag

            edges - list of edges
            result - list of resulting edges
        """
        out_edges = []

        for edge in edges:
            if (self.peaks[edge[0]][1-axis] > self.peaks[edge[1]][1-axis]):
                out_edges.append((edge[1], edge[0]))
            else:
                out_edges.append((edge[0], edge[1]))

        return out_edges

    def add_start_and_end_nodes(self, g: networkx.DiGraph):
        """
        connects the vertexes that does not have outcoming edges from the vertexes on the distance less than trashhold (vertical distance for rows and horisontal for colums) to the "start" node and vertex that does not have incoming edges to the "end" node
        input: 
            graph g
        """
        start_nodes = []
        for vertex in g.nodes():
            if not len(g.in_edges(vertex)):
                start_nodes.append(vertex)

        for vertex in start_nodes:
            g.add_edge("start", vertex)

        end_nodes = []
        for vertex in g.nodes():
            if vertex != "start" and not len(g.out_edges(vertex)):
                end_nodes.append(vertex)

        for vertex in end_nodes:
            g.add_edge(vertex, "end")

    def build_graph(self, axis: int = 0) -> networkx.DiGraph:
        """
        builds the graph, where each vertex u and v are connected if (x_u - x_v)^2/c1 + (y_u - y_v)^2/c2 <= dist^2, where c1, c2 are (4, 1/4) for colums and (1/4, 4) for rows
        return: graph
        """
        g = networkx.DiGraph()
        g.add_nodes_from(range(len(self.peaks)))

        stretch = numpy.array([self.stretch, 1/self.stretch])

        peaks_compressed = self.peaks * (stretch if axis == 0 else stretch[::-1])

        tree = KDTree(peaks_compressed)
        edges = self.remove_repeated(tree.query_pairs(self.dist), axis)
        g.add_edges_from(edges)

        self.create_weights(g, axis)

        return g

    def create_weights(self, g: networkx.DiGraph, axis: int):
        """
        Add weights to the graph `g`, for distances along axis `axis`.
        Modifies `g`.
        """
        weights = {}

        for i in g.edges:
            weights[i] = self.weight_of_edge(i[0], i[1], axis)

        networkx.set_edge_attributes(g, values = weights, name = 'weight')

    def get_path(self, pred: t.Dict[GraphKey, t.Any]) -> t.List[int]:
        """
        Given the list of previous vertexes in the shortest path from start node,
        returns the path from start to end in reverse order

        Args:
            Predecessor nodes in shortest path

        Returns:
            New shortest path from `'start'` to `'end'`
        """
        path = []
        vertex = 'end'

        while True:
            if vertex not in ('start', 'end'):
                path.append(vertex)
            try:
                vertex = pred[vertex][0]
            except (KeyError, IndexError):
                return path

    def create_columns_or_rows(self, axis: int) -> t.Tuple[t.Dict[int, int], int]:
        """
        Create columns or rows (controlled by `axis`).

        Args:
          axis: Axis to cluster along (0 for rows, 1 for columns)

        Returns:
            A tuple containing a dictionary and the number of columns/rows. The
            dictionary contains a mapping from peak index to the column/row it belongs to.
        """

        g = self.build_graph(axis)

        columns = {}
        counter = 0
        while True:
            self.add_start_and_end_nodes(g)
            path = []
            pred, _dist = networkx.bellman_ford_predecessor_and_distance(g, 'start', target = 'end', weight='weight', heuristic=False)
            path = self.get_path(pred)
            if not len(path):
                break

            for vertex in path:
                columns[vertex] = counter
                g.remove_node(vertex)

            counter += 1

        #put in array
        coordinate_of_column = numpy.full(counter, numpy.nan)
        for vertex in columns.keys():
            if (numpy.isnan(coordinate_of_column[columns[vertex]])):
                coordinate_of_column[columns[vertex]] = self.peaks[vertex][axis]

        sorted_columns = numpy.argsort(coordinate_of_column)
        inversed_sorted_columns = numpy.full(counter, numpy.nan)
        for i in range(len(sorted_columns)):
            inversed_sorted_columns[sorted_columns[i]] = i
        for vertex in columns.keys():
            columns[vertex] = int(inversed_sorted_columns[columns[vertex]])

        return (columns, counter)

    def cluster(self, peaks: t.Optional[numpy.ndarray] = None) -> t.Tuple[numpy.ma.masked_array, numpy.ma.masked_array]:
        """
        Perform clustering.

        Returns:
          Tuple of two masked arrays `(indices, positions)`, each with the shape
          of the indexed grid. Values in `indices` correspond to the peak
          indices passed in, and `positions` correspond to the peak positions
          passed in.
        """
        cols, _n_cols = self.create_columns_or_rows(axis=1)
        rows, _n_rows = self.create_columns_or_rows(axis=0)

        return _construct_index_grid(
            self.peaks if peaks is None else peaks,
            [rows[i] for i in range(len(self.peaks))],
            [cols[i] for i in range(len(self.peaks))],
        )


def _construct_index_grid(
        peaks: t.Union[NDArray[numpy.floating], NDArray[numpy.integer]],
        row_indices: t.Union[NDArray[numpy.integer], t.Sequence[int]],
        col_indices: t.Union[NDArray[numpy.integer], t.Sequence[int]],
    ) -> t.Tuple[numpy.ma.masked_array, numpy.ma.masked_array]:

    n_rows = int(numpy.max(row_indices)) + 1
    n_cols = int(numpy.max(col_indices)) + 1

    grid = numpy.full((n_rows, n_cols), -1)

    indices = numpy.arange(len(row_indices))

    grid[row_indices, col_indices] = indices

    mask = grid == -1
    return (
        numpy.ma.array(grid, mask=mask, dtype=numpy.int64),
        numpy.ma.array(peaks[grid], mask=numpy.stack((mask,) * 2, axis=-1),
                        dtype=numpy.float64, fill_value=numpy.nan) 
    )


def graph_index(
        peaks: numpy.ndarray, dirs: FoundLatticeDirections,
        x_dir: int = 0, y_dir: int = 1, *,
        flip_x: bool = False, flip_y: bool = False,
        stretch: float = 4.0,
    ) -> t.Tuple[numpy.ma.masked_array, numpy.ma.masked_array]:

    x_angle, y_angle = dirs.dirs[x_dir], dirs.dirs[y_dir]
    if flip_x:
        x_angle += numpy.pi
    if flip_y:
        y_angle += numpy.pi

    _T, U = calc_lattice_transform(y_angle, x_angle)
    # scale to unit grid
    U = numpy.diag([dirs.ks[y_dir], dirs.ks[x_dir]]) @ U

    transformed_peaks = peaks.astype('float') @ U.T

    indexer = GraphIndexer(transformed_peaks, 1.0, stretch=stretch)

    # return un-transformed peaks
    return indexer.cluster(peaks)


def circular_cutout(image: numpy.ndarray, r=None) -> numpy.ndarray:
    """
    Return a masked circular slice from `image`, with radius `r`.

    Returns a square ndarray zeroed outside the disk.
    """
    from skimage.draw import disk

    max_r = (min(image.shape) - 1) // 2
    if r is None:
        r = max_r
    elif r > max_r:
        pass

    d = 2*max_r + 1
    indices = tuple(slice((s-d)//2, (s-d)//2 + d) for s in image.shape)
    cutout = image[indices]
    assert cutout.shape == (d, d)

    center = ((cutout.shape[0] - 1) / 2., (cutout.shape[1] - 1) / 2.)
    rr, cc = disk(center, r, shape=cutout.shape)
    mask = numpy.zeros(cutout.shape, dtype=int)
    mask[rr, cc] = 1

    return cutout * mask