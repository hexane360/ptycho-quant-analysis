from abc import ABC, abstractmethod
from functools import update_wrapper
from itertools import chain
import math
import logging
import inspect
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray
from numpy.lib.recfunctions import append_fields, drop_fields
from lmfit import Parameters, Parameter
from lmfit.model import Model, ModelResult
from scipy.spatial import KDTree


T = t.TypeVar('T')


def yield_if(cond: bool, *values: T) -> t.Tuple[T, ...]:
    """Yields a tuple of `values` if `cond`"""
    return values if cond else ()


def gaussian_template(sigma: float, width: t.Optional[int] = None):
    """
    Create a gaussian template of kernel width `width` and standard deviation `sigma`.
    If `width` is not specified it will be set to `sigma*4`.

    Return a [`ndarray`][numpy.ndarray]
    """
    # width should be odd
    w = width or 2*numpy.ceil(2*sigma).astype(int) + 1

    center = (w-1)/2.  # center gaussian on middle of pixel

    ys, xs = numpy.indices((w, w))
    #xs, ys = numpy.meshgrid(width, width)
    vs = numpy.exp(-((xs - center)**2 + (ys - center)**2)/(2*sigma**2)) \
        / (sigma * math.sqrt(2*math.pi))
    return vs


def integrate_peaks(img: NDArray[numpy.floating], peaks: ArrayLike, radius: float) -> NDArray[numpy.floating]:
    pts = numpy.moveaxis(numpy.indices(img.shape), 0, -1)
    tree = KDTree(peaks)
    dists, idxs = tree.query(pts, distance_upper_bound=radius)
    return numpy.bincount(idxs.ravel(), weights=img.ravel())[:-1].astype(numpy.float64)


def make_cutout_mask(cutout_rs: ArrayLike, peak_offsets: ArrayLike = (0., 0.)) -> t.Tuple[NDArray[numpy.int64], NDArray[numpy.int64], NDArray[numpy.int64], NDArray[numpy.bool_]]:
    cutout_rs = numpy.atleast_1d(cutout_rs)
    peak_offsets = numpy.atleast_2d(peak_offsets)

    cutout_min = numpy.floor(numpy.min(peak_offsets - cutout_rs[:, None], axis=0)).astype(int)
    cutout_max = numpy.ceil(numpy.max(peak_offsets + cutout_rs[:, None] + 1., axis=0)).astype(int)

    yy = numpy.arange(cutout_min[0], cutout_max[0])
    xx = numpy.arange(cutout_min[1], cutout_max[1])
    yy, xx = numpy.meshgrid(yy, xx, indexing='ij')

    mask = numpy.bitwise_or.reduce(
        (yy - peak_offsets[:, 0, None, None])**2 + (xx - peak_offsets[:, 1, None, None])**2 <= cutout_rs[:, None, None]**2,
        axis=0
    )

    return (yy, xx, cutout_min, mask)


def find_peaks(img, sigma, threshold_rel, min_distance=5) -> t.Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Find the peaks in `img`, using normalized cross correlation with a gaussian template.
    `sigma` is the peak standard deviation to use, `threshold_rel` is the minimum required
    cross correlation, and `min_distance` is the minimum distance between peaks.

    Uses cross correlation and [`peak_local_max`][skimage.feature.peak_local_max].

    Returns the cross_correlation image and the peak locations ([y, x]).
    """
    from skimage.feature import match_template, peak_local_max
    from skimage.filters import gaussian

    template = gaussian_template(sigma)
    # blur prefilter (TODO can this be combined into one step?)
    blurred = gaussian(img, sigma=1, preserve_range=True, truncate=3.0)
    cross_correlation = match_template(blurred, template, pad_input=True, mode='edge')

    peaks = peak_local_max(cross_correlation, min_distance, threshold_rel=threshold_rel,
                           exclude_border=2)  # type: ignore
    return (cross_correlation, peaks)


class PeakFitModel(ABC):
    """
    Superclass for arbitrary peak-fitting models.

    This class (or its subclasses) are usually not used directly.
    Instead, [`fit_peaks`][aci.peaks.fit_peaks] and [`fit_2d_peaks`][aci.peaks.fit_2d_peaks]
    are used as a high-level interface.
    """
    model: Model

    @staticmethod
    def background(x: numpy.ndarray, y: numpy.ndarray, bg=0.0):
        return bg

    def fit_peak(self, params: Parameters, xx: numpy.ndarray, yy: numpy.ndarray, vv: numpy.ndarray,
                 mask: t.Optional[numpy.ndarray] = None) -> ModelResult:
        """
        Given initial parameters `params` and data `xx`, `yy`, and `vv`, fit the model
        and return a [`ModelResult`][lmfit.model.ModelResult].
        """
        return self.model.fit(vv, params, x=xx, y=yy, method='leastsq', weights=mask)

    @abstractmethod
    def params(self) -> t.Iterable[str]:
        """List of params (along with `result.redchi` and `i`) to store for each peak."""
        ...

    @abstractmethod
    def post_process_results(self, a: t.List[t.Tuple[t.Any, ...]],
                             index_dtype: numpy.dtype = numpy.dtype('uint32')) -> numpy.ndarray:
        """
        Given a list of raw results (corresponding to `(*self.params(), 'redchi', 'i')`),
        construct a final output. Must return a numpy structured array.
        """
        ...

    def fit_img(self, img: numpy.ndarray, peaks: numpy.ndarray,
                cutout_r: ArrayLike = 10., shift_r: t.Optional[ArrayLike] = None,
                use_mask: bool = True) -> t.Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Fit `peaks` in `img`, using a cutout of radius `cutout_r` around each peak.

        Peaks are allowed to shift up to `shift_r` from their original position.
        Returns a tuple of `(results, img_minus_peaks)`, where `results` is a numpy
        structured array of fit results. If a peak fit fails, it will be excluded from
        `result`. Use the `i` field to access an original peak index.
        """
        img = img.astype('float')
        yy, xx = numpy.indices(img.shape)

        cutout_yy, cutout_xx, cutout_offset, mask = make_cutout_mask(cutout_r)

        if hasattr(peaks, 'dtype') and peaks.dtype.fields is not None:
            # also support input from structured array
            peaks = numpy.stack((peaks['x'], peaks['y']), axis=-1)
        peaks = numpy.atleast_2d(peaks)
        if peaks.shape[-1] != 2:
            raise ValueError(f"Expected a ndarray of shape (..., 2). Got shape {peaks.shape} instead.")

        # dtype for (possibly multidimensional) index
        input_dims = peaks.ndim - 1
        index_dtype = numpy.dtype(f"({input_dims},)uint32") if input_dims > 1 else numpy.dtype('uint32')

        # stores image minues other peak fits
        img_minus_peaks = img.copy()
        # stores fit tuples
        results: t.List[t.Tuple[t.Any, ...]] = []

        # max peak motion (defaults to 50% of cutout radius)
        shift_r = numpy.max(numpy.array(cutout_r) * 0.5 if shift_r is None else numpy.minimum(shift_r, cutout_r))

        for idx in numpy.ndindex(*peaks.shape[:-1]):
            peak = peaks[idx]
            # if 1d, flatten multidimensional idx
            idx = idx[0] if len(idx) == 1 else idx
            int_peak = tuple(map(math.floor, peak)) # [y, x]
            peak_intensity = float(img[tuple(int_peak)])

            # create the cutout around this peak
            index = tuple(slice(
                max(v+cutout_offset[i], 0),
                min(v+cutout_offset[i]+cutout_yy.shape[i], img.shape[i])
            ) for (i, v) in enumerate(int_peak))

            cutout = img[index]

            if use_mask and cutout.shape != mask.shape:
                # skip partial cutout
                continue

            params = self.model.make_params(peak=peak_intensity)
            # constrain x and y to within shift_r
            params['center_y'] = Parameter(name='center_y', value=peak[0], min=peak[0] - shift_r, max=peak[0] + shift_r)
            params['center_x'] = Parameter(name='center_x', value=peak[1], min=peak[1] - shift_r, max=peak[1] + shift_r)

            result = self.fit_peak(params, xx[index], yy[index], cutout, mask if use_mask else None)

            if not result.success:
                logging.warning(f"Failed to complete fit on peak at {tuple(peak)}")
                continue
            img_minus_peaks[index] -= result.best_fit

            # extract params from ModelResult
            results.append((*map(lambda p: result.params[p].value, self.params()), result.redchi, idx))

        return (self.post_process_results(results, index_dtype), img_minus_peaks)


class MultiPeakFitModel(PeakFitModel, ABC):
    @abstractmethod
    def peak_suffixes(self) -> t.Sequence[str]:
        """Suffixes of peaks to fit"""
        ...

    @abstractmethod
    def peak_offsets(self) -> NDArray[numpy.floating]:
        ...

    def fit_img(self, img: numpy.ndarray, peaks: numpy.ndarray,
                cutout_r: ArrayLike = 10., shift_r: t.Optional[ArrayLike] = None,
                use_mask: bool = True) -> t.Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Fit `peaks` in `img`, using a cutout of radius `cutout_r` around each peak.

        Peaks are allowed to shift up to `shift_r` from their original position.
        Returns a tuple of `(results, img_minus_peaks)`, where `results` is a numpy
        structured array of fit results. If a peak fit fails, it will be excluded from
        `result`. Use the `i` field to access an original peak index.
        """
        img = img.astype('float')
        yy, xx = numpy.indices(img.shape)

        peak_offsets = self.peak_offsets()
        cutout_yy, cutout_xx, cutout_offset, mask = make_cutout_mask(cutout_r, peak_offsets)

        if hasattr(peaks, 'dtype') and peaks.dtype.fields is not None:
            # also support input from structured array
            peaks = numpy.stack((peaks['x'], peaks['y']), axis=-1)
        peaks = numpy.atleast_2d(peaks)
        if peaks.shape[-1] != 2:
            raise ValueError(f"Expected a ndarray of shape (..., 2). Got shape {peaks.shape} instead.")

        # dtype for (possibly multidimensional) index
        input_dims = peaks.ndim - 1
        index_dtype = numpy.dtype(f"({input_dims},)uint32") if input_dims > 1 else numpy.dtype('uint32')

        # stores image minues other peak fits
        img_minus_peaks = img.copy()
        # stores fit tuples
        results: t.List[t.Tuple[t.Any, ...]] = []

        # max peak motion (defaults to 50% of cutout radius)
        shift_rs = numpy.array(cutout_r) * 0.5 if shift_r is None else numpy.minimum(shift_r, cutout_r)
        shift_rs = numpy.broadcast_to(shift_rs, len(self.peak_suffixes()))

        for idx in numpy.ndindex(*peaks.shape[:-1]):
            base_position = peaks[idx].astype(numpy.floating)
            int_position = tuple(map(math.floor, base_position))

            # create the cutout around this peak
            index = tuple(slice(
                max(v+cutout_offset[i], 0),
                min(v+cutout_offset[i]+cutout_yy.shape[i], img.shape[i])
            ) for (i, v) in enumerate(int_position))

            cutout = img[index]

            if use_mask and cutout.shape != mask.shape:
                # skip partial cutout
                continue

            # if 1d, flatten multidimensional idx
            idx = idx[0] if len(idx) == 1 else idx

            param_vals = {}
            param_replace = {}

            for offset, suffix, shift_r in zip(peak_offsets, self.peak_suffixes(), shift_rs):
                peak = base_position + offset
                int_peak = tuple(map(math.floor, peak))

                param_vals[f'peak_{suffix}'] = float(img[tuple(int_peak)])

                # constrain x and y to within shift_r
                param_replace[f'center_y_{suffix}'] = Parameter(name=f'center_y_{suffix}', value=peak[0], min=peak[0] - shift_r, max=peak[0] + shift_r)
                param_replace[f'center_x_{suffix}'] = Parameter(name=f'center_x_{suffix}', value=peak[1], min=peak[1] - shift_r, max=peak[1] + shift_r)

            params = self.model.make_params(**param_vals)
            for k, param in param_replace.items():
                params[k] = param

            result = self.fit_peak(params, xx[index], yy[index], cutout, mask if use_mask else None)

            if not result.success:
                logging.warn(f"Failed to complete fit on peak at {tuple(peak)}")
                continue
            img_minus_peaks[index] -= result.best_fit

            # extract params from ModelResult
            results.append((*map(lambda p: result.params[p].value, self.params()), result.redchi, idx))

        return (self.post_process_results(results, index_dtype), img_minus_peaks)


class GaussianPeakFit(PeakFitModel):
    """
    Axisymmetric Gaussian peak fitting model.
    """

    def __init__(self, sigma: float = 3., max_sigma: t.Optional[float] = None, use_bg: bool = False):
        self.sigma = sigma
        if max_sigma is None:
            max_sigma = numpy.inf

        self.model = Model(self.gaussian, ('x', 'y'))

        self.use_bg = use_bg
        if self.use_bg:
            # add background term
            self.model += Model(self.background, ('x', 'y'))
            self.model.set_param_hint('bg', value=0.)

        self.model.set_param_hint('peak', min=0)
        self.model.set_param_hint('sigma', min=1., value=sigma, max=max_sigma)

    @staticmethod
    def gaussian(x, y, center_x, center_y, peak, sigma):
        """
        Make a 2D Gaussian function centered at `(center_x, center_y)`,
        with peak `peak` and standard deviation `sigma`.
        """
        #prefactor = amplitude / (sigma * 2*math.pi)
        return peak * numpy.exp(((x - center_x)**2 + (y - center_y)**2) / (-2 * sigma**2))

    def params(self) -> t.Iterable[str]:
        """List of params (along with result.redchi and i) to store for each peak."""
        return ('center_x', 'center_y', 'sigma', 'peak', *yield_if(self.use_bg, 'bg'))

    def post_process_results(self, a: t.List[t.Tuple[t.Any, ...]],
                             index_dtype: numpy.dtype = numpy.dtype('uint32')) -> numpy.ndarray:
        """
        Given a list of raw results (corresponding to `(*self.params(), 'redchi', 'i')`),
        construct a final output. Must return a numpy structured array.

        In this case, post-processing calculates `amp` as a function of `peak`.
        """
        results = numpy.array(a, dtype=[
            ('x', 'd'), ('y', 'd'), ('sigma', 'd'), ('peak', 'd'),
            *yield_if(self.use_bg, ('bg', 'd')),
            ('redchi', 'd'), ('i', index_dtype),
        ])
        amp = results['peak'] * (results['sigma']**2 * 2*numpy.pi)
        return append_fields(results, 'amp', amp, numpy.double, usemask=False)


class Gaussian2dPeakFit(PeakFitModel):
    """
    Asymmetric Gaussian peak fitting model.
    """

    def __init__(self, sigma=3., max_sigma: t.Optional[float] = None, use_bg: bool = False):
        min_a = 1/(2 * abs(max_sigma)**2) if max_sigma is not None else 0.
        guess_a = 1 / (2. * sigma**2)
        max_a = 1. # 1/4 px radius

        # define model and indepedent variables
        self.model = Model(self.gaussian_2d, ('x', 'y'))

        self.use_bg = use_bg
        if self.use_bg:
            # add background term
            self.model += Model(self.background, ('x', 'y'))
            self.model.set_param_hint('bg', value=0.)

        # define parameters
        self.model.set_param_hint('peak', min=0.)
        # a, b, and c are the components of a positive-definite transformation matrix
        # b = b_rel * numpy.sqrt(a*c), |b_rel| < 1 ensures the matrix is positive-definite
        self.model.set_param_hint('a', min=min_a, max=max_a, value=guess_a)
        self.model.set_param_hint('b_rel', min=-0.99, max=0.99, value=0.)
        self.model.set_param_hint('c', min=min_a, max=max_a, value=guess_a)

    @staticmethod
    def gaussian_2d(x, y, center_x, center_y, peak, a, b_rel, c):
        """
        Given the indices `x` and `y`, calculate the 2d Gaussian function
        at (`center_x`, `center_y`) with peak intensity `peak` and coordinate
        transform matrix `[[a, b], [b, c]]`, where `b = b_rel * numpy.sqrt(a*c)`.
        """
        #print(f"gaussian({x}, {y}, {center_x}, {center_y}, {peak}, {a}, {b}, {c})")
        x = x - center_x
        y = y - center_y
        exponent = a*x**2 + 2*b_rel*numpy.sqrt(a*c)*x*y + c*y**2
        return peak * numpy.exp(-exponent)

    def params(self) -> t.Iterable[str]:
        """List of params (along with result.redchi and i) to store for each peak."""
        return ('center_x', 'center_y', 'peak', 'a', 'b_rel', 'c', *yield_if(self.use_bg, 'bg'))

    def post_process_results(self, arr: t.List[t.Tuple[t.Any, ...]],
                             index_dtype: numpy.dtype = numpy.dtype('uint32')) -> numpy.ndarray:
        """
        Given a list of raw results (corresponding to `(*self.params(), 'redchi', 'i')`),
        construct a final output. Must return a numpy structured array.

        In this case, post-processing takes the covariance matrix elements
        `a`, `b`, and `c`, and computes more relevant parameters (`sigma_x`,
        `sigma_y`, `sigma`, `sigma_1`, `sigma_2`, `ecc`, and `theta`).
        It additionally calculates `amp` given `peak`.
        """
        # TODO cleaner way to do this?
        results = numpy.array(arr, dtype=[
            ('x', 'd'), ('y', 'd'), ('peak', 'd'), ('a', 'd'), ('b_rel', 'd'), ('c', 'd'),
            *yield_if(self.use_bg, ('bg', 'd')),
            ('redchi', 'd'), ('i', index_dtype)
        ])
        (x, y, peak, a, b_rel, c, redchi, i) = (results[field] for field in ('x', 'y', 'peak', 'a', 'b_rel', 'c', 'redchi', 'i'))
        b = b_rel * numpy.sqrt(a*c)

        r = numpy.sqrt(a**2 + 4*b**2 - 2*a*c + c**2)
        a1 = (a + c - r)/2.  # major axis
        a2 = (a + c + r)/2.  # minor axis

        ecc = numpy.sqrt(1 - a1 / a2)  # eccentricity
        theta = numpy.pi/2 - numpy.arctan2(2*b, a - c + r)
        # theta is undefined for ecc = 0
        theta[ecc <= numpy.finfo(ecc.dtype).eps] = 0.

        sigma_x = 1/numpy.sqrt(2*a)
        sigma_y = 1/numpy.sqrt(2*c)
        sigma_1 = 1/numpy.sqrt(2*a1)
        sigma_2 = 1/numpy.sqrt(2*a2)
        # mean standard deviation
        sigma_m = (sigma_1 + sigma_2) / 2.

        # integrated intensity of image
        amp = results['peak'] * (2.*numpy.pi) * sigma_1 * sigma_2

        return make_structured_array(x=x, y=y, peak=peak, amp=amp, sigma=sigma_m,
                                     sigma_x=sigma_x, sigma_y=sigma_y, sigma_1=sigma_1,
                                     sigma_2=sigma_2, ecc=ecc, theta=theta,
                                     **(dict(bg=results['bg']) if self.use_bg else {}),
                                     redchi=redchi, i=i)


class LorentzianPeakFit(PeakFitModel):
    """
    Axisymmetric Lorentzian (Cauchy-Lorentz distribution) peak fitting model.

    Guesses for this model are specified with `sigma`, but this is not the standard deviation
    of the Lorentzian (which is infinite). Instead, this is converted to the HWHM of an equivalent
    Gaussian, and supplied as `gamma` (the HWHM of a Lorentzian).
    """

    GAUSSIAN_HWHM = numpy.sqrt(2 * numpy.log(2))

    def __init__(self, sigma: float = 3., max_sigma: t.Optional[float] = None, use_bg: bool = False):
        self.gamma = sigma * self.GAUSSIAN_HWHM
        max_gamma = numpy.inf if max_sigma is None else max_sigma * self.GAUSSIAN_HWHM

        self.model = Model(self.lorentzian, ['x', 'y'])

        self.use_bg = use_bg
        if self.use_bg:
            # add background term
            self.model += Model(self.background, ['x', 'y'])
            self.model.set_param_hint('bg', value=0.)

        self.model.set_param_hint('peak', min=0)
        self.model.set_param_hint('gamma', min=1., value=self.gamma, max=max_gamma)

    @staticmethod
    def lorentzian(x, y, center_x, center_y, peak, gamma):
        """
        Make a Lorentzian function centered at `(center_x, center_y)`,
        with peak intensity `peak` and HWHM `gamma`.
        """
        x = x - center_x
        y = y - center_y
        return peak * gamma**2 / (x**2 + y**2 + gamma**2)

    def params(self) -> t.Iterable[str]:
        """List of params (along with result.redchi and i) to store for each peak."""
        return ('center_x', 'center_y', 'peak', 'gamma', *yield_if(self.use_bg, 'bg'))

    def post_process_results(self, a: t.List[t.Tuple[t.Any, ...]],
                             index_dtype: numpy.dtype = numpy.dtype('uint32')) -> numpy.ndarray:
        r = numpy.array(a, dtype=[
            ('x', 'd'), ('y', 'd'), ('peak', 'd'), ('gamma', 'd'),
            *yield_if(self.use_bg, ('bg', 'd')),
            ('redchi', 'd'), ('i', index_dtype),
        ])
        amp = r['peak'] / (numpy.pi * r['gamma'])
        sigma = r['gamma'] / self.GAUSSIAN_HWHM
        return make_structured_array(i=r['i'], x=r['x'], y=r['y'], peak=r['peak'], amp=amp, sigma=sigma,
                                     gamma=r['gamma'], **(dict(bg=r['bg']) if self.use_bg else {}),
                                     redchi=r['redchi'])
        #return append_fields(results, ['sigma', 'amp'], [sigma, amp], [numpy.double, numpy.double], usemask=False)


class MultiGaussian2dPeakFit(MultiPeakFitModel):
    """
    Asymmetric Gaussian peak fitting model.
    """

    def peak_suffixes(self) -> t.Sequence[str]:
        return self._peak_suffixes

    def peak_offsets(self) -> NDArray[numpy.floating]:
        return self._peak_offsets

    def __init__(self, peak_offsets: ArrayLike, sigma: ArrayLike = 3., max_sigma: t.Optional[ArrayLike] = None, use_bg: bool = False,
                 names: t.Optional[t.Iterable[str]] = None):
        peak_offsets = numpy.atleast_2d(peak_offsets)
        assert peak_offsets.shape[-1] == 2
        self._peak_offsets: NDArray[numpy.floating] = peak_offsets
        self.n_peaks = peak_offsets.shape[0]

        self._peak_suffixes = tuple(names) if names is not None else tuple(map(str, range(self.n_peaks)))

        # define model and indepedent variables
        # TODO this is cursed because lmfit sucks
        sig = inspect.Signature([
            inspect.Parameter('x', inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter('y', inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in self._main_params()),
        ])
        self.model: Model = Model(
            _mock_signature(self.multi_gaussian_2d, sig),
            independent_vars=('x', 'y'),
            param_names=self._main_params(),
        )

        self.use_bg = use_bg
        if self.use_bg:
            # add background term
            self.model += Model(self.background, ('x', 'y'))
            self.model.set_param_hint('bg', value=0.)

        min_as = numpy.broadcast_to(1 / (2*numpy.abs(max_sigma)**2) if max_sigma is not None else 0., self.n_peaks)
        guess_as = numpy.broadcast_to(1 / (2*numpy.abs(sigma)**2), self.n_peaks)
        max_a = 1. # 1/4 px radius

        for suffix, min_a, guess_a in zip(self.peak_suffixes(), min_as, guess_as):
            # define parameters
            self.model.set_param_hint(f'peak_{suffix}', min=0.)
            # a, b, and c are the components of a positive-definite transformation matrix
            # b = b_rel * numpy.sqrt(a*c), |b_rel| < 1 ensures the matrix is positive-definite
            self.model.set_param_hint(f'a_{suffix}', min=min_a, max=max_a, value=guess_a)
            self.model.set_param_hint(f'b_rel_{suffix}', min=-0.99, max=0.99, value=0.)
            self.model.set_param_hint(f'c_{suffix}', min=min_a, max=max_a, value=guess_a)

    def multi_gaussian_2d(self, x, y, **kwargs):
        out = numpy.zeros_like(x)
        for suffix in self.peak_suffixes():
            out += Gaussian2dPeakFit.gaussian_2d(
                x, y,
                kwargs[f'center_x_{suffix}'],
                kwargs[f'center_y_{suffix}'],
                kwargs[f'peak_{suffix}'],
                kwargs[f'a_{suffix}'],
                kwargs[f'b_rel_{suffix}'],
                kwargs[f'c_{suffix}'],
            )
        return out

    def _main_params(self) -> t.Iterable[str]:
        return tuple(chain.from_iterable(
            (f'center_x_{suffix}', f'center_y_{suffix}', f'peak_{suffix}', f'a_{suffix}', f'b_rel_{suffix}', f'c_{suffix}')
            for suffix in self.peak_suffixes()
        ))

    def params(self) -> t.Iterable[str]:
        """List of params (along with result.redchi and i) to store for each peak."""
        return (*self._main_params(), *yield_if(self.use_bg, 'bg'))

    def post_process_results(self, arr: t.List[t.Tuple[t.Any, ...]],
                             index_dtype: numpy.dtype = numpy.dtype('uint32')) -> numpy.ndarray:
        """
        Given a list of raw results (corresponding to `(*self.params(), 'redchi', 'i')`),
        construct a final output. Must return a numpy structured array.

        In this case, post-processing takes the covariance matrix elements
        `a`, `b`, and `c`, and computes more relevant parameters (`sigma_x`,
        `sigma_y`, `sigma`, `sigma_1`, `sigma_2`, `ecc`, and `theta`).
        It additionally calculates `amp` given `peak`.
        """
        results = numpy.array(arr, dtype=[
            *((p, 'd') for p in self.params()),
            ('redchi', 'd'), ('i', index_dtype)
        ])

        d = {}

        for suffix in self.peak_suffixes():
            (x, y, peak, a, b_rel, c) = (
                results[f"{field}_{suffix}"] for field in ('center_x', 'center_y', 'peak', 'a', 'b_rel', 'c')
            )
            b = b_rel * numpy.sqrt(a*c)

            r = numpy.sqrt(a**2 + 4*b**2 - 2*a*c + c**2)
            a1 = (a + c - r)/2.  # major axis
            a2 = (a + c + r)/2.  # minor axis

            ecc = numpy.sqrt(1 - a1 / a2)  # eccentricity
            theta = numpy.pi/2 - numpy.arctan2(2*b, a - c + r)
            # theta is undefined for ecc = 0
            theta[ecc <= numpy.finfo(ecc.dtype).eps] = 0.

            sigma_x = 1/numpy.sqrt(2*a)
            sigma_y = 1/numpy.sqrt(2*c)
            sigma_1 = 1/numpy.sqrt(2*a1)
            sigma_2 = 1/numpy.sqrt(2*a2)
            # mean standard deviation
            sigma = (sigma_1 + sigma_2) / 2.

            # integrated intensity of image
            amp = peak * (2.*numpy.pi) * sigma_1 * sigma_2

            for field in ('x', 'y', 'peak', 'amp', 'sigma', 'sigma_x', 'sigma_y', 'sigma_1', 'sigma_2', 'ecc', 'theta'):
                # don't look
                d[f"{field}_{suffix}"] = locals()[field]

        if self.use_bg:
            d['bg'] = results['bg']

        d['redchi'] = results['redchi']
        d['i'] = results['i']

        return make_structured_array(**d)


def _mock_signature(f: t.Callable, sig: inspect.Signature) -> t.Callable:
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    setattr(wrapper, '__signature__', sig)
    return update_wrapper(wrapper, f)


PeakModel = t.Literal['gaussian', 'gaussian2d', 'lorentzian']
"""Supported peak fit models."""
_MODELS: t.Tuple[str] = tuple(PeakModel.__args__)  # type: ignore


def fit_peaks(img: numpy.ndarray, peaks: numpy.ndarray, model: PeakModel = 'gaussian', *,
              cutout_r: int = 10, shift_r: t.Optional[float] = None, sigma: float = 3.,
              max_sigma: t.Optional[float] = None, use_bg: bool = False) -> t.Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Fit peaks using lmfit least squares regression, with model `model`.

    #Parameters:

    - `cutout_r`: The radius, in pixels, of sub-images used to fit each peak.
    - `shift_r`: The maximum radius a peak is allowed to shift while fitting.
    - `sigma`: A guess for peak standard deviation.
    - `max_sigma`: Limit for maximum peak standard deviation.
    - `use_bg`: If `True`, a background term will be added to the peak fit.

    Not all options are used by all models.

    Status: Stable, options may change

    Returns a structured ndarray of shape `peaks.shape[:-1]`, and the image minus fit peaks.
    The columns present depend on the fitting method, but always include the following:

    - `i`: (possibly multidimensional) Peak index in the input array
    - `x`: Peak x position
    - `y`: Peak y position
    - `amp`: Integrated peak amplitude
    - `redchi`: Reduced chi-squared for the peak fit
    - `bg`: Background intensity. Present only if `use_bg` was enabled.

    In the case of 1d Gaussian peak fitting (the default), the following columns are also included:

    - `sigma`: Peak standard deviation
    - `peak`: Peak peak (maximum) intensity
    """
    model = t.cast(PeakModel, model.lower())
    if model == 'gaussian':
        fitter = GaussianPeakFit(sigma, max_sigma=max_sigma, use_bg=use_bg)
    elif model == 'gaussian2d':
        fitter = Gaussian2dPeakFit(sigma, max_sigma=max_sigma, use_bg=use_bg)
    elif model == 'lorentzian':
        fitter = LorentzianPeakFit(sigma, max_sigma=max_sigma, use_bg=use_bg)
    else:
        raise ValueError(f"Unknown peak fitting model '{model}'")
    
    return fitter.fit_img(img, peaks, cutout_r=cutout_r, shift_r=shift_r)


def fit_2d_peaks(img: numpy.ndarray, peaks: numpy.ndarray, model: PeakModel = 'gaussian2d', *,
                 cutout_r: int = 10, shift_r: t.Optional[float] = None, sigma: float = 3.,
                 max_sigma: t.Optional[float] = None, use_bg: bool = False) -> t.Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Fit peaks using lmfit least squares regression, with model `model`.
    Unlike `fit_peaks`, this function exclusively uses models which support asymmetric peaks.

    #Parameters:

    - `cutout_r`: The radius, in pixels, of sub-images used to fit each peak.
    - `shift_r`: The maximum radius a peak is allowed to shift while fitting.
    - `sigma`: A guess for peak standard deviation.
    - `max_sigma`: Limit for maximum peak standard deviation.
    - `use_bg`: If `True`, a background term will be added to the peak fit.

    Not all options are used by all models.

    Status: Stable, options may change

    The following columns are always returned:

    - `i`: (possibly multidimensional) Peak index in the input array
    - `x`: Peak x position
    - `y`: Peak y position
    - `amp`: Integrated peak amplitude
    - `ecc`: Peak eccentricity (0 for a circle, approaches 1 for a very distorted ellipse)
    - `theta`: Angle from horizontal to peak major axis, in the range [0, pi]
    - `redchi`: Reduced chi-squared for the peak fit
    - `bg`: Background intensity. Present only if `use_bg` was enabled.

    For 2d Gaussian peaks (the default), the following columns are also included:

    - `sigma`: Mean standard deviation
    - `sigma_1`: First principal standard deviation
    - `sigma_2`: Second principal standard deviation
    - `sigma_x`: Standard deviation in x direction
    - `sigma_y`: Standard deviation in y direction
    - `peak`: Peak peak (maximum) intensity
    """
    
    model = t.cast(PeakModel, model.lower())
    if model not in _MODELS:
        raise ValueError(f"Unknown peak fitting model '{model}'")

    if not model.endswith('2d'):
        # switch to 2d version of model (e.g. 'gaussian' -> 'gaussian2d')
        new_model = model + "2d"
        if new_model not in _MODELS:
            raise ValueError("Model '{model}' has no 2d variant.")
        model = t.cast(PeakModel, new_model)

    return fit_peaks(img, peaks, model, cutout_r=cutout_r, shift_r=shift_r,
                     sigma=sigma, max_sigma=max_sigma, use_bg=use_bg)


def fit_peaks_masked(img: numpy.ndarray, peaks: numpy.ndarray, model: PeakModel = 'gaussian', *,
                     cutout_r: int = 10, shift_r: t.Optional[float] = None, sigma: float = 3.,
                     max_sigma: t.Optional[float] = None, use_bg: bool = False) -> t.Tuple[numpy.ma.masked_array, numpy.ndarray]:
    if not isinstance(peaks, numpy.ma.masked_array):
        mask = numpy.zeros(peaks.shape[:-1], dtype=numpy.bool_)
        idxs = numpy.indices(peaks.shape[:-1])
        flat_peaks = peaks.flatten()
    else:
        mask = numpy.bitwise_or.reduce(peaks.mask, axis=-1)
        idxs = numpy.indices(peaks.shape[:-1])[:, ~mask]
        flat_peaks = peaks.data[~mask]

    flat_result, resids = fit_peaks(
        img, flat_peaks, model, cutout_r=cutout_r, shift_r=shift_r,
        sigma=sigma, max_sigma=max_sigma, use_bg=use_bg
    )

    result_idxs = flat_result['i']
    flat_result = drop_fields(flat_result, 'i')

    out = numpy.empty(peaks.shape[:-1], dtype=flat_result.dtype)
    out[tuple(idxs[:, result_idxs])] = flat_result
    out = numpy.ma.masked_array(out, mask=mask, fill_value=numpy.nan)

    return out, resids


def fit_2d_peaks_masked(img: numpy.ndarray, peaks: numpy.ndarray, model: PeakModel = 'gaussian2d', *,
                        cutout_r: int = 10, shift_r: t.Optional[float] = None, sigma: float = 3.,
                        max_sigma: t.Optional[float] = None, use_bg: bool = False) -> t.Tuple[numpy.ma.masked_array, numpy.ndarray]:
    if not isinstance(peaks, numpy.ma.masked_array):
        mask = numpy.zeros(peaks.shape[:-1], dtype=numpy.bool_)
        idxs = numpy.indices(peaks.shape[:-1])
        flat_peaks = peaks.flatten()
    else:
        mask = numpy.bitwise_or.reduce(peaks.mask, axis=-1)
        idxs = numpy.indices(peaks.shape[:-1])[:, ~mask]
        flat_peaks = peaks.data[~mask]

    flat_result, resids = fit_2d_peaks(
        img, flat_peaks, model, cutout_r=cutout_r, shift_r=shift_r,
        sigma=sigma, max_sigma=max_sigma, use_bg=use_bg
    )

    result_idxs = flat_result['i']
    flat_result = drop_fields(flat_result, 'i')

    out = numpy.full(peaks.shape[:-1], numpy.nan, dtype=flat_result.dtype)
    out[tuple(idxs[:, result_idxs])] = flat_result
    out = numpy.ma.masked_array(out, mask=mask, fill_value=numpy.nan)

    return out, resids


def make_structured_array(**kwargs: numpy.ndarray) -> numpy.ndarray:
    """
    Make a structured array with columns (in order) from `kwargs`.

    The first dimension of all arrays should be equal, and becomes
    the shape of the output structured array.
    Extra dimensions of values will be nested inside of that values' column.
    """
    # construct output dtype
    dtype = [(k, (v.dtype, v.shape[1:])) for (k, v) in kwargs.items()]
    lengths = list(map(len, kwargs.values()))
    length = max(lengths)

    if not all(length == lengths[0] for length in lengths):
        raise ValueError("Not all array lengths are equal")
    
    output = numpy.empty(shape=length, dtype=dtype)
    for (k, v) in kwargs.items():
        output[k] = v
    return output