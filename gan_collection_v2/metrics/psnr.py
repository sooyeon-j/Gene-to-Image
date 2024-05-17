from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from monai.metrics.regression import RegressionMetric
from monai.utils import MetricReduction

class PSNRMetric(RegressionMetric):
    r"""Compute Peak Signal To Noise Ratio between two tensors using function:

    .. math::
        \operatorname{PSNR}\left(Y, \hat{Y}\right) = 20 \cdot \log_{10} \left({\mathit{MAX}}_Y\right) \
        -10 \cdot \log_{10}\left(\operatorname{MSE\left(Y, \hat{Y}\right)}\right)

    More info: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Help taken from:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py line 4139

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        max_val: The dynamic range of the images/volumes (i.e., the difference between the
            maximum and the minimum allowed values e.g. 255 for a uint8 image).
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(
        self, max_val: int | float, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.max_val = max_val
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> Any:
        mse_out = self.compute_mean_error_metrics(y_pred, y, func=self.sq_func)
        return 20 * math.log10(self.max_val) - 10 * torch.log10(mse_out)
    
    def compute_mean_error_metrics(self, y_pred: torch.Tensor, y: torch.Tensor, func: Callable) -> torch.Tensor:
        # reducing in only channel + spatial dimensions (not batch)
        # reduction of batch handled inside __call__() using do_metric_reduction() in respective calling class
        flt = partial(torch.flatten, start_dim=0)
        return torch.mean(flt(func(y - y_pred)), dim=-1, keepdim=True)