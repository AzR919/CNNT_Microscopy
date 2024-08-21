import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torch import Tensor
from typing import Any, List, Optional, Sequence, Tuple, Union, Literal


def _gaussian_kernel_3d(channel: int, kernel_size: Sequence[int], sigma: Sequence[float], dtype: torch.dtype, device: torch.device) -> Tensor:
    """Creates a 3D Gaussian kernel."""
    coords = [torch.arange(size, dtype=dtype, device=device) for size in kernel_size]
    coords = [coord - (size - 1) / 2 for coord, size in zip(coords, kernel_size)]
    kernel = torch.exp(-0.5 * sum((coord ** 2 / s ** 2 for coord, s in zip(coords, sigma))))
    kernel = kernel / kernel.sum()
    kernel = kernel.expand(channel, 1, *kernel_size)
    return kernel


def _reflection_pad_3d(x: Tensor, pad_d: int, pad_w: int, pad_h: int) -> Tensor:
    """Pads a 5D tensor with reflection padding."""
    return F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode='reflect')


def _ssim_check_inputs(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Structural Similarity Index Measure."""
    if preds.dtype != target.dtype:
        target = target.to(preds.dtype)
    if preds.shape != target.shape:
        raise ValueError(f"Expected `preds` and `target` to have the same shape, but got {preds.shape} and {target.shape}.")
    if len(preds.shape) != 5:
        raise ValueError(f"Expected `preds` and `target` to have BxCxDxHxW shape, but got {preds.shape}.")
    return preds, target


class StructuralSimilarityIndexMeasure3D(Metric):
    """Compute Structural Similarity Index Measure (SSIM) for 3D images."""

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        data_range: Optional[Union[float, Tuple[float, float]]] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        return_full_image: bool = False,
        return_contrast_sensitivity: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        valid_reduction = ("elementwise_mean", "sum", "none", None)
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")

        if reduction in ("elementwise_mean", "sum"):
            self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("similarity", default=[], dist_reduce_fx="cat")

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        if return_contrast_sensitivity or return_full_image:
            self.add_state("image_return", default=[], dist_reduce_fx="cat")

        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.return_full_image = return_full_image
        self.return_contrast_sensitivity = return_contrast_sensitivity

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _ssim_check_inputs(preds, target)
        similarity_pack = self._ssim_update(preds, target)

        if isinstance(similarity_pack, tuple):
            similarity, image = similarity_pack
        else:
            similarity = similarity_pack

        if self.return_contrast_sensitivity or self.return_full_image:
            self.image_return.append(image)

        if self.reduction in ("elementwise_mean", "sum"):
            self.similarity += similarity.sum()
            self.total += preds.shape[0]
        else:
            self.similarity.append(similarity)

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute SSIM over state."""
        if self.reduction == "elementwise_mean":
            similarity = self.similarity / self.total
        elif self.reduction == "sum":
            similarity = self.similarity
        else:
            similarity = torch.cat(self.similarity, dim=0)

        if self.return_contrast_sensitivity or self.return_full_image:
            image_return = torch.cat(self.image_return, dim=0)
            return similarity, image_return

        return similarity

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[Any] = None
    ) -> Any:
        """Plot a single or multiple values from the metric."""
        return self._plot(val, ax)

    def _ssim_update(
        self,
        preds: Tensor,
        target: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute Structural Similarity Index Measure."""
        if not isinstance(self.kernel_size, Sequence):
            self.kernel_size = 3 * [self.kernel_size]
        if not isinstance(self.sigma, Sequence):
            self.sigma = 3 * [self.sigma]

        if len(self.kernel_size) != 3:
            raise ValueError(f"`kernel_size` should have 3 dimensions, but got {len(self.kernel_size)}")
        if len(self.sigma) != 3:
            raise ValueError(f"`sigma` should have 3 dimensions, but got {len(self.sigma)}")

        if self.return_full_image and self.return_contrast_sensitivity:
            raise ValueError("Arguments `return_full_image` and `return_contrast_sensitivity` are mutually exclusive.")

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected `kernel_size` to have odd positive numbers. Got {self.kernel_size}")

        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected `sigma` to have positive numbers. Got {self.sigma}")

        if self.data_range is None:
            self.data_range = max(preds.max() - preds.min(), target.max() - target.min())
        elif isinstance(self.data_range, tuple):
            preds = torch.clamp(preds, min=self.data_range[0], max=self.data_range[1])
            target = torch.clamp(target, min=self.data_range[0], max=self.data_range[1])
            self.data_range = self.data_range[1] - self.data_range[0]

        #print(f"Data Range: {self.data_range}")

        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2
        device = preds.device

        #print(f"C1: {c1}, C2: {c2}")

        channel = preds.size(1)
        dtype = preds.dtype
        gauss_kernel_size = [int(3.5 * s + 0.5) * 2 + 1 for s in self.sigma]

        pad_h = (gauss_kernel_size[0] - 1) // 2
        pad_w = (gauss_kernel_size[1] - 1) // 2
        pad_d = (gauss_kernel_size[2] - 1) // 2

        preds = _reflection_pad_3d(preds, pad_d, pad_w, pad_h)
        target = _reflection_pad_3d(target, pad_d, pad_w, pad_h)

        if self.gaussian_kernel:
            kernel = _gaussian_kernel_3d(channel, gauss_kernel_size, self.sigma, dtype, device)
        else:
            kernel = torch.ones((channel, 1, *self.kernel_size), dtype=dtype, device=device) / torch.prod(
                torch.tensor(self.kernel_size, dtype=dtype, device=device)
            )

        input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))

        outputs = F.conv3d(input_list, kernel, groups=channel)

        output_list = outputs.split(preds.shape[0])

        mu_pred = output_list[0]
        mu_target = output_list[1]
        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = torch.clamp(output_list[2] - mu_pred_sq, min=0.0)
        sigma_target_sq = torch.clamp(output_list[3] - mu_target_sq, min=0.0)
        sigma_pred_target = torch.clamp(output_list[4] - mu_pred_target, min=0.0)

        #print(f"mu_pred: {mu_pred.mean().item()}, mu_target: {mu_target.mean().item()}, mu_pred_target: {mu_pred_target.mean().item()}")
        #print(f"sigma_pred_sq: {sigma_pred_sq.mean().item()}, sigma_target_sq: {sigma_target_sq.mean().item()}, sigma_pred_target: {sigma_pred_target.mean().item()}")

        upper = 2 * sigma_pred_target.to(dtype) + c2
        lower = (sigma_pred_sq + sigma_target_sq).to(dtype) + c2

        #print(f"Upper: {upper.mean().item()}, Lower: {lower.mean().item()}")

        ssim_idx_full_image = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)

        ssim_idx = ssim_idx_full_image[..., pad_h:-pad_h, pad_w:-pad_w, pad_d:-pad_d]

        if self.return_contrast_sensitivity:
            contrast_sensitivity = upper / lower
            contrast_sensitivity = contrast_sensitivity[..., pad_h:-pad_h, pad_w:-pad_w, pad_d:-pad_d]
            return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1), contrast_sensitivity.reshape(
                contrast_sensitivity.shape[0], -1
            ).mean(-1)

        if self.return_full_image:
            return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1), ssim_idx_full_image

        return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1)


# Example Usage for 3D SSIM
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssim3d = StructuralSimilarityIndexMeasure3D(data_range=1.0).to(device)

    # Example 3D tensors (batch_size, channels, depth, height, width)
    preds = torch.rand([3, 1, 32, 256, 256], device=device)
    target = preds * 0.75

    # Update and compute SSIM
    ssim3d.update(preds, target)
    ssim_value = ssim3d.compute()

    print(f"SSIM3D: {ssim_value.item()}")