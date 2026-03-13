import torch.nn.functional as F
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from gridfm_graphkit.io.registries import LOSS_REGISTRY
from torch_scatter import scatter_add

from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    QG_H,
    VM_H,
    VA_H,
    QD_H,
    PD_H,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    QG_OUT,
    PG_OUT,
    # Generator feature indices
    PG_H,
    C0_H,
    C1_H,
    C2_H,
    # Qg Limits
    MIN_QG_H, 
    MAX_QG_H,
)


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all custom loss functions.
    """

    @abstractmethod
    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        """
        Compute the loss.

        Parameters:
        - pred: Predictions.
        - target: Ground truth.
        - edge_index: Optional edge index for graph-based losses.
        - edge_attr: Optional edge attributes for graph-based losses.
        - mask: Optional mask to filter the inputs for certain losses.
        - model: Optional model reference for accessing internal states.

        Returns:
        - A dictionary with the total loss and any additional metrics.
        """
        pass


@LOSS_REGISTRY.register("MaskedMSE")
class MaskedMSELoss(BaseLoss):
    """
    Mean Squared Error loss computed only on masked elements.
    """

    def __init__(self, loss_args, args):
        super(MaskedMSELoss, self).__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        loss = F.mse_loss(pred[mask], target[mask], reduction=self.reduction)
        return {"loss": loss, "Masked MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedGenMSE")
class MaskedGenMSE(torch.nn.Module):
    def __init__(self, loss_args, args):
        super().__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        loss = F.mse_loss(
            pred_dict["gen"][mask_dict["gen"][:, : (PG_H + 1)]],
            target_dict["gen"][mask_dict["gen"][:, : (PG_H + 1)]],
            reduction=self.reduction,
        )
        return {"loss": loss, "Masked generator MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedBusMSE")
class MaskedBusMSE(torch.nn.Module):
    def __init__(self, loss_args, args):
        super().__init__()
        self.reduction = "mean"
        self.args = args

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        if self.args.task == "OptimalPowerFlow":
            pred_cols = [VM_OUT, VA_OUT, QG_OUT]
            target_cols = [VM_H, VA_H, QG_H]
        else:
            pred_cols = [VM_OUT, VA_OUT]
            target_cols = [VM_H, VA_H]

        pred_bus = pred_dict["bus"][:, pred_cols]  # shape: [N, 3]
        target_bus = target_dict["bus"][:, target_cols]

        mask = mask_dict["bus"][:, target_cols]

        loss = F.mse_loss(
            pred_bus[mask],
            target_bus[mask],
            reduction=self.reduction,
        )
        return {"loss": loss, "Masked bus MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MSE")
class MSELoss(BaseLoss):
    """Standard Mean Squared Error loss."""

    def __init__(self, loss_args, args):
        super(MSELoss, self).__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return {"loss": loss, "MSE loss": loss.detach()}


class MixedLoss(BaseLoss):
    """
    Combines multiple loss functions with weighted sum.

    Args:
        loss_functions (list[nn.Module]): List of loss functions.
        weights (list[float]): Corresponding weights for each loss function.
    """

    def __init__(self, loss_functions, weights):
        super(MixedLoss, self).__init__()

        if len(loss_functions) != len(weights):
            raise ValueError(
                "The number of loss functions must match the number of weights.",
            )

        self.loss_functions = nn.ModuleList(loss_functions)
        self.weights = weights

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        """
        Compute the weighted sum of all specified losses.

        Parameters:

        - pred: Predictions.
        - target: Ground truth.
        - edge_index: Optional edge index for graph-based losses.
        - edge_attr: Optional edge attributes for graph-based losses.
        - mask: Optional mask to filter the inputs for certain losses.

        Returns:
        - A dictionary with the total loss and individual losses.
        """
        total_loss = 0.0
        loss_details = {}

        for i, loss_fn in enumerate(self.loss_functions):
            loss_output = loss_fn(
                pred,
                target,
                edge_index,
                edge_attr,
                mask,
                model,
                x_dict,
                batch_dict,
            )

            # Assume each loss function returns a dictionary with a "loss" key
            individual_loss = loss_output.pop("loss")
            weighted_loss = self.weights[i] * individual_loss

            total_loss += weighted_loss

            # Add other keys from the loss output to the details
            for key, val in loss_output.items():
                loss_details[key] = val

        loss_details["loss"] = total_loss
        return loss_details


@LOSS_REGISTRY.register("LayeredWeightedPhysics")
class LayeredWeightedPhysicsLoss(BaseLoss):
    def __init__(self, loss_args, args) -> None:
        super().__init__()
        self.base_weight = loss_args.base_weight

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        total_loss = 0.0
        loss_details = {}

        layer_keys = sorted(model.layer_residuals.keys())
        L = len(layer_keys)

        # Compute raw weights (geometric decay)
        raw_weights = [self.base_weight ** (L - idx - 1) for idx in range(L)]

        # Normalize so weights sum to 1
        weight_sum = sum(raw_weights)
        norm_weights = [w / weight_sum for w in raw_weights]

        for key, weight in zip(layer_keys, norm_weights):
            residual = model.layer_residuals[key]
            total_loss = total_loss + weight * residual
            loss_details[f"layer_{key}_residual"] = residual.item()
            loss_details[f"layer_{key}_weight"] = weight

        loss_details["loss"] = total_loss
        loss_details["Layered Weighted Physics Loss"] = total_loss.item()
        return loss_details


@LOSS_REGISTRY.register("LossPerDim")
class LossPerDim(BaseLoss):
    def __init__(self, loss_args, args):
        super(LossPerDim, self).__init__()
        self.reduction = "mean"
        self.loss_str = loss_args.loss_str
        self.dim = loss_args.dim
        if self.dim not in ["VM", "VA", "P_in", "Q_in"]:
            raise ValueError(
                f"LossPerDim initialized with not valid dim: {self.dim}",
            )

        elif self.loss_str not in ["MAE", "MSE"]:
            raise ValueError(
                f"LossPerDim initialized with not valid loss_str: {self.loss_str}",
            )

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        if self.dim == "VM":
            temp_pred = pred_dict["bus"][:, VM_OUT]
            temp_target = target_dict["bus"][:, VM_H]
        elif self.dim == "VA":
            temp_pred = pred_dict["bus"][:, VA_OUT]
            temp_target = target_dict["bus"][:, VA_H]
        elif self.dim == "P_in":
            temp_pred = pred_dict["bus"][:, PG_OUT]
            num_bus = temp_pred.size(0)
            gen_to_bus_index = edge_index[("gen", "connected_to", "bus")]
            temp_gen = scatter_add(
                target_dict["gen"][:, PG_H],
                gen_to_bus_index[1, :],
                dim=0,
                dim_size=num_bus,
            )
            temp_target = temp_gen - target_dict["bus"][:, PD_H]
        elif self.dim == "Q_in":
            temp_pred = pred_dict["bus"][:, QG_OUT]
            temp_target = target_dict["bus"][:, QG_H] - target_dict["bus"][:, QD_H]

        mse_loss = F.mse_loss(temp_pred, temp_target, reduction=self.reduction)
        mae_loss = F.l1_loss(temp_pred, temp_target, reduction=self.reduction)

        loss = mse_loss if self.loss_str == "mse" else mae_loss
        return {
            "loss": loss,
            f"MSE loss {self.dim}": mse_loss.detach(),
            f"MAE loss {self.dim}": mae_loss.detach(),
        }


@LOSS_REGISTRY.register("QgViolationPenalty")
class QgViolationPenaltyLoss(BaseLoss):
    """Standard Mean Squared Error loss."""

    def __init__(self, loss_args, args):
        super().__init__()

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        # --- Qg limit violation mask ---
        Qg_pred = pred["bus"][:, QG_OUT]
        Qg_max = x_dict["bus"][:, MAX_QG_H]
        Qg_min = x_dict["bus"][:, MIN_QG_H]

        max_penalty_mask = (Qg_pred > Qg_max) 
        min_penalty_mask = (Qg_pred < Qg_min)

        mask_PQ = mask["PQ"]  # PQ buses
        mask_PV = mask["PV"]  # PV buses
        mask_REF = mask["REF"]  # Reference buses

        loss = 0.0
        # where there are violations, compute penalty loss
        Qg_over = F.relu(Qg_pred - Qg_max)  # amount above max limit
        Qg_under = F.relu(Qg_min - Qg_pred)  # amount below min limit

        Qg_over = Qg_over[max_penalty_mask].mean()
        Qg_under = Qg_under[min_penalty_mask].mean()
        
        if Qg_over!=Qg_over: # replacing nan with 0 
            Qg_over = 0.0
        if Qg_under!=Qg_under: # replacing nan with 0 
            Qg_under = 0.0

        penalty_loss = Qg_over + Qg_under            
        loss += penalty_loss

        try:
            output = {"loss": loss, "Qg Violation Penalty loss": loss.detach()}
        except:
            output = {"loss": loss, "Qg Violation Penalty loss": loss}

        return output




@LOSS_REGISTRY.register("QgViolationBarrier")
class QgViolationBarrierLoss(BaseLoss):
    """
    QgViolation Barrier loss function.
    * https://en.wikipedia.org/wiki/Barrier_function
    Available barrier functions are defined in the self.barriers dictionary.
    References for relaxed barrier functions:
    * https://arxiv.org/abs/1602.01321
    * https://arxiv.org/abs/1904.04205v2
    * https://ieeexplore.ieee.org/document/7493643/

    Modified from https://github.com/pnnl/neuromancer
    Copyright © 2021, Battelle Memorial Institute
    https://github.com/pnnl/neuromancer/blob/master/LICENSE.md 
    """
    def __init__(self, loss_args, args):
        super().__init__()

        self.barrier_name = getattr(loss_args, "barrier", 'log10')
        self.shift = getattr(loss_args, "shift", 1)
        self.alpha = getattr(loss_args, "alpha", 0.5)
        self.upper_bound = getattr(loss_args, "upper_bound", 1.0)

        # choices of barrier functions
        #   warning: log10, log, inverse, and softlog might get numerically unstable
        #   softexp is numerically stable and thus a prefered option
        self.barriers = {
                'log10': lambda value: -torch.log10(-value),
                'log': lambda value: -torch.log(-value),
                'inverse': lambda value: 1 / (-value),
                'softexp': lambda value: (torch.exp(self.alpha * value) - 1) / self.alpha + self.alpha,
                'softlog': lambda value: -torch.log(1 + self.alpha * (-value - self.alpha)) / self.alpha,
                'expshift': lambda value: torch.exp(value + self.shift)
                    }
        self.barrier = self._set_barrier()

    def _set_barrier(self):
        if self.barrier_name in self.barriers:
            return self.barriers[self.barrier_name]
        else:
            assert callable(barrier), \
                f'The barrier, {barrier} must be a key in {self.barriers} or a callable.'
            return barrier

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        """
        Calculate the magnitudes of constraint violations via log barriers
            cviolation > 0 -> penalty (i.e. beyond constraints)
            cviolation <= 0 -> barrier (i.e. within constraints)
        """

        # --- Qg limit violation mask ---
        Qg_pred = pred["bus"][:, QG_OUT]
        Qg_max = x_dict["bus"][:, MAX_QG_H]
        Qg_min = x_dict["bus"][:, MIN_QG_H]

        max_penalty_mask = (Qg_pred > Qg_max) 
        min_penalty_mask = (Qg_pred < Qg_min)

        mask_PQ = mask["PQ"]  # PQ buses
        mask_PV = mask["PV"]  # PV buses
        mask_REF = mask["REF"]  # Reference buses

        loss = 0.0
        
        if max_penalty_mask.any() or min_penalty_mask.any():
            # where there are violations, compute penalty loss
            Qg_over = F.relu(Qg_pred - Qg_max)  # amount above max limit
            Qg_under = F.relu(Qg_min - Qg_pred)  # amount below min limit

            Qg_over = Qg_over[max_penalty_mask].mean()
            Qg_under = Qg_under[min_penalty_mask].mean()

            if Qg_over!=Qg_over: # replacing nan with 0 
                Qg_over = 0.0
            if Qg_under!=Qg_under: # replacing nan with 0 
                Qg_under = 0.0

            penalty_loss = Qg_over + Qg_under            
            loss += penalty_loss

        if (~max_penalty_mask).any() or (~min_penalty_mask).any():
            Qg_barrier_amount_max = Qg_pred - Qg_max
            Qg_barrier_amount_min = Qg_min - Qg_pred
            
            cbarrier_max = self.barrier(Qg_barrier_amount_max)
            cbarrier_max[cbarrier_max != cbarrier_max] = 0.0  # replacing nan with 0 -> infeasibility
            cbarrier_max[cbarrier_max == float("Inf")] = 0.0  # replacing inf with 0 -> active constraints
            cbarrier_max = torch.clamp(cbarrier_max, min=0.0, max=self.upper_bound)
            barrier_loss_max = cbarrier_max[~max_penalty_mask].mean()
            
            cbarrier_min = self.barrier(Qg_barrier_amount_min)
            cbarrier_min[cbarrier_min != cbarrier_min] = 0.0  # replacing nan with 0 -> infeasibility
            cbarrier_min[cbarrier_min == float("Inf")] = 0.0  # replacing inf with 0 -> active constraints
            cbarrier_min = torch.clamp(cbarrier_min, min=0.0, max=self.upper_bound)
            barrier_loss_min = cbarrier_min[~min_penalty_mask].mean()

            barrier_loss = barrier_loss_min + barrier_loss_max
            loss+=barrier_loss

        return {"loss": loss, "Qg Violation Barrier loss": loss.detach()}



@LOSS_REGISTRY.register("OptimalityLoss")
class OptimalityLoss(BaseLoss):
    """
    """

    def __init__(self, loss_args, args):
        super().__init__()

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
        x_dict=None,
        batch_dict=None,
    ):
        c0 = x_dict["gen"][:, C0_H]
        c1 = x_dict["gen"][:, C1_H]
        c2 = x_dict["gen"][:, C2_H]
        target_pg = target["gen"].squeeze()
        pred_pg = pred["gen"].squeeze()
        gen_cost_gt = c0 + c1 * target_pg + c2 * target_pg**2
        gen_cost_pred = c0 + c1 * pred_pg + c2 * pred_pg**2

        gen_batch = batch_dict["gen"]  # shape: [N_gen_total]

        cost_gt = scatter_add(gen_cost_gt, gen_batch, dim=0)
        cost_pred = scatter_add(gen_cost_pred, gen_batch, dim=0)

        loss = torch.mean(torch.abs((cost_pred - cost_gt) / cost_gt * 100))
        if  loss!=loss:
            loss=0.0

        try:
            output = {"loss": loss, "Optimality loss": loss.detach()}
        except:
            output = {"loss": loss, "Optimality loss": loss}

        return output
