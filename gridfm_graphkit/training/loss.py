import torch.nn.functional as F
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from gridfm_graphkit.io.registries import LOSS_REGISTRY
from torch_scatter import scatter_add
from torch_geometric.utils import to_torch_coo_tensor

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
    PD_OUT,
    QD_OUT,
    # Generator feature indices
    PG_H,
    # Edge feature indices
    YFF_TT_R,
    YFF_TT_I,
    YFT_TF_R,
    YFT_TF_I,
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


@LOSS_REGISTRY.register("MaskedReconstructionMSE")
class MaskedReconstructionMSE(BaseLoss):
    """Unified masked MSE over bus-level quantities [VM, VA, PG, QG, PD, QD].

    Mirrors the homogeneous reference MaskedMSE by combining bus predictions
    and aggregated generator PG into a single prediction/target/mask tensor.
    PG targets are aggregated from generator ground truth onto buses via
    scatter_add; the bus-level PG mask is True when any generator at the bus
    is masked, indicating that the model must reconstruct that quantity.

    Replaces the separate MaskedBusMSE + MaskedGenMSE pair.
    Requires output_bus_dim >= 6 so the bus head predicts
    [VM, VA, PG, QG, PD, QD].
    """

    def __init__(self, loss_args, args):
        super().__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index_dict,
        edge_attr_dict,
        mask_dict,
        model=None,
    ):
        pred_bus = pred_dict["bus"]
        target_bus = target_dict["bus"]
        num_bus = target_bus.size(0)
        gen_to_bus_ei = edge_index_dict[("gen", "connected_to", "bus")]

        # --- Build target: [VM, VA, PG_agg, QG, PD, QD] ---
        target_pg_agg = scatter_add(
            target_dict["gen"][:, PG_H],
            gen_to_bus_ei[1],
            dim=0,
            dim_size=num_bus,
        )
        target = torch.stack([
            target_bus[:, VM_H],
            target_bus[:, VA_H],
            target_pg_agg,
            target_bus[:, QG_H],
            target_bus[:, PD_H],
            target_bus[:, QD_H],
        ], dim=1)

        # --- Build mask: [N_bus, 6] ---
        # PG bus-level mask: True if any generator at the bus has PG masked
        gen_pg_masked = mask_dict["gen"][:, PG_H].float()
        any_gen_masked = scatter_add(
            gen_pg_masked,
            gen_to_bus_ei[1],
            dim=0,
            dim_size=num_bus,
        ) > 0

        mask = torch.stack([
            mask_dict["bus"][:, VM_H],
            mask_dict["bus"][:, VA_H],
            any_gen_masked,
            mask_dict["bus"][:, QG_H],
            mask_dict["bus"][:, PD_H],
            mask_dict["bus"][:, QD_H],
        ], dim=1)

        # --- Prediction: [VM, VA, PG, QG, PD, QD] from bus head ---
        pred = pred_bus[:, [VM_OUT, VA_OUT, PG_OUT, QG_OUT, PD_OUT, QD_OUT]]

        loss = F.mse_loss(pred[mask], target[mask], reduction=self.reduction)
        return {"loss": loss, "Masked reconstruction MSE loss": loss.detach()}


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
    
@LOSS_REGISTRY.register("PBE")
class PBELoss(BaseLoss):
    """
    Loss based on the Power Balance Equations.

    Adapted for the heterogeneous graph convention: predictions and targets
    are passed as dicts (``{"bus": …, "gen": …}``).  Generator active power
    is aggregated onto bus nodes via the ``(gen, connected_to, bus)`` edge
    index before computing the power balance.
    """

    def __init__(self, loss_args, args):
        super(PBELoss, self).__init__()
        self.visualization = getattr(loss_args, "visualization", False)

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index_dict,
        edge_attr_dict,
        mask_dict,
        model=None,
    ):
        pred_bus = pred_dict["bus"]          # [N_bus, output_bus_dim]
        target_bus = target_dict["bus"]      # [N_bus, bus_feat_dim]
        num_bus = target_bus.size(0)

        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]
        mask_bus = mask_dict["bus"]

        # --- Voltage: use prediction where masked, target where known ---
        Vm_pred = pred_bus[:, VM_OUT]
        Va_pred = pred_bus[:, VA_OUT]
        Vm_target = target_bus[:, VM_H]
        Va_target = target_bus[:, VA_H]

        mask_Vm = mask_bus[:, VM_H]
        mask_Va = mask_bus[:, VA_H]

        V_m = torch.where(mask_Vm, Vm_pred, Vm_target)
        V_a = torch.where(mask_Va, Va_pred, Va_target)

        # Complex voltage
        V = V_m * torch.exp(1j * V_a)
        V_conj = torch.conj(V)

        # --- Admittance matrix from bus-bus edge attrs ---
        # Use Yff (diagonal-block) real/imag as the admittance entries
        edge_complex = bus_edge_attr[:, YFF_TT_R] + 1j * bus_edge_attr[:, YFF_TT_I]

        Y_bus_sparse = to_torch_coo_tensor(
            bus_edge_index,
            edge_complex,
            size=(num_bus, num_bus),
        )
        Y_bus_conj = torch.conj(Y_bus_sparse)

        # Complex power injection:  S_inj = diag(V) * conj(Y) * conj(V)
        S_injection = torch.diag(V) @ Y_bus_conj @ V_conj

        # --- Net power from predictions/targets ---
        # Pg: aggregate generator predictions onto buses
        gen_to_bus_ei = edge_index_dict[("gen", "connected_to", "bus")]
        Pg_per_bus = scatter_add(
            pred_dict["gen"].squeeze(-1),
            gen_to_bus_ei[1],
            dim=0,
            dim_size=num_bus,
        )

        Pd = target_bus[:, PD_H]
        Qd = target_bus[:, QD_H]

        # Qg: use prediction if the model predicts it, else use target
        if pred_bus.size(1) > QG_OUT:
            Qg = torch.where(mask_bus[:, QG_H], pred_bus[:, QG_OUT], target_bus[:, QG_H])
        else:
            Qg = target_bus[:, QG_H]

        net_P = Pg_per_bus - Pd
        net_Q = Qg - Qd
        S_net = net_P + 1j * net_Q

        # --- Loss ---
        loss = torch.mean(torch.abs(S_net - S_injection))

        real_loss = torch.mean(
            torch.abs(torch.real(S_net - S_injection)),
        )
        imag_loss = torch.mean(
            torch.abs(torch.imag(S_net - S_injection)),
        )

        result = {
            "loss": loss,
            "Power loss in p.u.": loss.detach(),
            "Active Power Loss in p.u.": real_loss.detach(),
            "Reactive Power Loss in p.u.": imag_loss.detach(),
        }
        if self.visualization:
            result["Nodal Active Power Loss in p.u."] = torch.abs(
                torch.real(S_net - S_injection),
            )
            result["Nodal Reactive Power Loss in p.u."] = torch.abs(
                torch.imag(S_net - S_injection),
            )
        return result
