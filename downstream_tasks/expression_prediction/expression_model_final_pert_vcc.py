"""Thin composite-loss adapter around the stock ``final_pert`` model.

The original architecture remains in ``expression_model_final_pert.py``. This
module only adds exact activation-checkpoint microbatching over perturbations
and evaluates the coupled composite loss after restoring the complete G x P
prediction rectangle.
"""
from __future__ import annotations

import torch

from downstream_tasks.expression_prediction.expression_model_final_pert import (
    ExpressionCounts,
    ExpressionModelOutput,
)
from downstream_tasks.expression_prediction.vcc_composite_loss import VCCBatch
from downstream_tasks.expression_prediction.distributed_vcc import (
    distributed_model_forward,
)


class VCCExpressionCounts(ExpressionCounts):
    def __init__(self, *args, micro_perturbations: int = 4, **kwargs):
        vcc_loss = kwargs.pop("vcc_loss", None)
        super().__init__(*args, **kwargs)
        if vcc_loss is None:
            raise ValueError("VCCExpressionCounts requires model_kwargs.vcc_loss")
        self.vcc_loss = vcc_loss
        if micro_perturbations < 1:
            raise ValueError("micro_perturbations must be >= 1")
        self.micro_perturbations = int(micro_perturbations)

    def forward_base(self, **kwargs):
        """Call the unchanged stock ``final_pert`` model forward."""
        return super().forward(**kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_mask=None,
        labels=None,
        return_dict=None,
        tf_ids=None,
        tf_attention_mask=None,
        dataset_flag=None,
        control_value=None,
        target_gene_row=None,
        lfc_true=None,
        sig_mask=None,
        **kwargs,
    ):
        if dataset_flag is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_mask=labels_mask,
                labels=labels,
                return_dict=return_dict,
                tf_ids=tf_ids,
                tf_attention_mask=tf_attention_mask,
                dataset_flag=dataset_flag,
                **kwargs,
            )

        genes, perturbations = dataset_flag.shape
        if not dataset_flag.bool().all():
            raise ValueError("composite loss requires dataset_flag == 1")

        def grid(value):
            if value is None:
                return None
            if value.shape[:2] == (genes, perturbations):
                return value
            return value.reshape(genes, perturbations, *value.shape[1:])

        ids = grid(input_ids)
        attention = grid(attention_mask)
        tf_grid = grid(tf_ids)
        tf_mask_grid = grid(tf_attention_mask)

        logits_grid = distributed_model_forward(
            self,
            input_ids=ids,
            attention_mask=attention,
            tf_ids=tf_grid,
            tf_attention_mask=tf_mask_grid,
            dataset_flag=dataset_flag,
            micro_p=self.micro_perturbations,
        )
        logits = logits_grid.reshape(genes * perturbations, *logits_grid.shape[2:])
        if labels is None:
            return ExpressionModelOutput(logits=logits)

        labels_reshaped = labels.reshape(genes * perturbations, *labels.shape[-2:]).to(logits.device)
        masks_reshaped = labels_mask.reshape(
            genes * perturbations, *labels_mask.shape[-2:]
        ).to(logits.device)
        if not masks_reshaped[:, 0, 0].bool().all():
            raise ValueError("composite loss requires a complete G x P rectangle")

        rows = target_gene_row.to(logits.device).long()
        if rows.ndim == 2:
            if not torch.equal(rows, rows[0].expand_as(rows)):
                raise ValueError("target_gene_row differs between response genes")
            rows = rows[0]
        if rows.shape != (perturbations,):
            raise ValueError(f"target_gene_row must have shape {(perturbations,)}")

        def flatten(value, name):
            if value is None:
                return None
            if value.shape != (genes, perturbations):
                raise ValueError(
                    f"{name} shape {tuple(value.shape)}; expected {(genes, perturbations)}"
                )
            return value.to(logits.device).detach().reshape(-1)

        batch = VCCBatch(
            G=genes,
            P=perturbations,
            labels=labels_reshaped[:, 0, 0].detach(),
            control_value=flatten(control_value, "control_value"),
            target_gene_row=rows.detach(),
            lfc_true=flatten(lfc_true, "lfc_true"),
            sig_mask=flatten(sig_mask, "sig_mask"),
        )
        report = self.vcc_loss(logits[:, 0, 0], batch)
        if return_dict is False:
            return report.total, logits
        return ExpressionModelOutput(
            loss=report.total,
            logits=logits,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=masks_reshaped,
            cls_loss=report.components.get("huber"),
            vcc_loss_components=report.components,
        )
