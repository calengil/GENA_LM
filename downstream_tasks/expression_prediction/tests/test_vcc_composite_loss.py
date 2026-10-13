import pytest
import torch

from downstream_tasks.expression_prediction.vcc_composite_loss import (
    VCCBatch,
    VCCCompositeLoss,
)


def make_batch(G=8, P=3):
    torch.manual_seed(7)
    control = torch.full((G, P), 2.0)
    lfc = torch.randn(G, P) * 0.7
    labels = torch.log1p(torch.expm1(control) * torch.pow(2.0, lfc))
    sig = (lfc.abs() > 0.4).float()
    return VCCBatch(
        G=G,
        P=P,
        labels=labels.reshape(-1),
        control_value=control.reshape(-1),
        target_gene_row=torch.tensor([0, 1, -1]),
        lfc_true=lfc.reshape(-1),
        sig_mask=sig.reshape(-1),
    )


def make_loss(arm):
    if arm == "A":
        return VCCCompositeLoss(use_huber=True, use_pds=True, normalize="none")
    return VCCCompositeLoss(
        use_huber=True, use_pds=True, use_nmae=True, use_fid=True,
        use_reach=True, use_jaccard=True, normalize="none",
        fid_variant="soft_yield_v2", reach_variant="topk_prefix_v2",
        w_pred_variant="A", weight_reach=0.0333,
    )


@pytest.mark.parametrize("arm", ["A", "B"])
def test_forward_backward_is_finite(arm):
    batch = make_batch()
    logits = (batch.labels + 0.05 * torch.randn_like(batch.labels)).requires_grad_(True)
    report = make_loss(arm)(logits, batch)
    assert torch.isfinite(report.total)
    report.total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_perfect_prediction_has_zero_huber_and_jaccard():
    batch = make_batch()
    report = make_loss("B")(batch.labels.clone().requires_grad_(True), batch)
    assert report.components["huber"] < 1e-8
    assert report.components["jaccard"].abs() < 1e-6


def test_arm_b_requires_offline_de():
    batch = make_batch()
    batch.lfc_true = None
    batch.sig_mask = None
    with pytest.raises(ValueError, match="require batch.lfc_true"):
        make_loss("B")(batch.labels, batch)


def test_incomplete_shape_is_rejected():
    batch = make_batch()
    batch.control_value = batch.control_value[:-1]
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        make_loss("A")(batch.labels, batch)


def test_arm_b_stays_finite_for_negative_identity_head_predictions():
    batch = make_batch()
    logits = torch.full_like(batch.labels, -0.5, requires_grad=True)
    report = make_loss("B")(logits, batch)
    assert torch.isfinite(report.total)
    report.total.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("value", [-100.0, -1.0, 0.0, 20.0, 100.0])
def test_arm_b_extreme_predictions_have_finite_gradients(value):
    batch = make_batch()
    logits = torch.full_like(batch.labels, value, requires_grad=True)
    report = make_loss("B")(logits, batch)
    assert torch.isfinite(report.total)
    report.total.backward()
    assert torch.isfinite(logits.grad).all()


def test_arm_b_rejects_negative_control():
    batch = make_batch()
    batch.control_value[0] = -0.1
    with pytest.raises(ValueError, match="control must be non-negative"):
        make_loss("B")(batch.labels, batch)


def test_arm_b_nmae_handles_significant_zero_lfc():
    batch = make_batch()
    batch.lfc_true[0] = 0.0
    batch.sig_mask[0] = 1.0
    logits = batch.labels.clone().requires_grad_(True)
    report = make_loss("B")(logits, batch)
    assert torch.isfinite(report.components["nmae"])
    report.total.backward()
    assert torch.isfinite(logits.grad).all()
