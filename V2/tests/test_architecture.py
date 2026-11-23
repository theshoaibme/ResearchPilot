"""Unit and gradient tests for Pan-Organ Net V2 unified architecture and trainer."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from panorgan.config import RuntimeProfile, load_config
from panorgan.models import BaselineModelFactory, PanOrganNetV2
from panorgan.engine import ColabResilientTrainer


def _make_dummy_batch(batch_size: int = 4, h: int = 64, w: int = 64):
    prompts = [
        "segment abdominal organs in 3D computed tomography",
        "segment glioma tumor in brain FLAIR MRI",
        "classify thoracic pathologies in frontal chest radiograph",
        "segment and classify breast lesion in ultrasound",
    ]
    return {
        "image": torch.rand(batch_size, 5, h, w),
        "mask": torch.randint(0, 4, (batch_size, h, w)),
        "has_mask": torch.tensor([True, True, False, True][:batch_size]),
        "saliency": torch.rand(batch_size, h, w),
        "meta_vec": torch.randn(batch_size, 10),
        "modality_id": torch.tensor([0, 1, 2, 3][:batch_size]),
        "is_3d": torch.tensor([True, True, False, False][:batch_size]),
        "label": torch.tensor([1, 0, -1, 2][:batch_size]),
        "multilabel": torch.tensor(
            [
                [-1.0] * 14,
                [-1.0] * 14,
                [1.0, 0.0, 1.0] + [0.0] * 11,
                [-1.0] * 14,
            ][:batch_size]
        ),
        "body_coord": torch.rand(batch_size, 3),
        "prompt": prompts[:batch_size],
    }


def test_panorgan_v2_forward_and_backward_mixed_2d_25d():
    model = PanOrganNetV2(
        backbone_name="lightweight_test",
        pretrained=False,
        embed_dim=128,
        use_lora=True,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        use_sacr=True,
        masking_mode="ma_ugm",
    )
    batch = _make_dummy_batch(batch_size=4, h=64, w=64)

    out = model(batch["image"], batch["meta_vec"], batch["is_3d"], prompts=batch["prompt"])
    assert out["seg_logits"].shape == (4, 16, 64, 64)
    assert out["multilabel_logits"].shape == (4, 14)
    assert out["multiclass_logits"].shape == (4, 4)
    assert out["pred_coords"].shape == (4, 3)
    assert out["gate_probs"].shape == (4, 4)
    assert torch.allclose(out["gate_probs"].sum(dim=-1), torch.ones(4), atol=1e-4)

    loss = (
        out["seg_logits"].mean()
        + out["multilabel_logits"].mean()
        + out["multiclass_logits"].mean()
        + out["moe_aux_loss"]
        + out["pred_coords"].mean()
    )
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


def test_ssl_masking_ablation_modes():
    batch = _make_dummy_batch(batch_size=2, h=64, w=64)
    for mode in ("random", "gradient_sgm", "ma_ugm"):
        model = PanOrganNetV2(
            backbone_name="lightweight_test",
            pretrained=False,
            embed_dim=64,
            masking_mode=mode,
        )
        out = model.forward_ssl(
            batch["image"],
            batch["meta_vec"],
            batch["is_3d"],
            saliency=batch["saliency"],
            teacher_uncertainty=torch.rand(2, 64, 64),
            ema_blend=0.5,
        )
        assert out["recon"].shape == (2, 3, 64, 64)
        assert torch.isfinite(out["recon_loss"]).item()
        assert 0.2 <= float(out["pixel_mask"].mean().item()) <= 0.9


def test_trainer_ssl_multitask_and_atomic_resume(tmp_path: Path):
    model = PanOrganNetV2(
        backbone_name="lightweight_test",
        pretrained=False,
        embed_dim=64,
        use_moe=True,
        use_sacr=True,
    )
    runtime = RuntimeProfile(
        gpu_name="CPU", tier="CPU", batch_size=2, grad_accum=1, amp_dtype="float32", device="cpu"
    )
    batches = [_make_dummy_batch(batch_size=2, h=64, w=64) for _ in range(2)]
    loader = DataLoader(batches, batch_size=None)

    ckpt_dir = tmp_path / "ckpts"
    trainer = ColabResilientTrainer(model, runtime, ckpt_dir, lr=1e-3, checkpoint_every_steps=2)
    hist1 = trainer.fit(loader, loader, num_epochs=1, mode="multitask", resume=False)
    assert len(hist1["train_loss"]) == 1
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "best.pt").exists()

    # Resume from saved checkpoint into epoch 2
    trainer2 = ColabResilientTrainer(model, runtime, ckpt_dir, lr=1e-3)
    hist2 = trainer2.fit(loader, loader, num_epochs=2, mode="multitask", resume=True)
    assert len(hist2["train_loss"]) == 2


def test_baseline_factory_and_config_load():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "v2_panorgan.yaml")
    assert cfg["project"]["version"] == "2.0.0"
    seg = BaselineModelFactory.build_segmenter(in_channels=5, num_classes=4)
    y = seg(torch.rand(2, 5, 64, 64))
    assert y.shape == (2, 4, 64, 64)
