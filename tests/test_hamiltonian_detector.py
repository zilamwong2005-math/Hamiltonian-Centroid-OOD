import pytest
import torch
import torch.nn.functional as F

from hamiltonian_detector import (
    POTENTIAL_NAMES,
    HamiltonianDetector,
    image_effective_rank,
    normalize_anchor_masses,
    radial_affinity_and_force_coefficient,
)


@pytest.mark.parametrize("potential", POTENTIAL_NAMES)
def test_analytic_force_matches_autograd(potential):
    q = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.double, requires_grad=True)
    anchor = torch.tensor([[[[-0.1, 0.3, 0.5]]]], dtype=torch.double)
    sigma = torch.tensor([[[0.8]]], dtype=torch.double)
    difference = q[:, None, None, :] - anchor
    dist2 = difference.square().sum(-1)
    affinity, coefficient = radial_affinity_and_force_coefficient(
        dist2, sigma, potential
    )
    expected = torch.autograd.grad(affinity.sum(), q)[0]
    actual = (coefficient[..., None] * difference).sum(dim=(1, 2))
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("potential", POTENTIAL_NAMES)
def test_force_points_toward_single_anchor(potential):
    detector = HamiltonianDetector(
        feat_dim=2,
        n_classes=1,
        n_anchors_per_class=1,
        n_steps=1,
        potential=potential,
    )
    detector.anchors[0, 0] = torch.tensor([1.0, 0.0])
    detector.refresh_centroids()
    q = F.normalize(torch.tensor([[0.1, 1.0]]), dim=-1)
    _, force = detector._field(q, candidates=None)
    toward_anchor = detector.anchors[0, 0] - q[0]
    assert torch.dot(force[0], toward_anchor) > 0


def test_candidate_and_exact_scores_have_expected_shapes():
    torch.manual_seed(7)
    exact = HamiltonianDetector(8, 5, 3, n_steps=2, candidate_k=0, sim_batch=2)
    exact.anchors.copy_(F.normalize(torch.randn_like(exact.anchors), dim=-1))
    exact.refresh_centroids()
    queries = F.normalize(torch.randn(4, 8), dim=-1)
    assert exact.score(queries).shape == (4,)
    assert exact.predict(queries).shape == (4,)

    approximate = HamiltonianDetector(
        8, 5, 3, n_steps=2, candidate_k=2, sim_batch=2
    )
    approximate.load_state_dict(exact.state_dict())
    assert approximate.score(queries).shape == (4,)
    assert approximate.predict(queries).shape == (4,)


def test_candidate_field_matches_explicit_difference_implementation():
    torch.manual_seed(23)
    detector = HamiltonianDetector(
        7, 5, 3, potential="matern32", candidate_k=2
    )
    detector.anchors.copy_(F.normalize(torch.randn_like(detector.anchors), dim=-1))
    detector.masses.copy_(torch.rand_like(detector.masses) + 0.2)
    detector.refresh_centroids()
    queries = F.normalize(torch.randn(4, 7), dim=-1)
    candidates = detector.select_candidates(queries)
    affinity, force = detector._candidate_field(queries, candidates, True)

    anchors = detector.anchors[candidates]
    difference = queries[:, None, None, :] - anchors
    expected_affinity, coefficient = radial_affinity_and_force_coefficient(
        difference.square().sum(-1),
        detector.sigma[candidates],
        detector.potential,
    )
    masses = detector.masses[candidates]
    expected_affinity = (expected_affinity * masses).sum(-1)
    expected_force = ((coefficient * masses)[..., None] * difference).sum((1, 2))
    assert torch.allclose(affinity, expected_affinity, atol=1e-6)
    assert torch.allclose(force, expected_force, atol=1e-6)


def test_checkpoint_round_trip():
    detector = HamiltonianDetector(
        4,
        3,
        2,
        n_steps=3,
        potential="matern32",
        candidate_k=2,
        sigma_init=0.7,
    )
    detector.anchors.copy_(F.normalize(torch.randn_like(detector.anchors), dim=-1))
    detector.refresh_centroids()
    restored = HamiltonianDetector.from_checkpoint(detector.export_checkpoint())
    assert restored.potential == "matern32"
    assert restored.candidate_k == 2
    for key, value in detector.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_effective_rank_constant_and_rank_two_images():
    constant = torch.ones(1, 1, 4, 4)
    rank_two = torch.zeros(1, 1, 4, 4)
    rank_two[0, 0] = torch.diag(torch.tensor([1.0, -1.0, 0.0, 0.0]))
    assert torch.equal(image_effective_rank(constant), torch.ones(1))
    assert torch.allclose(
        image_effective_rank(rank_two), torch.tensor([2.0]), atol=1e-5
    )


def test_mass_normalization_preserves_explicit_conventions():
    masses = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
    by_class = normalize_anchor_masses(masses, "class_mean")
    globally = normalize_anchor_masses(masses, "global_mean")
    assert torch.allclose(by_class.mean(dim=1), torch.ones(2))
    assert torch.allclose(globally.mean(), torch.tensor(1.0))
    assert torch.equal(normalize_anchor_masses(masses, "none"), masses)


@pytest.mark.parametrize("candidate_k", [0, 2])
def test_trajectory_loss_backpropagates_to_bandwidths(candidate_k):
    torch.manual_seed(11)
    detector = HamiltonianDetector(
        5, 3, 2, n_steps=2, candidate_k=candidate_k, potential="cauchy"
    )
    detector.anchors.copy_(F.normalize(torch.randn_like(detector.anchors), dim=-1))
    detector.refresh_centroids()
    queries = F.normalize(torch.randn(4, 5), dim=-1)
    labels = torch.tensor([0, 1, 2, 0])
    loss, affinity = detector.training_loss(
        queries, labels, objective="trajectory", trajectory_steps=2
    )
    loss.backward()
    assert affinity.shape == (4, 3 if candidate_k == 0 else candidate_k)
    assert detector.log_sigma.grad is not None
    assert torch.isfinite(detector.log_sigma.grad).all()
    assert detector.log_sigma.grad.abs().sum() > 0


def test_zero_step_trajectory_loss_matches_static_loss():
    torch.manual_seed(13)
    detector = HamiltonianDetector(4, 3, 2, n_steps=3)
    detector.anchors.copy_(F.normalize(torch.randn_like(detector.anchors), dim=-1))
    detector.refresh_centroids()
    queries = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 1, 0])
    static_loss, static_affinity = detector.training_loss(queries, labels)
    trajectory_loss, trajectory_affinity = detector.training_loss(
        queries, labels, objective="trajectory", trajectory_steps=0
    )
    assert torch.allclose(static_affinity, trajectory_affinity)
    assert torch.allclose(static_loss, trajectory_loss)
