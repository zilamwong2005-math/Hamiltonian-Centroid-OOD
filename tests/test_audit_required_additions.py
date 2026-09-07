from audit_required_additions import T10_CONFIG, _is_formal_t10_config


def _formal_config(seed=0):
    return {
        **T10_CONFIG,
        "seed": seed,
        "potentials": ["gaussian", "imq"],
    }


def test_t10_audit_accepts_only_the_fixed_formal_command(tmp_path):
    result = (
        tmp_path
        / "imagenet1k"
        / "imagenet1k_resnet50_tvsv1"
        / "seed0"
        / "mass-uniform-none_loss-static-ts10_pred-backbone_eval-full"
        / "openood_metrics_all_potentials.csv"
    )
    result.parent.mkdir(parents=True)
    assert _is_formal_t10_config(_formal_config(), result)

    wrong = _formal_config()
    wrong["candidate_k"] = 0
    assert not _is_formal_t10_config(wrong, result)


def test_t10_audit_rejects_the_right_config_at_the_wrong_output_tag(tmp_path):
    result = (
        tmp_path
        / "imagenet1k"
        / "imagenet1k_resnet50_tvsv1"
        / "seed0"
        / "mass-uniform-none_loss-static-ts3_pred-backbone_eval-full"
        / "openood_metrics_all_potentials.csv"
    )
    result.parent.mkdir(parents=True)
    assert not _is_formal_t10_config(_formal_config(), result)
