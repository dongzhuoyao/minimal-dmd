from typing import Any, Dict
from types import SimpleNamespace


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. Install it with `pip install pyyaml`."
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping/object at top-level: {path}")
    return data


def _flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
      - flat keys matching config attributes (e.g., lr, batch_size, wandb_project)
      - a nested `wandb:` mapping with nicer names.
    """
    out: Dict[str, Any] = dict(cfg)

    wandb_cfg = cfg.get("wandb")
    if isinstance(wandb_cfg, dict):
        mapping = {
            "enabled": "wandb",
            "project": "wandb_project",
            "entity": "wandb_entity",
            "run_name": "wandb_run_name",
            "mode": "wandb_mode",
            "tags": "wandb_tags",
            "dir": "wandb_dir",
            "log_every": "wandb_log_every",
            "watch": "wandb_watch",
            "log_images": "wandb_log_images",
            "num_log_images": "wandb_num_log_images",
            "log_checkpoints": "wandb_log_checkpoints",
            "log_samples": "wandb_log_samples",
            "sample_every": "wandb_sample_every",
            "sample_num_images": "wandb_sample_num_images",
            "sample_steps": "wandb_sample_steps",
            "sample_conditioning_sigma": "wandb_sample_conditioning_sigma",
        }
        saw_any_wandb_setting = False
        for k, v in wandb_cfg.items():
            dest = mapping.get(k)
            if dest is not None:
                out[dest] = v
                if k != "enabled":
                    saw_any_wandb_setting = True

        # YAML-only W&B: if you provide a `wandb:` block but omit `enabled`,
        # assume you intended to enable W&B.
        if "enabled" not in wandb_cfg and saw_any_wandb_setting:
            out["wandb"] = True

        # Remove the nested block to avoid confusion.
        out.pop("wandb", None)

    # YAML-only W&B (flat style): if any wandb_* key is present but `wandb` itself
    # isn't, assume intent is to enable W&B.
    if "wandb" not in out:
        for k in (
            "wandb_project",
            "wandb_entity",
            "wandb_run_name",
            "wandb_mode",
            "wandb_tags",
            "wandb_log_every",
            "wandb_watch",
            "wandb_log_images",
            "wandb_num_log_images",
            "wandb_log_checkpoints",
        ):
            if k in out:
                out["wandb"] = True
                break

    return out


def load_config(config_path: str) -> SimpleNamespace:
    """
    Load YAML config file and return a config object with defaults.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        SimpleNamespace object with all config values
    """
    # Default values
    defaults = {
        "data_dir": "./data",
        "output_dir": "./log/checkpoints/dmd2",
        "teacher_checkpoint": None,
        "batch_size": 128,
        "generator_lr": 2e-6,
        "guidance_lr": 2e-6,
        "step_number": 100_000,
        "num_train_timesteps": 1000,
        "sigma_min": 0.002,
        "sigma_max": 80.0,
        "sigma_data": 0.5,
        "rho": 7.0,
        "min_step_percent": 0.02,
        "max_step_percent": 0.98,
        "conditioning_sigma": 80.0,
        "dfake_gen_update_ratio": 10,
        "dm_loss_weight": 1.0,
        "max_grad_norm": 1.0,
        "save_every": 5000,
        "wandb": False,
        "wandb_project": "minimal-dmd",
        "wandb_entity": None,
        "wandb_run_name": None,
        "wandb_mode": "online",
        "wandb_tags": None,
        "wandb_dir": "./log/wandb",
        "wandb_log_every": 50,
        "wandb_watch": False,
        "wandb_log_images": False,
        "wandb_num_log_images": 32,
        "wandb_log_checkpoints": False,
        "wandb_log_samples": False,
        "wandb_sample_every": 1000,
        "wandb_sample_num_images": 64,
        "resume_from_checkpoint": None,
    }
    
    # Load YAML config
    yaml_cfg = _flatten_config(_load_yaml(config_path))
    
    # Merge defaults with YAML config (YAML overrides defaults)
    config_dict = {**defaults, **yaml_cfg}
    
    # Convert to SimpleNamespace for attribute access
    return SimpleNamespace(**config_dict)
