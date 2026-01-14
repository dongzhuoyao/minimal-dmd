import argparse
from typing import Any, Dict, List, Optional


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
      - flat keys matching argparse dests (e.g., lr, batch_size, wandb_project)
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

        # Remove the nested block to avoid "unknown key" warnings later.
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


def apply_yaml_config_to_parser(parser: argparse.ArgumentParser, config_path: Optional[str]) -> None:
    if not config_path:
        return

    cfg = _flatten_config(_load_yaml(config_path))

    allowed_dests = {a.dest for a in parser._actions}
    to_set: Dict[str, Any] = {}
    unknown: Dict[str, Any] = {}

    for k, v in cfg.items():
        if k in allowed_dests:
            to_set[k] = v
        else:
            unknown[k] = v

    if unknown:
        # Keep it simple: warn once, but don't crash (useful for sharing configs across scripts).
        unknown_keys = ", ".join(sorted(unknown.keys()))
        print(f"[config] Warning: ignoring unknown keys: {unknown_keys}")

    if to_set:
        parser.set_defaults(**to_set)


def parse_args_with_optional_yaml(
    parser: argparse.ArgumentParser, argv: Optional[List[str]] = None
) -> argparse.Namespace:
    # Pre-parse only --config so we can load defaults before validating required args.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    apply_yaml_config_to_parser(parser, pre_args.config)

    # Parse full args now (CLI flags override YAML defaults).
    return parser.parse_args(argv)

