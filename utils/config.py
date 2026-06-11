from copy import deepcopy
from pathlib import Path


def _strip_quotes(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _split_inline_list(value):
    items = []
    current = []
    quote = None
    for char in value:
        if char in {"'", '"'}:
            quote = None if quote == char else char
        if char == "," and quote is None:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current or value.endswith(","):
        items.append("".join(current).strip())
    return items


def _parse_scalar(value):
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in _split_inline_list(inner)]
    value = _strip_quotes(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path):
    root = {}
    stack = [(-1, root)]
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line in {path}: {raw_line}")
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            node = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_yaml(path):
    path = Path(path)
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def merge_dicts(base, override):
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def set_by_path(config, dotted_key, value):
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = _parse_scalar(value)


def apply_cli_overrides(config, overrides):
    for override in overrides or []:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Override must use key=value syntax, got: {override}")
        set_by_path(config, key.strip(), value.strip())
    return config


def load_config(path, ablation=None, overrides=None):
    path = Path(path)
    config = load_yaml(path)

    base_name = config.pop("base_config", None)
    if base_name is not None:
        base_path = Path(base_name)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        config = merge_dicts(load_config(base_path), config)

    ablation_name = ablation or config.pop("active_ablation", None)
    ablation_overrides = config.pop("ablation_overrides", None)
    if ablation_name:
        if not ablation_overrides or ablation_name not in ablation_overrides:
            raise KeyError(f"Unknown ablation '{ablation_name}' in {path}")
        config = merge_dicts(config, ablation_overrides[ablation_name])

    return apply_cli_overrides(config, overrides)


def config_section(config, name):
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return value
