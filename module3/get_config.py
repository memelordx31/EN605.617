#!/usr/bin/env python3
"""
get_config.py - tiny dependency-free reader for the flat, nested-dict-only
YAML subset used by assignment_config.yaml (no lists, no multi-line strings, no
anchors). Avoids requiring PyYAML to be installed just to read a handful
of scalar values.

Usage:
    python3 get_config.py assignment_config.yaml project.src_dir
    python3 get_config.py assignment_config.yaml run.blockSize
"""

import sys


def load(path):
    """Parse indentation-nested 'key: value' YAML into a nested dict."""
    root = {}
    stack = [(-1, root)]  # (indent_level, dict_at_that_level)

    with open(path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            # strip inline comments on the value (simple '#'-based, fine for this config)
            if "#" in value:
                value = value.split("#", 1)[0].strip()

            # pop stack back to the parent whose indent is less than this line's
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]

            if value == "":
                # New nested section
                new_dict = {}
                parent[key] = new_dict
                stack.append((indent, new_dict))
            else:
                parent[key] = _coerce(value)

    return root


def _coerce(value):
    value = value.strip().strip('"').strip("'")
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def get(config, dotted_key):
    node = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    return node


def main():
    if len(sys.argv) != 3:
        print("Usage: get_config.py <config.yaml> <dotted.key>", file=sys.stderr)
        sys.exit(1)

    config_path, dotted_key = sys.argv[1], sys.argv[2]
    config = load(config_path)
    try:
        value = get(config, dotted_key)
    except KeyError:
        print(f"Error: key '{dotted_key}' not found in {config_path}", file=sys.stderr)
        sys.exit(1)

    print(value)


if __name__ == "__main__":
    main()
