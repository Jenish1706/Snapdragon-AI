"""ADB device discovery, properties parsing, and command execution environment."""

from __future__ import annotations

import re


def parse_devices(text: str) -> list[tuple[str, str, dict[str, str]]]:
    """Parse adb devices -l output into (serial, state, attributes)."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines and "List of devices attached" in lines[0]:
        lines = lines[1:]
    results = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state = parts[0], parts[1]
        attrs = {}
        for token in parts[2:]:
            if ":" in token:
                k, v = token.split(":", 1)
                attrs[k] = v
        results.append((serial, state, attrs))
    return results


def parse_getprop(text: str) -> dict[str, str]:
    """Parse getprop output lines like [ro.soc.model]: [SM8850]."""
    props = {}
    pattern = re.compile(r"\[(.*?)\]:\s*\[(.*?)\]")
    for match in pattern.finditer(text):
        props[match.group(1)] = match.group(2)
    return props


class Device:
    def __init__(self, serial: str, state: str, props: dict[str, str] | None = None):
        self.serial = serial
        self.state = state
        self.props = props or {}

    @property
    def label(self) -> str:
        soc = self.props.get("ro.soc.model", "")
        if "SM8850" in soc:
            return "pad4"
        if "SM7550" in soc:
            return "nordce4"
        model = self.props.get("ro.product.model", "")
        if model:
            return re.sub(r"[^a-zA-Z0-9]", "", model).lower()
        return self.serial.lower()


def llama_cmd(bin_name: str, args: list[str], unit: str) -> str:
    """Construct command line for llama.cpp binary inside on-device Snapdragon environment."""
    env = ["LD_LIBRARY_PATH=./lib", "ADSP_LIBRARY_PATH=./lib"]
    if unit.lower() == "npu":
        env.append("GGML_HEXAGON_DEVICES=HTP0")
    env_str = " ".join(env)
    arg_str = " " + " ".join(args) if args else ""
    return f"{env_str} ./bin/{bin_name}{arg_str}"
