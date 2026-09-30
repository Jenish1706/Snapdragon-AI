PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: venv test lint fmt pilot-prepare pilot-run pilot-report

venv:
	python3 -m venv .venv && .venv/bin/pip install -q -U pip && .venv/bin/pip install -q -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff format .

pilot-prepare:
	$(PY) scripts/pilot_mlp_profile.py prepare

pilot-run:
	$(PY) scripts/pilot_mlp_profile.py run

pilot-report:
	$(PY) scripts/pilot_mlp_profile.py report
