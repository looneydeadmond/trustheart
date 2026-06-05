PKG := trustheart
PY  := python

.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "install  install package (no deps)"
	@echo "dev      install dev tooling"
	@echo "lint     ruff + black --check + isort --check"
	@echo "type     mypy --strict"
	@echo "test     pytest"
	@echo "smoke    2-step training on the smoke config"
	@echo "docker   build the container image"
	@echo "clean    remove caches"

install:
	$(PY) -m pip install -e . --no-deps

dev:
	$(PY) -m pip install -e ".[dev,export]"

lint:
	ruff check .
	black --check src tests
	isort --check-only src tests

type:
	$(PY) -m mypy --strict src/$(PKG)

test:
	$(PY) -m pytest

smoke:
	$(PY) -m $(PKG).console.run train --config configs/experiment/_smoke.yaml

docker:
	docker build -t $(PKG):latest .

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache **/__pycache__ *.egg-info src/*.egg-info
