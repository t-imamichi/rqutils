VERSION ?= $(shell git describe --always --dirty)
ZIP_EXCLUDE := -xr!.DS_Store -xr!__pycache__
RUFF_TY_TARGET := rqutils test poc

all: check

format:
	ruff check --fix $(RUFF_TY_TARGET)
	ruff format $(RUFF_TY_TARGET)
	ty check --fix $(RUFF_TY_TARGET)
	rumdl fmt
	rumdl check --fix

check:
	codespell -I spell.txt *.md *.toml markdown $(RUFF_TY_TARGET)
	ruff check $(RUFF_TY_TARGET)
	ruff format --check $(RUFF_TY_TARGET)
	ty check $(RUFF_TY_TARGET)
	rumdl fmt --check
	rumdl check

build:
	uv build

zip: build
	7za a dist/rqutils-src-$(VERSION).zip pyproject.toml Makefile spell.txt rqutils test *.md $(ZIP_EXCLUDE)

test:
	pytest test -n auto

clean-cache:
	rm -rf $${XDG_CACHE_HOME:-$$HOME/.cache}/rqutils-jax

.PHONY: all format check build test zip clean-cache
