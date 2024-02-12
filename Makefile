.PHONY: install build install-tensorflow install-torch

build:
	pip install build
	python -m build

install:
	pip install .

install-tensorflow:
	pip install ".[tensorflow]"

install-torch:
	pip install ".[torch]"
