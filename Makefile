.PHONY: install build upload-pypi install-tensorflow install-torch

build:
	pip install build
	python -m build

upload-pypi:
	twine upload dist/*

install:
	pip install .

install-tensorflow:
	pip install ".[tensorflow]"

install-torch:
	pip install ".[torch]"
