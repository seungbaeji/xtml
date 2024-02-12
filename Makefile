.PHONY: install build clean pypi install-tensorflow install-torch

build:
	pip install build
	python -m build

clean:
	rm -rf ./dist ./build ./xtml.egg-info

pypi:
	twine upload dist/*

install:
	pip install .

install-tensorflow:
	pip install ".[tensorflow]"

install-torch:
	pip install ".[torch]"
