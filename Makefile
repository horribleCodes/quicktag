install:
	pre-commit install
	pip install -e ".[dev]"

install-onnx:
	pre-commit install
	pip install -e ".[dev]"
	pip install -e ".[onnx]"

test:
	pytest -m "not integration"

test-full:
	pytest -m ""

build-linux:
	bash ./build.sh

build-win:
	powershell -ExecutionPolicy Bypass -File ./build.ps1

format:
	ruff format .
	ruff check --fix .
