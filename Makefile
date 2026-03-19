.PHONY: install-dev test smoke demo clean

install-dev:
	pip install -e .[dev]

test:
	pytest -q

smoke:
	python examples/smoke_demo.py
	python examples/minibert_smoke_demo.py
	python examples/minislm_smoke_demo.py

demo: smoke

clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
