.PHONY: clean install graph train inference run test lint

# Python Interpreter
PYTHON = python
PIP = pip

# Installation
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

# Pipeline Steps
graph:
	$(PYTHON) scripts/build_graph.py

train:
	$(PYTHON) scripts/train.py

inference:
	$(PYTHON) scripts/inference.py

# Run the Dashboard
run:
	streamlit run dashboard/app.py

# Testing & Quality
test:
	pytest tests/

lint:
	flake8 src/ scripts/

# Cleaning up
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf .pytest_cache
	rm -rf models/artifacts/*.pkl