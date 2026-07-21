.PHONY: setup prepare pilot test validation

setup:
	./scripts/bootstrap_runpod.sh

prepare:
	./scripts/prepare_dataset.sh

pilot:
	./scripts/synthesize_pilot.sh

test:
	.venv/bin/python -m pytest -m "not gpu"

validation:
	./scripts/synthesize_validation.sh
