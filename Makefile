.PHONY: setup test sim solve sweep android-build android-test clean
SEED ?= 1

setup:
	uv sync

test:
	uv run pytest

# simulate a walk through the default house -> data/sim_seed$(SEED)
sim:
	uv run python -m sim.cli --seed $(SEED) --out data/sim_seed$(SEED)

# run full pipeline on the simulated session and evaluate against its truth
solve:
	uv run python -m solver.cli run --session data/sim_seed$(SEED) --out eval/out/sim_seed$(SEED)

# noise / sampling-rate sweep (slow, several minutes)
sweep:
	uv run python -m eval.sweep --out eval/out/sweep

android-build:
	cd android && ./gradlew assembleDebug

android-test:
	cd android && ./gradlew testDebugUnitTest

clean:
	rm -rf eval/out/* data/sim_*
