.PHONY: setup test sim solve sweep calibration fakereal android-build android-test clean
SEED ?= 1

setup:
	uv sync

test:
	uv run pytest

# simulate a walk through the default house -> data/sim_seed$(SEED)  (+ eval/out/sim_seed$(SEED)_truth.png)
sim:
	uv run python -m sim.cli --seed $(SEED) --out data/sim_seed$(SEED)

# full pipeline on the simulated session, evaluated against its truth -> eval/out/sim_seed$(SEED)/
solve: sim
	uv run python -m solver.cli run --session data/sim_seed$(SEED) --out eval/out/sim_seed$(SEED)

# simulator run -> "fake real" session (+ tape-style ground_truth.yaml) -> solver -> overlay (M4 acceptance, by hand)
fakereal: sim
	uv run python -m sim.fake_real --sim data/sim_seed$(SEED) --out data/fake_real_seed$(SEED)
	uv run python -m solver.cli run --session data/fake_real_seed$(SEED) --truth data/fake_real_seed$(SEED)/ground_truth.yaml --out eval/out/fake_real_seed$(SEED)
	uv run python -m eval.overlay --result eval/out/fake_real_seed$(SEED)/result.json --truth data/fake_real_seed$(SEED)/ground_truth.yaml

# noise / sampling-rate sweeps (slow: ~10 min on 4 cores)
sweep:
	uv run python -m eval.sweep --out eval/out/sweep

# is per-wall confidence meaningful? (8 simulated houses, full ensemble)
calibration:
	uv run python -m eval.calibration --out eval/out/calibration

android-build:
	cd android && ./gradlew assembleDebug

android-test:
	cd android && ./gradlew testDebugUnitTest

clean:
	rm -rf eval/out/* data/sim_* data/fake_real_*
