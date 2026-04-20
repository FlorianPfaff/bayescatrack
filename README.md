# Track2p / Suite2p bridge for PyRecEst

This utility is aimed at the
Track2p benchmark and related multi-session calcium-imaging datasets that store either:

- `suite2p/planeX` folders, or
- `data_npy/planeX` folders in the Track2p raw-NumPy convention.

## What it gives you

- Loads Track2p-style subject/session directories.
- Reconstructs Suite2p ROI masks from `stat.npy`.
- Computes ROI centroids and spatial covariance matrices.
- Builds constant-velocity state moments that can initialize PyRecEst filters.
- Lazily creates PyRecEst `GaussianDistribution` and `KalmanFilter` objects when PyRecEst is installed.
- Exports per-session measurements and state moments to a single `.npz` archive.
- Includes a CLI for quick inspection.

## Files

- `track2p_pyrecest_bridge.py` — standalone script and importable module.
- `test_track2p_pyrecest_bridge.py` — synthetic tests.

## CLI examples

Inspect one subject:

```bash
python -m track2p_pyrecest_bridge summary /path/to/jm039 --plane plane0
```

Export PyRecEst-ready states:

```bash
python -m track2p_pyrecest_bridge export /path/to/jm039 /tmp/jm039_plane0.npz \
  --plane plane0 \
  --input-format auto \
  --weighted-masks \
  --weighted \
  --velocity-variance 25.0
```

Validate that PyRecEst objects can be instantiated during export:

```bash
python -m track2p_pyrecest_bridge export /path/to/jm039 /tmp/jm039_plane0.npz \
  --validate-pyrecest
```

## Python example

```python
from track2p_pyrecest_bridge import load_track2p_subject

sessions = load_track2p_subject("/path/to/jm039", plane_name="plane0", input_format="auto")
first_session = sessions[0].plane_data

measurements = first_session.to_measurement_matrix(order="xy")
means, covariances = first_session.to_constant_velocity_state_moments(
    order="xy",
    velocity_variance=25.0,
)

# Requires PyRecEst to be importable.
filters = first_session.to_pyrecest_kalman_filters(
    order="xy",
    velocity_variance=25.0,
)
```

## Notes

- The state layout is `[pos_1, vel_1, pos_2, vel_2]`.
- The script intentionally stops at representation/export and does not add Track2p-specific
  benchmark logic into the PyRecEst core package.
- `--validate-pyrecest` is useful when you want the export step to fail early if the current
  environment cannot instantiate the expected PyRecEst classes.
