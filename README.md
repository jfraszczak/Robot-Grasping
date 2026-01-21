# Robot-Grasping

```sh
uv run python -m src.3d_reconstruction.pycolmap.run_sparse_reconstruction  --images_path "data/green-rasp-1" --calibration_path "data/calibration/*.jpg" --aruco_markers --verbose
```

```sh
uv run python -m src.3d_reconstruction.stereo.compute_disparity
```
