# Robot-Grasping

## Insallation
sh```
git clone <git_url>
```

```sh
pip install uv
```

## Run PyCOLMAP Reconstruction

```sh
uv run python -m src.3d_reconstruction.pycolmap.run_sparse_reconstruction  --images_path "data/green-rasp-1" --aruco_markers --verbose
```

```sh
uv run python -m src.3d_reconstruction.pycolmap.run_dense_reconstruction  --images_path "data/green-rasp-1" --calibration_path "data/calibration/*.jpg" --aruco_markers --verbose
```

## Compute Disparity
```sh
uv run python -m src.3d_reconstruction.stereo.compute_disparity
```

## Visualize Reconstructions
```sh
uv run python -m reconstructions.visualize
```
