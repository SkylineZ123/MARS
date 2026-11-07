# Simulation environment

Use Linux, Conda Python 3.12, PyTorch 2.7.1, and torchvision 0.22.1 with the
CUDA 12.6 wheels. Installation commands are in the [README](../README.md).
The optional `smolvla`, `siglip2`, `tubes`, and `test` dependencies are declared
in `pyproject.toml`; `requirements-release.txt` installs them from a source checkout.

The published weights are at [XDEDM7/MARS](https://huggingface.co/XDEDM7/MARS).
They contain model files and configuration, not an environment archive.
The backbone integration lives in `mars/backbones`; an adjacent `routerlt`
checkout or a private Python search path is not required.

## Camera rendering

Set these variables before importing robosuite:

```bash
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=egl
```

EGL requires system OpenGL libraries and a compatible graphics driver.
For software rendering, select `MUJOCO_GL=osmesa` and install system OSMesa.
Use `TubesEnv(render=False)` for physics-only checks.

## OpenCV version

LeRobot requires `opencv-python-headless`, while robosuite requires
`opencv-python`. Both distributions install `cv2`; package metadata alone
cannot establish which binary is imported. To match the OpenCV 5.0.0 binary
used during migration, finalize installation with:

```bash
python -m pip install --no-deps --force-reinstall opencv-python==5.0.0.93
python -c "import cv2; assert cv2.__version__ == '5.0.0', cv2.__version__"
python -m pip check
```

A complete fresh-environment installation has not been validated. This guide
specifies the deployment dependencies, not a claim of reproducing paper scores.
