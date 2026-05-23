# Inter Prediction Optimization Method for RAHT in MPEG G-PCC

This repository provides the source code used in the paper
**Inter Prediction Optimization Method for RAHT in MPEG G-PCC**.

## Repository Structure

This repository provides the source code and datasets used in the paper RATH-based Improved Inter Prediction for Geometry-based Point Cloud Compression

| Path | Description |
| --- | --- |
| `mpeg-pcc-tmc13-release-v28.0-rc2/` | The original TMC13 v28.0 reference software used as the baseline. |
| `mpeg-pcc-tmc13-release-v28.0-R-EDA&R-AIA/` | The modified TMC13 v28.0 with inter prediction optimization (R-EDA & R-AIA) for RAHT. |

## Building

Each code project contains its own README file with build and usage
instructions:

- Baseline TMC13 v28.0:
  [`mpeg-pcc-tmc13-release-v28.0-rc2/README.md`](mpeg-pcc-tmc13-release-v28.0-rc2/README.md)
- TMC13 v28.0 with Inter Prediction Optimization:
  [`mpeg-pcc-tmc13-release-v28.0-R-EDA&R-AIA/README.md`](mpeg-pcc-tmc13-release-v28.0-R-EDA&R-AIA/README.md)

All projects are based on CMake. A typical build workflow is:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

Please refer to the README file in each project directory for platform-specific
commands, release build options, and usage examples.

## Notes

- `mpeg-pcc-tmc13-release-v28.0-rc2/` should be treated as the reference TMC13 v28.0 code.
- `mpeg-pcc-tmc13-release-v28.0-R-EDA&R-AIA/` is a modified version of the baseline code with inter-frame prediction and global motion optimization for RAHT attribute coding.
