# MLOps Homework (IE-7374)

Lab submissions for the MLOps course, based on the labs in
[raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps). Each lab lives
in its own folder with its own README, requirements and tests.

| Lab | Folder | Topic | CI workflow |
|---|---|---|---|
| Lab 1 | [`Lab1/`](Lab1/) | Virtualenv, pytest + unittest, GitHub Actions CI | `pytest_action.yml`, `unittest_action.yml` |
| Lab 2 | [`Lab2/`](Lab2/) | GitHub Actions for model training, calibration, evaluation and versioning | `lab2_model_pipeline.yml` |

All workflows are in [`.github/workflows/`](.github/workflows/) (GitHub only
reads workflows from the repo root). Each one has a `paths:` filter, so a
change in one lab's folder only triggers that lab's workflow.
