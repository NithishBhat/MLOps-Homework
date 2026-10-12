# LAB1 — MLOps (IE-7374)

Lab 1 submission: virtual environment, repository/folder structure, a geometry
module, pytest + unittest test suites, and two GitHub Actions workflows.

## Structure

```
MLOps-Homework/
├── .github/workflows/
│   ├── pytest_action.yml      # CI: runs pytest in Lab1/, uploads JUnit XML report
│   └── unittest_action.yml    # CI: runs python -m unittest test.test_unittest in Lab1/
└── Lab1/
    ├── data/                  # project data files
    ├── src/
    │   └── shapes.py          # area calculations
    ├── test/
    │   ├── test_pytest.py     # pytest tests (incl. a parametrized example)
    │   └── test_unittest.py   # unittest.TestCase tests
    ├── requirements.txt
    └── README.md
```

GitHub only reads workflows from the repository root, so they live in
`/.github/workflows/` and use `working-directory: Lab1` plus a `paths:` filter
so they only run when Lab 1 files change.

## Functions in `src/shapes.py`

| Function | Behaviour |
|---|---|
| `rectangle_area(length, width)` | area of a rectangle, `length * width` |
| `triangle_area(base, height)` | area of a triangle, `0.5 * base * height` |
| `circle_area(radius)` | area of a circle, `pi * radius ** 2` |
| `total_area(a1, a2, a3)` | sum of the three areas above |

The three area functions raise `ValueError` if an input is not a number or is
negative.

## Setup

```
git clone https://github.com/NithishBhat/MLOps-Homework.git
cd MLOps-Homework/Lab1
python -m venv lab_01
lab_01\Scripts\activate          # Windows
source lab_01/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Running the tests

```
pytest                                  # pytest suite
python -m unittest test.test_unittest   # unittest suite
```

Both suites are also run automatically by GitHub Actions on every push and pull
request to `main`.
