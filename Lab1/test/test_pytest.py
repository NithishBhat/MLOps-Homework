import math
import os
import sys

import pytest

# Make the project root importable so `from src import shapes` works
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import shapes


def test_rectangle_area():
    assert shapes.rectangle_area(2, 3) == 6
    assert shapes.rectangle_area(5, 0) == 0
    assert shapes.rectangle_area(1, 1) == 1
    assert shapes.rectangle_area(2.5, 4) == 10


def test_triangle_area():
    assert shapes.triangle_area(2, 3) == 3
    assert shapes.triangle_area(5, 0) == 0
    assert shapes.triangle_area(1, 1) == 0.5
    assert shapes.triangle_area(3, 4) == 6


def test_circle_area():
    assert shapes.circle_area(0) == 0
    assert shapes.circle_area(1) == pytest.approx(math.pi)
    assert shapes.circle_area(2) == pytest.approx(4 * math.pi)
    assert shapes.circle_area(0.5) == pytest.approx(0.25 * math.pi)


def test_total_area():
    assert shapes.total_area(2, 3, 5) == 10
    assert shapes.total_area(6, 3, 0) == 9
    assert shapes.total_area(0, 0, 0) == 0
    assert shapes.total_area(1.5, 2.5, 96) == 100


# Parametrized example (see Step 4 of the lab README): the same test function
# runs once per (length, width, expected) tuple.
@pytest.mark.parametrize(
    "length, width, expected",
    [
        (2, 3, 6),
        (5, 0, 0),
        (1, 1, 1),
        (2.5, 4, 10),
    ],
)
def test_rectangle_area_parametrized(length, width, expected):
    assert shapes.rectangle_area(length, width) == expected


def test_raises_on_non_numbers():
    with pytest.raises(ValueError):
        shapes.rectangle_area("2", 3)
    with pytest.raises(ValueError):
        shapes.circle_area("2")


def test_raises_on_negative_dimensions():
    with pytest.raises(ValueError):
        shapes.triangle_area(-1, 3)
    with pytest.raises(ValueError):
        shapes.circle_area(-1)
