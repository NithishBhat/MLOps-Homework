import math
import os
import sys
import unittest

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import shapes


class TestShapes(unittest.TestCase):

    def test_rectangle_area(self):
        self.assertEqual(shapes.rectangle_area(2, 3), 6)
        self.assertEqual(shapes.rectangle_area(5, 0), 0)
        self.assertEqual(shapes.rectangle_area(1, 1), 1)
        self.assertEqual(shapes.rectangle_area(2.5, 4), 10)

    def test_triangle_area(self):
        self.assertEqual(shapes.triangle_area(2, 3), 3)
        self.assertEqual(shapes.triangle_area(5, 0), 0)
        self.assertEqual(shapes.triangle_area(1, 1), 0.5)
        self.assertEqual(shapes.triangle_area(3, 4), 6)

    def test_circle_area(self):
        self.assertEqual(shapes.circle_area(0), 0)
        self.assertAlmostEqual(shapes.circle_area(1), math.pi)
        self.assertAlmostEqual(shapes.circle_area(2), 4 * math.pi)
        self.assertAlmostEqual(shapes.circle_area(0.5), 0.25 * math.pi)

    def test_total_area(self):
        self.assertEqual(shapes.total_area(2, 3, 5), 10)
        self.assertEqual(shapes.total_area(6, 3, 0), 9)
        self.assertEqual(shapes.total_area(0, 0, 0), 0)
        self.assertEqual(shapes.total_area(1.5, 2.5, 96), 100)

    def test_raises_on_non_numbers(self):
        with self.assertRaises(ValueError):
            shapes.rectangle_area("2", 3)
        with self.assertRaises(ValueError):
            shapes.circle_area("2")

    def test_raises_on_negative_dimensions(self):
        with self.assertRaises(ValueError):
            shapes.triangle_area(-1, 3)
        with self.assertRaises(ValueError):
            shapes.circle_area(-1)


if __name__ == '__main__':
    unittest.main()
