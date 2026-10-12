import math


def rectangle_area(length, width):
    """
    Computes the area of a rectangle.
    Args:
        length (int/float): Length of the rectangle.
        width (int/float): Width of the rectangle.
    Returns:
        int/float: Area of the rectangle.
    Raises:
        ValueError: If length or width is not a number, or is negative.
    """
    if not (isinstance(length, (int, float)) and isinstance(width, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    if length < 0 or width < 0:
        raise ValueError("Dimensions must not be negative.")

    return length * width


def triangle_area(base, height):
    """
    Computes the area of a triangle.
    Args:
        base (int/float): Base of the triangle.
        height (int/float): Height of the triangle.
    Returns:
        float: Area of the triangle.
    Raises:
        ValueError: If base or height is not a number, or is negative.
    """
    if not (isinstance(base, (int, float)) and isinstance(height, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    if base < 0 or height < 0:
        raise ValueError("Dimensions must not be negative.")

    return 0.5 * base * height


def circle_area(radius):
    """
    Computes the area of a circle.
    Args:
        radius (int/float): Radius of the circle.
    Returns:
        float: Area of the circle.
    Raises:
        ValueError: If radius is not a number, or is negative.
    """
    if not isinstance(radius, (int, float)):
        raise ValueError("Radius must be a number.")
    if radius < 0:
        raise ValueError("Radius must not be negative.")

    return math.pi * radius ** 2


def total_area(a1, a2, a3):
    """
    Adds three areas together.
    Args:
        a1 (int/float): First area.
        a2 (int/float): Second area.
        a3 (int/float): Third area.
    Returns:
        int/float: Sum of the three areas.
    """
    combined = a1 + a2 + a3
    return combined


# r_op = rectangle_area(2, 3)
# t_op = triangle_area(2, 3)
# c_op = circle_area(2)
# all_op = total_area(r_op, t_op, c_op)
