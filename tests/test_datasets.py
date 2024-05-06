import unittest


# Function to be tested
def add(a, b):
    return a + b


# Test case class
class TestAddition(unittest.TestCase):

    # Test case to check addition of two positive numbers
    def test_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)

    # Test case to check addition of two negative numbers
    def test_negative_numbers(self):
        self.assertEqual(add(-2, -3), -5)

    # Test case to check addition of positive and negative numbers
    def test_mixed_numbers(self):
        self.assertEqual(add(2, -3), -1)
        self.assertEqual(add(-2, 3), 1)


# Function to be tested
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b


# Test case class
class TestDivision(unittest.TestCase):

    # Test case to check division by a non-zero number
    def test_divide_non_zero(self):
        self.assertEqual(divide(6, 2), 3)

    # Test case to check division by zero
    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            divide(6, 0)


if __name__ == '__main__':
    unittest.main()
