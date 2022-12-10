import unittest
from adaboost import AdaBoost

class TestAdaBoost(unittest.TestCase):
    def test_fit(self):
        # Create a dummy dataset
        X = [[1,2],[2,3],[3,4]]
        y = [1,2,3]

        # Create an instance of AdaBoost
        model = AdaBoost()
        # Fit the model to the data
        model.fit(X, y)
        # Check that the model has been successfully fitted
        self.assertTrue(model.fitted)

if __name__ == '__main__':
    unittest.main()
