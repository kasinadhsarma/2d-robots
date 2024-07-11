import unittest
from behaviors.squats import SquatEnv

class TestSquatEnv(unittest.TestCase):
    def setUp(self):
        self.env = SquatEnv()

    def test_initial_state(self):
        state = self.env.reset()
        self.assertEqual(len(state), 3)
        self.assertTrue(all(isinstance(x, float) for x in state))

    def test_step_function(self):
        self.env.reset()
        action = [0.5]
        state, reward, done, _ = self.env.step(action)
        self.assertEqual(len(state), 3)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

if __name__ == '__main__':
    unittest.main()
