import unittest

import numpy as np

from week3_yolo import draw_detections


class DrawDetectionsTest(unittest.TestCase):
    def test_draws_box_without_changing_input(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        output = draw_detections(image, [(10, 10, 60, 60, "object", 0.9)])
        self.assertEqual(int(image.sum()), 0)
        self.assertGreater(int(output.sum()), 0)


if __name__ == "__main__":
    unittest.main()
