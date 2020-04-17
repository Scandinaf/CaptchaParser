import unittest

from PIL import Image

from ImageClassifier import ImageClassifier


class TestImageClassifier(unittest.TestCase):
    def test_get_classification(self):
        classifier = ImageClassifier("../data/saved_models/1587154471")

        image_3 = Image.open("./validation/xestlvphik_0.png")
        result = classifier.get_classification(image_3)

        self.assertTrue(result.__eq__(3))

        image_plus = Image.open("./validation/xestlvphik_1.png")
        result = classifier.get_classification(image_plus)

        self.assertTrue(result.__eq__(0))

        image_9 = Image.open("./validation/xestlvphik_2.png")
        result = classifier.get_classification(image_9)

        self.assertTrue(result.__eq__(9))


if __name__ == '__main__':
    unittest.main()
