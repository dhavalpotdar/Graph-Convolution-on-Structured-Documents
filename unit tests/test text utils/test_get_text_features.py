import unittest
from text_utils import get_text_features
import numpy as np

class TestGetTextFeatures(unittest.TestCase):
    def test_get_text_features_empty(self):
        STRING = ''
        gtf = get_text_features(STRING)

        ANSWER = np.zeros(shape=22)
        self.assertListEqual(gtf.tolist(), ANSWER.tolist())

    def test_string1(self):
        STRING = 'this is SPARTA'
        gtf = get_text_features(STRING)

        ANSWER = np.array([6,6,2,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        assert len(ANSWER) == 22, 'Check length of answer'

        self.assertListEqual(gtf.tolist(), ANSWER.tolist())

    def test_string2(self):
        STRING = '##this is comment 12 of 30'
        gtf = get_text_features(STRING)

        ANSWER = np.array([15,0,5,15,4,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0])
        assert len(ANSWER) == 22, 'Check length of answer'

        self.assertSequenceEqual(gtf.tolist(), ANSWER.tolist())

    def test_string3(self):
        STRING = '+/ /-'
        gtf = get_text_features(STRING)

        ANSWER = np.array([0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,2,0,0])
        assert len(ANSWER) == 22, 'Check length of answer'

        self.assertSequenceEqual(gtf.tolist(), ANSWER.tolist())

    def test_string4(self):
        STRING = '!!!::'
        gtf = get_text_features(STRING)

        ANSWER = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2])
        assert len(ANSWER) == 22,'Check length of answer'

        self.assertSequenceEqual(gtf.tolist(), ANSWER.tolist())