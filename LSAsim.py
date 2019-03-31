from gensim.models import Word2Vec
from language_model import get_training_testing
import numpy as np
import math
from itertools import zip_longest

class LSAsim:

    def similarity(self, vecA, vecB):
        return np.dot(vecA, vecB)/math.sqrt(np.dot(vecA, vecA)*np.dot(vecB, vecB))

    def similarity2(self, vecA, vecB):
        return sum([a*b for a,b in zip_longest(vecA, vecB)])/math.sqrt(sum([math.pow(a, 2) for a in vecA])*sum([math.pow(b, 2) for b in vecB]))


