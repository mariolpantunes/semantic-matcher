
from rapidfuzz.distance import Levenshtein


class Levenshtein_model():

    def __init__(self, output) -> None:
        pass

    def fit(self, description):
        pass

    def predict(self, x,y):
        return Levenshtein.distance(x, y)
    
    def calculate_bias(self, list_words):
        pass