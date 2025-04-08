from rapidfuzz.distance import Levenshtein

#TODO: use edit distance offers more control
class Levenshtein_model():

    def __init__(self) -> None:
        pass

    def fit(self, description):
        pass

    def predict(self, x,y):
        return Levenshtein.distance(x, y)
    
    def calculate_bias(self, list_words):
        pass