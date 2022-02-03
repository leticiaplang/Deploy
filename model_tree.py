
import pickle as pk  #para carregar os arquivos serializados
import pandas as pd  #fazer feature engineering manual

class DecisionTreeModel:

    def __init__(self):  #self referencia a classe
        self.model_pipe = pk.load(open("./pickles/decision_tree_pipe_spotify.pkl", 'rb'))
        self.scaler = {
            "z_score": pk.load(open("./pickles/encoder_z_score.pkl", "rb")),
            "min_max": pk.load(open("./pickles/encoder_min_max.pkl", "rb")),
            "one_hot": pk.load(open("./pickles/encoder_one_hot.pkl", "rb")),
            "ta_encorder": pk.load(open("./pickles/encoder_ta.pkl", "rb"))
        }
        self.model_manual = pk.load(open("./pickles/decision_tree.pkl", "rb"))

    def predict(self, df):
        return self.model_pipe.predict(df)