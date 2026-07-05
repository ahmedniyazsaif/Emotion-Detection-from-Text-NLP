import pickle

import helper, numpy as np
from bayes_naive import BayesNaive, BayesNaiveClass

f_handler = open("model_4", 'rb')
bn_4: BayesNaive = pickle.load(f_handler)

f_handler = open("model_11", 'rb')
bn_11: BayesNaive = pickle.load(f_handler)

if __name__ == "__main__":
    while True:
        txt = input("Text: ")
        result = bn_4.pred_textual(txt)
        #top_classes: list[tuple[BayesNaiveClass, np.float64]] = result.get("top_classes")
        top_classes: list[tuple[str, np.float64]] = result.get("top_classes")
        words: list[dict] = result.get("words_impact")
        
        print("\nClass: Probability")
        
        for c in top_classes:
            print("\t", end="")
            print(c[0], end = ": ")
            print(round(c[1] * 100, 2))
            
        print("===")
        
        print("Word: Impact: Its class")
        for word in words:
            print("\t", end="")
            word_ = word.get("word")
            impact = word.get("impact")
            #its_class: BayesNaiveClass = word.get("class")
            class_name = word.get("class_name")
            
            print(f"{word_}: {impact}: {class_name}")