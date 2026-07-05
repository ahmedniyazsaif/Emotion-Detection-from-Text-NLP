import json, time

start = time.time()


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

labels = {
    0: "sadness",
    1: "happiness",
    2: "love",
    3: "anger",
    4: "worry",
    5: "surprise"
}

df = pd.read_csv('data/THE_dataset.csv')
# df = df[df['label'] != 4] # remove worry
X = df['text']
Y = df['label'].apply(
     lambda txt: labels[txt]
)

# df = pd.read_csv("data/original_train-00000-of-00001.csv")
# X = df['text']
# Y = df['label'].apply(
#      lambda txt: labels[txt]
# )

# df = pd.read_csv("data/emotion_sentimen_dataset.csv")
# df = df[df['Emotion'] != "boredom"]
# X = df["text"]
# Y = df["Emotion"]


import helper
    
X = X.apply(
   helper.clean_text
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report

X_train, X_test, Y_train, Y_test = list(X), list(X), list(Y), list(Y)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)

from bayes_naive import BayesNaive
bn = BayesNaive()


print('\ndividing')
bn.divide_data(X_train, Y_train)
print('done')

# ========== ==========
for c in bn.classes:
    #c.words_probs = remove_less_than(-10)
    print(c.class_name)
    print(c.top_used_words())
    print("\n")

print ("predicting")

preds = [
    bn.predict(i).get("classes_preds")
    for i in list(X_test)
]

preds_text = [
    bn.classes[np.argmax(i)].class_name 
    for i in preds
]
print('Done Prediction\n')

# ========= Displaying Confusion Matrix =========

print("Displaying Confusion Matrix")
acc = accuracy_score(list(Y_test), preds_text)*100
cm = confusion_matrix(Y_test, preds_text)
# Display the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= np.unique(Y) ).plot()
plt.show()

# ========= Printing Statistics =========
print(classification_report(Y_test, preds_text))
print("\nAccuracy: ", round(acc, 2), "%", sep="")
print("Time:", round(time.time() - start, 2))
print(f"Number of Words: {len(bn.words)}")
    

# ========= Showing Graphs =========

from graph import data_distribution, top_words_freqs

data_distribution(bn)
top_words_freqs(bn)

# ========= Saving Model =========

import pickle
f_handler = open("model", 'wb')
pickle.dump(bn, f_handler)
f_handler.close()