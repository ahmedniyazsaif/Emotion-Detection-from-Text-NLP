import numpy as np
import helper

class BayesNaive:
    def __init__(self):
        pass

    def top_used_words(self, top = 10):
        sorted_indices = np.argsort(list(self.words_count.values()))[::-1]
        top_indices = sorted_indices[:top]

        words = list(self.words_count.keys())
        top_words = [words[i] for i in top_indices]

        return top_words

    def divide_data(self, x: list[str], y: list[str]):    
        divided_data: dict[str, list[str]] = {}
        self.words  = {}
        self.words_count = {}

        # Dividing data into classes
        # counting and storing unique words
        
        for (sentence, class_name) in zip(x, y):
            sentence = sentence.lower()
            # if this is the first time a class appears,
            # create an empty array for it
            if(not divided_data.get(class_name)):
                divided_data[class_name] = []
            
            # add the data to its class
            divided_data[class_name].append(sentence)
            
            # filtering words such that they're not 
            # stop words nor have length < 2 
            words_of_sentence = sentence.split(" ")
            x_words = words_of_sentence
            
            for word in x_words:
                # mapping: word -> its index
                self.words[word] = len(self.words)
                self.words_count[word] = self.words_count.get(word, 0) + 1
                
                
        #self.divided_data = divided_data
        self.classes: list[BayesNaiveClass] = []
        
        for d in divided_data:
            class_name = d
            class_data = divided_data[d]
            # P(class) probability of class in data
            p_class = len(class_data)/len(x)

            self.classes.append(BayesNaiveClass(
                class_name, class_data, p_class, self.words
            ))

    def predict(self, text: str):
        text = helper.clean_text(text)
        words = [w.strip() for w in text.split(" ") if w.strip() != ""]
        
        # initial probabilities are the P(class)
        classes_preds = [c.p_class for c in self.classes]
        #classes_preds = [1 for c in self.classes]
        
        existing_words_impact = {}

        for word in words:
            existing_words_impact[word] = [1e-10, 'None']
            
            for i in range(len(classes_preds)):
                prob_of_word = self.classes[i].words_probs.get(word, -1)
                
                if(prob_of_word == -1):
                    # this happens when the data we intend to predict
                    # has a word that we HAVE NOT processed
                    prob_of_word = 1e-10
                
                if(prob_of_word > existing_words_impact[word][0]):
                    existing_words_impact[word] = [prob_of_word, self.classes[i]]

                # updating each class probability by multiplying
                # it by the probability of the word appearing
                # in the class
                classes_preds[i] += np.log2(prob_of_word)
                #or
                # classes_preds[i] *= prob_of_word
        
        words_impact = []

        sorted_keys = sorted(existing_words_impact, key = lambda x: existing_words_impact.get(x)[0])
        for word in sorted_keys:
            prob = existing_words_impact[word][0]
            class_object: BayesNaiveClass = existing_words_impact[word][1]
            
            if(prob > 1e-10):
                words_impact.append({
                    "word": word,
                    "impact": prob,
                    #"class": class_object,
                    "class_name": class_object.class_name
                })
            
        return {
            "classes_preds": classes_preds,
            "words_impact": words_impact
        }

    def pred_textual(self, text, top = 3):
        text = helper.clean_text(text)
        
        result = self.predict(text)
    
        pred = result.get("classes_preds")
        pred_perc = helper.get_softmax(pred)
        
        words_impact = result.get("words_impact")

        top_indices = np.argsort(pred)[-top:][::-1]
        #top_classes = [  (self.classes[i], pred_perc[i] ) for i in top_indices   ]
        top_classes = [  (self.classes[i].class_name, pred_perc[i] ) for i in top_indices   ]
        
        return {
            "words_impact": words_impact,
            "top_classes": top_classes
        }
    


    
class BayesNaiveClass:
    def __init__(self, class_name, class_data, p_class, total_words):
        
        self.class_name = class_name
        self.class_data = class_data
        self.p_class = p_class

        self.number_of_words = 0
        self.words_count = {}
        # Counting words occurances and number of words
        for data in self.class_data:
            for word in data.split(" "):
                self.number_of_words += 1
                count = self.words_count.get(word, 1)
                
                self.words_count[word] = count + 1

        # Calculating P(word | class)
        # what are the chances of a word appearing in a given class
        self.words_probs = {}
        for word in total_words:
            count = self.words_count.get(word, 0) 
            if (count == 0):
                count = 1e-10 #additive smoothin

            word_prob = count / (self.number_of_words + len(total_words))
            self.words_probs[word] = word_prob


    def top_used_words(self, top=10):
        words = list(self.words_probs.keys())
        wordsProbs = list(self.words_probs.values())

        top_indexes = np.argsort(wordsProbs)[::-1][:top]        
        return [words[i] for i in top_indexes]

