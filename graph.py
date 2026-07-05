from bayes_naive import BayesNaive
import matplotlib.pyplot as plt

colors = [
    'red',
    'green',
    'blue',
    'magenta',
    'black',
    'gray',
    'orange',
    'purple',
    'brown',
    'pink',
    'lightblue',
    'lightgreen',
    'darkred',
    'darkgreen',
    'darkblue',
    'lightgray',
    'gold',
    'navy',
    'teal',
    'violet',
    'indigo',
    'coral',
    'salmon',
    'lavender',
    'khaki',
    'plum',
    'tan'
]

def data_distribution(bn: BayesNaive):
    x = [c.class_name for c in bn.classes] # emotions classes
    y = [len(c.class_data) for c in bn.classes] # frequency

    plt.bar(x, y, color = 'skyblue')
    plt.title(f'Data Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Amount')

    plt.grid()
    plt.show()

def top_words_freqs(bn: BayesNaive):    
    top_words =  bn.top_used_words(10)

    x = top_words
    ci = 0
    
    for c in bn.classes:
        y = []
        for word in top_words:
            y.append( c.words_count.get(word) )

        plt.plot(x, y, color = colors[ci % len(colors)], marker = "h",
                 label = c.class_name)
        ci+= 1
        
        plt.title("Classes & Most Frequent Words")
        plt.xlabel = "word"
        plt.ylabel = "amount"

    plt.legend()
    plt.grid()
    plt.show()
            
        
