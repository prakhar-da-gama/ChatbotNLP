import numpy as np
import tensorflow as tf
import tflearn
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import json
import pickle
import warnings
warnings.filterwarnings("ignore")

nltk.download("punkt")# required package for tokenization
nltk.download("wordnet")# word database

print("Processing the Intents.....")
with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        w = nltk.word_tokenize(pattern)
        
        words.extend(w)
        
        documents.append((w, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print("Stemming, Lowering and Removing Duplicates.......")
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))


classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

training = []
output = []

output_empty = [0] * len(classes)


for doc in documents:
    
    bag = []
    
    pattern_words = doc[0]
    
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
   
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)


train_x = list(training[:,0])
train_y = list(training[:,1])

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)

model.save('model.tflearn')

import pickle

pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']


with open('intents.json') as json_data:
    intents = json.load(json_data)
    

model.load('./model.tflearn')

def clean_up_sentence(sentence):
    # It Tokenize or Break it into the constituents parts of Sentense.
    sentence_words = nltk.word_tokenize(sentence)
    # Stemming means to find the root of the word.
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Return the Array of Bag of Words: True or False and 0 or 1 for each word of bag that exists in the Sentence
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                
    return(np.array(bag))

ERROR_THRESHOLD = 0.25


def classify(sentence):
    
    results = model.predict([bow(sentence, words)])[0]
    
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1])) #Tuppl -> Intent and Probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # That Means if Classification is Done then Find the Matching Tag.
    if results:
        # Long Loop to get the Result.
        while results:
            for i in intents['intents']:
                # Tag Finding
                if i['tag'] == results[0][0]:
                    # Random Response from High Order Probabilities
                    return random.choice(i['responses'])

#questions = ["hi", "how are you"]
if __name__ == '__response__':
   response(sentence = "", userID='123', show_details=False),


############################################################
# Callback function called on update config
############################################################


############################################################
# Callback function called on each execution pass
############################################################
"""def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        res = response(text)
        output.append(res)
        

    return SimpleText(dict(text=output))"""
