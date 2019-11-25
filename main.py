from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import nltk
import string
import random

# getting my hands dirty :p

file = open('./chatbot.txt', 'r', errors='ignore')
content = file.read().lower()
sentences = nltk.sent_tokenize(content)
words = nltk.word_tokenize(content)


# pre-processing shit

l = nltk.stem.WordNetLemmatizer()


def LTokens(tokens):
    return [l.lemmatize(token) for token in tokens]


remove_puncts = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(t):
    return LTokens(nltk.word_tokenize(t.lower().translate(remove_puncts)))


Greetings_input = ("hello", "hi", "hey", "greetings",
                   "sup", "what's up", "hola", "hie",)
Greetings_output = ("hi", "hey", "hi there", "You know what, I guess today is lucky as you are talking to me",
                    "I am glad you called!", "hey there", "*nods*", "hiiiiii")


def greetings(sent):
    for word in sent.split():
        if word.lower() in Greetings_input:
            return random.choice(Greetings_output)

# working for actual responsing shit


def response(user_response):
    dever_response = ''
    sentences.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentences)
    vals = cosine_similarity(tfidf[-1], tfidf)
    #idx = vals.argsort()[0][-2]
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        dever_response = dever_response + "Sorry, I am having trouble understanding that!"
        return dever_response
    else:
        dever_response = dever_response + sentences[idx]
        return dever_response


flag = True
print("DEVER : Hello, I am dever, a simple Chatbot ._.")
print("If u wanna exit just type bue")

while(flag == True):
    print("User : ", end="")
    user_response = input()
    user_response = user_response.lower()
    if(user_response == 'bye'):
        flag = False
        print("DEVER : Bye! nice talking to you!")
    else:
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("Ur welcome, my friend")
        else:
            if(greetings(user_response) != None):
                print("DEVER : " + greetings(user_response))
            else:
                print("DEVER: ", end="")
                print(response(user_response))
                sentences.remove(user_response)
