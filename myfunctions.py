import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch
from nltk.tokenize import word_tokenize

def loadCsv(directory,filename,test=False):

    #On load le dataset
    panda = pd.read_csv(directory+filename, index_col=[0])

    #Les lists qu'on va return
    tweet = []
    sarcasm = []

    #On assigne aux listes
    if test == False:
        tweet = panda['tweet']
        sarcasm = panda['sarcastic']
    else:
        tweet = panda.index
        sarcasm = panda['sarcastic']
    
    #Traitement des NaN
    index = 0
    indsNan = []
    for phrase in tweet:
        if pd.isna(phrase):
            if len(indsNan) == 0:
                print("\nERROR - il y a des NAN dans le dataset "+str(filename)+"\n")
            indsNan.append(index)
        index += 1

    #On drop les Nan    
    tweet = tweet.drop(indsNan)
    sarcasm = sarcasm.drop(indsNan)

    return [tweet, sarcasm]

def pos_neg_Count(tweet, positive = True):
    
    count = []
    array_check = []

    if positive == True:
        with open('poswordlist.txt', 'rt') as c:
            array_check = c.readlines()
    else:
        with open('negwordlist.txt', 'rt') as c:
            array_check = c.readlines()

    for phrase in tqdm(tweet):

        cpt = 0
        
        for mot in word_tokenize(phrase):
            if str(mot) == str(array_check):
                cpt += 1

        count.append(cpt)

    return count

def hashtag(tweet):
    hashtag_list = []
    goingtobehashtag = False

    for phrase in tqdm(tweet):
        for mot in word_tokenize(phrase):

            if goingtobehashtag == True:
                hashtag_list.append(mot)
                goingtobehashtag = False

            if "#" in str(mot):
                goingtobehashtag = True

    
    return hashtag_list

def at(tweet):
    count_at = []

    for phrase in tqdm(tweet):
        cpt = 0

        for mot in word_tokenize(phrase):
            if "@" == mot:
                cpt+=1
        
        count_at.append(cpt)

    return count_at

def ponctuation(tweet):
    count_at = []
    count_exclamation = []
    count_interrogation = []
    ratio = []

    for phrase in tqdm(tweet):
        cpt_at = 0
        cpt_exclamation = 0
        cpt_interrogation = 0

        for mot in word_tokenize(phrase):
            if "@" in mot:
                cpt_at += 1
            if "!" in mot:
                cpt_exclamation += 1
            if "?" in mot:
                cpt_interrogation += 1

        count_at.append(cpt_at)
        count_exclamation.append(cpt_exclamation)
        count_interrogation.append(cpt_interrogation)

        ratioToAppend = 0
        #Ici, on est balancé donc 0.5
        if (cpt_interrogation == 0) and (cpt_exclamation == 0):
            ratio.append(0.5)
        #Ici, on a un dénominateur égale à 0 et un numérateur != de 0, on a donc un ratio de 1 pour 0
        elif cpt_interrogation == 0:
            ratio.append(1)
        #Ici, on append le ratio (exclam / (exclam + inter))
        else:
            ratio.append(cpt_exclamation/(cpt_exclamation + cpt_interrogation))

    return [count_at,count_exclamation,count_interrogation,ratio]
        

def getTokens(tweet):

    tokenList = []
    cpt = 0

    for phrase in tqdm(tweet):

        #Tous les 20, on factorise les tokens pour la mémoire
        if cpt % 20 == 0:
            tokenList = np.unique(tokenList).tolist()

        #On append tous les mots
        for mot in word_tokenize(phrase):
            tokenList.append(mot)

        cpt += 1

    return tokenList

def maxlen(tweet_Train,tweet_Test):
    max = 0

    for phrase in tqdm(tweet_Train):
        cpt = 0
        for _ in word_tokenize(phrase):
            cpt += 1
        
        if max < cpt:
            max = cpt

    for phrase in tqdm(tweet_Test):
        cpt = 0
        for _ in word_tokenize(phrase):
            cpt += 1

        if max < cpt:
            max = cpt

    return max
    
def tokenizeMyCorpus(tweet,vocabulary,maxlength):
    
    tokenizedCorpus = np.zeros([len(tweet),maxlength])
    phrase_index = 0

    #On parcours toutes les phrases
    for phrase in tqdm(tweet):

        #On tokenize notre phrase pour avoir nos mots
        tokens = word_tokenize(phrase)

        #On parcours tous les mots de la phrase
        for index in range(maxlength):

            #Si on a encore des mots dans la phrase à traiter, on les ajoute
            if index < len(tokens):
                #Si le mot est dans le vocabulaire, on l'ajoute, sinon, on donne unknown
                if tokens[index] in vocabulary:
                    tokenizedCorpus[phrase_index,index] = vocabulary.index(tokens[index])
                else:
                    tokenizedCorpus[phrase_index,index] = vocabulary.index("<UKN>")
            #Sinon, on ajoute du padding
            else:
                tokenizedCorpus[phrase_index,index] = vocabulary.index("<PAD>")

        phrase_index += 1

    return tokenizedCorpus

class iSarcasmDataset(Dataset):
  def __init__(self, sentence, labels, engineering):
        self.sentence = sentence
        self.labels = labels
        self.engineering = engineering

  def __len__(self):
        return len(self.sentence)

  def __getitem__(self, index):
        x = self.sentence[index]
        y = self.labels[index]
        engi = self.engineering[index]

        return x, y, engi

def loadGenerator(X,y,engineering,params):
    train_iSarcasm = iSarcasmDataset(X, y, engineering)
    training_generator = DataLoader(train_iSarcasm, **params)

    #On regarde que tout soit bien chargé
    for i, (seq, labels, engi) in enumerate(training_generator):
        if i > 0:
            break
        print("Batch size : "+str(len(seq)))

    return training_generator


class iSarcasmDatasetBasic(Dataset):
  def __init__(self, sentence, labels,):
        self.sentence = sentence
        self.labels = labels

  def __len__(self):
        return len(self.sentence)

  def __getitem__(self, index):
        x = self.sentence[index]
        y = self.labels[index]

        return x, y
  
def loadGeneratorBasic(X,y,params):
    train_iSarcasm = iSarcasmDatasetBasic(X, y)
    training_generator = DataLoader(train_iSarcasm, **params)

    #On regarde que tout soit bien chargé
    for i, (seq, labels) in enumerate(training_generator):
        if i > 0:
            break
        print("Batch size : "+str(len(seq)))

    return training_generator