import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

phase_test = False

#Chargement de mes fonctions
from myfunctions import *

###########################################
###########################################
#          Chargement du dataset          #
###########################################
###########################################

#On donne les informations de l'emplacement du dataset
directory = "dataset/"
train_name = "train.En.csv"
test_name = "task_A_En_test.csv"

#On load les datasets
X_train, y_train = loadCsv(directory,train_name,test=False)
X_test, y_test = loadCsv(directory,test_name,test=True)


###########################################
###########################################
# On va maintenant traiter notre dataset  #
###########################################
###########################################

if phase_test:

    #Aucun mot positif ...
    # 1] On compte les mots positifs
    pos_word_train = pos_neg_Count(X_train,positive=True)
    print(np.unique(pos_word_train))

    #Aucun mot négatif ...
    # 2] On compte les mots négatifs
    neg_word_train = pos_neg_Count(X_train,positive=False)
    print(np.unique(neg_word_train))


    # 3] On regarde les hashtags (à envoyer dans une couche embeddings)
    hashtags = hashtag(X_train)
    print(np.unique(hashtags)[0:10])

    # 4] On compte les ponctuations et le ratio est exclam/interr
    atCount, exclaCount, interCount, ratio = ponctuation(X_train)
    print(atCount[0:10])
    print(exclaCount[0:10])
    print(interCount[0:10])
    print(ratio[0:10])




################################################
################################################
# Création de ce qui sera donné à manger au CNN#
################################################
################################################


#Premièrement, on fait nos traitement de données (feature engineering)
print("--- TRAIN ---")
print("--- Positive/Négative count ---")
pos_word_train = pos_neg_Count(X_train,positive=True)
neg_word_train = pos_neg_Count(X_train,positive=False)

print("--- Ponctuation ---")
atCount_train, exclaCount_train, interCount_train, ratio_train = ponctuation(X_train)

print("--- TEST ---")
print("--- Positive/Négative count ---")
pos_word_test = pos_neg_Count(X_test,positive=True)
neg_word_test = pos_neg_Count(X_test,positive=False)
print("--- Ponctuation ---")
atCount_test, exclaCount_test, interCount_test, ratio_test = ponctuation(X_test)

#Deuxièmement, on créé notre vocabulaire sur le train et on transforme notre dataset

print("--- Tokenisation, construction du vocabulaire ---")
all_tokens = getTokens(X_train)
all_tokens.append("<UKN>")
all_tokens.append("<PAD>")

max_phrase_length = maxlen(X_train,X_test)

print("--- Tokenisation du train ---")
X_train = tokenizeMyCorpus(X_train,all_tokens,max_phrase_length)
print("--- Tokenisation du test ---")
X_test = tokenizeMyCorpus(X_test,all_tokens,max_phrase_length)

###########################################
###########################################
#   Chargement du dataset dans PyTorch    #
###########################################
###########################################

import torch

#On défini les paramètres du dataloader
batch_size = 5
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}

X_train = torch.from_numpy(X_train)
y_train = torch.tensor(y_train.values).unsqueeze(1).float()
features_train = torch.from_numpy(np.transpose(np.array([pos_word_train,neg_word_train,atCount_train, exclaCount_train, interCount_train, ratio_train]))).float()
# print(features_train.size())
# print(X_train.size())

train_generator = loadGenerator(X_train,y_train,features_train,params)

X_test = torch.from_numpy(X_test)
y_test = torch.tensor(y_test.values).unsqueeze(1).float()
features_test = torch.from_numpy(np.transpose(np.array([pos_word_test,neg_word_test,atCount_test, exclaCount_test, interCount_test, ratio_test]))).float()

test_generator = loadGenerator(X_test,y_test,features_test,params)


###########################################
###########################################
#            On défini le CNN             #
###########################################
###########################################


from torch import nn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class CNNBasic(nn.Module):
    def __init__(self, len_vocab, embed_size):
        super(CNNBasic, self).__init__()

        self.len_embedding = embed_size

        # On créé la layer de l'embedding
        self.embedding = nn.Embedding(len_vocab+1, embed_size, padding_idx=len_vocab)

        # La layer fully connected pour déterminer si la phrase est sarcastique ou non
        self.leaky1 = nn.LeakyReLU()
        self.leaky2 = nn.LeakyReLU()
        self.fc = nn.Linear(17670, 512)
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 1)


    def forward(self, x, engineering):
        
        #Pass avant de notre modèle
        x = self.embedding(x) * math.sqrt(self.len_embedding)

        x = x.permute(1,0,2)
        # print(x.size())
        x = x.flatten(1,2)

        # print(x.type())

        # print(x.size())
        # print(engineering.size())

        #On ajoute les features
        x = torch.cat((x,engineering),dim=1)
        
        # print(x.size())

        # print(x.type())
        
        x = self.fc(x)
        x = self.leaky1(x)
        x = self.fc1(x)
        x = self.leaky2(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

###########################################
###########################################
#           Entrainement du CNN           #
###########################################
###########################################

# Paramètres de notre modèle
taille_embeddings = 128
n_epoch = 40

# On charge le CPU ou le GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# On va optimiser notre modèle
import torch.optim as optim

# On déclare notre modèle
model = CNNBasic(len(all_tokens),taille_embeddings).to(device)

# On déclare notre loss
criterion = nn.BCELoss()

# On se déclare un optimiseur qui effectuera la descente de gradient
optimizer = optim.Adam(model.parameters(), lr=0.0001)
                       
# L'historique pour print plus tard
loss_history, train_accuracy_history, valid_accuracy_history, test_accuracy_history = [], [], [], []
train_f1_history, test_f1_history = [], []

# On réalise notre nombre d'epochs
for epoch in tqdm(range(n_epoch)):
    
    loss_train = []
    accuracy_train = []
    f1_train = []
    # On loop sur le batch
    for i, (seq, labels, engineering) in tqdm(enumerate(train_generator)):
        
        sequences = torch.transpose(torch.cat([sequence[None] for sequence in  list(seq)]),0,1).to(torch.int64).cpu().to(device)
        labels = labels.cpu().to(device)
        engineering = engineering.cpu().to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = model(sequences,engineering)

        
        # print(np.unique(outputs.detach().numpy().round()))
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        # print(np.unique(outputs.detach().numpy()))

        # print statistics
        loss_train.append(loss.item())
        for i in outputs.detach().numpy().round():
            f1_train.append(i)

    loss_history.append(np.mean(loss_train))
    train_accuracy_history.append(np.mean(accuracy_score(y_train,f1_train)))
    train_f1_history.append(f1_score(y_train,f1_train))

    accuracy_test = []
    f1_test = []
    for i, (seq, labels, engineering) in enumerate(test_generator):

        sequences = torch.transpose(torch.cat([sequence[None] for sequence in  list(seq)]),0,1).to(torch.int64).cpu().to(device)
        labels = labels.cpu().to(device)  
        engineering = engineering.cpu().to(device)

        outputs_test = model(sequences,engineering)   

        for i in outputs_test.detach().numpy().round():
            f1_test.append(i)

    test_accuracy_history.append(np.mean(accuracy_score(y_test,f1_test)))
    test_f1_history.append(f1_score(y_test,f1_test))


plt.plot(train_accuracy_history, 'r') # plotting t, a separately 
plt.plot(test_accuracy_history, 'g') # plotting t, a separately 
plt.plot(loss_history, 'b') # plotting t, a separately 
plt.show()

print(train_accuracy_history)
print("---")
print(test_accuracy_history)
print("---")
print(loss_history)
print("---")
print(train_f1_history)
print("---")
print(test_f1_history)

            
#Entrainement terminé)


