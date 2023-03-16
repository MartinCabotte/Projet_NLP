import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim
from typing import Union, Iterable
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class iSarcasmDataset(Dataset):
  def __init__(self, sentence, labels):
        self.sentence = sentence
        self.labels = labels

  def __len__(self):
        return len(self.sentence)

  def __getitem__(self, index):
        x = self.sentence[index]
        y = self.labels[index]

        return x, y


print("--- Chargement des datasets ---")


#On donne les informations du dataset
directory = "dataset/"
train_name = "train.En.csv"
test_name = "task_A_En_test.csv"


#On défini les paramètres du dataloader
params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 1}


#On créé nos données de test
panda_Train = pd.read_csv(directory+train_name, index_col=[0])
X_train = panda_Train['tweet']
y_train = panda_Train['sarcastic']

cpt = 0
for phrase in X_train:
    if pd.isna(phrase):
        print("ERROR - il y a des NAN dans le dataset")
        print("--- Traitement en cours ---")
    cpt += 1

indsNan = []
cpt = 0
for phrase in X_train:
    if pd.isna(phrase):
        indsNan.append(cpt)
    cpt += 1
    
X_train = X_train.drop(indsNan)
y_train = y_train.drop(indsNan)

cpt = 0
for phrase in X_train:
    if pd.isna(phrase):
        print("ERROR - il y a encore des NAN dans le dataset")
    cpt += 1

#On créé notre dataset pyTorch
train_iSarcasm = iSarcasmDataset(X_train, y_train)
training_generator = DataLoader(train_iSarcasm, **params)


#On regarde que tout soit bien chargé
for i, (batch, labels) in enumerate(training_generator):
    if i > 0:
        break
    print("Batch size : "+str(len(batch)))
    
#On va créer nos données de validation
panda_Test = pd.read_csv(directory+test_name, index_col=[0])
X_test = panda_Test.index
y_test = panda_Test['sarcastic']

#On créé notre dataset pyTorch
test_iSarcasm = iSarcasmDataset(X_test, y_test)
test_generator = DataLoader(test_iSarcasm, **params)

#On regarde que tout soit bien chargé
for i, (batch, labels) in enumerate(test_generator):
    if i > 0:
        break
    print("Batch size : "+str(len(batch)))
    
print("--- Chargement des datasets terminé ---")


# On créé le vocabulaire

print("--- Traitement sur notre vocabulaire ---")

list_of_words = []
for phrase in X_train:
    list_of_words.extend(phrase.split())
    
list_of_words = np.unique(list_of_words)
len_Vocabulary = len(list_of_words)

print("La longueur du vocabulaire est de : "+str(len_Vocabulary))

print("--- Création du Transformer basique ---")

class Transformer(nn.Module):
    def __init__(self, len_vocab, len_embedding, nhead, num_layers):
        super(Transformer, self).__init__()
        
        #On créé la layer de l'embedding
        self.embedding = nn.Embedding(len_vocab, len_embedding)

        #Le transformer
        self.transformer = nn.Transformer(len_embedding, nhead, num_layers)

        #La layer fully connected pour déterminer si la phrase est sarcastique ou non
        self.fc = nn.Linear(len_embedding, 2)

    def forward(self, x):
        
        #Pass avant de notre modèle
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x
    
print("--- Création du Transformer basique terminée ---")

tokenizer = get_tokenizer("basic_english")
tokens = [tokenizer(doc) for doc in list(X_train)]

voc = build_vocab_from_iterator(tokens)

print(voc.__getitem__("the"))

def my_one_hot(voc, keys: Union[str, Iterable]):
    if isinstance(keys, str):
        keys = [keys]
    return F.one_hot(torch.tensor(voc(keys)), num_classes=len(voc))

# Paramètres de notre modèle
taille_embeddings = 10
nhead = 2
num_layers = 3
n_epoch = 1


# On va optimiser notre modèle
import torch.optim as optim

# On déclare notre modèle
model = Transformer(len_Vocabulary,taille_embeddings,nhead,num_layers)

# On déclare notre loss
loss = nn.CrossEntropyLoss()

# On se déclare un optimiseur qui effectuera la descente de gradient
optimizer = optim.Adam(model.parameters())

# L'historique pour print plus tard
loss_history, train_accuracy_history, valid_accuracy_history = [], [], []

# On réalise notre nombre d'epochs
for epoch in range(n_epoch):
    
    running_loss = 0
    # On loop sur le batch
    for i, (batch, labels) in enumerate(training_generator):
        
        tokens = [tokenizer(doc) for doc in list(batch)]
        print(tokens)
        print(my_one_hot(voc,tokens))
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(my_one_hot(voc,tokens))
        loss = loss(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            
#Entrainement terminé