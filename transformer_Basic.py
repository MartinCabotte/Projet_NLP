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
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
# device = 'cpu'

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
batch_size = 20
params = {'batch_size': batch_size,
          'shuffle': True,
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

# https://stackoverflow.com/questions/67312321/how-to-remove-urls-between-texts-in-pandas-dataframe-rows
X_train = X_train.str.replace(r' https:\/\/.*', ' ', regex=True).str.strip()

cpt = 0
for phrase in X_train:
    if pd.isna(phrase):
        print("ERROR - il y a encore des NAN dans le dataset")
    cpt += 1

# Prétraitement de X_train
tokenizer = get_tokenizer("basic_english")
tokenized_sequences = [tokenizer(doc) for doc in list(X_train)]
max_sequence_length = max([len(i) for i in tokenized_sequences]) # dimension maximale des sequences
voc = build_vocab_from_iterator(tokenized_sequences)
indexed_sequences = [torch.tensor(voc.forward(sequence)) for sequence in tokenized_sequences]
X_train = [torch.cat([sequence,torch.tensor([voc.__len__()]).expand(max_sequence_length- len(sequence))]) for sequence in indexed_sequences]
y_train = torch.tensor(y_train.values).unsqueeze(1).float()

print(type(X_train))
print(type(y_train))


#On créé notre dataset pyTorch
train_iSarcasm = iSarcasmDataset(X_train, y_train)
training_generator = DataLoader(train_iSarcasm, **params)


#On regarde que tout soit bien chargé
for i, (seq, labels) in enumerate(training_generator):
    if i > 0:
        break
    print("Batch size : "+str(len(seq)))
    
#On va créer nos données de validation
panda_Test = pd.read_csv(directory+test_name, index_col=[0])
X_test = panda_Test.index
y_test = panda_Test['sarcastic']

#On créé notre dataset pyTorch
test_iSarcasm = iSarcasmDataset(X_test, y_test)
test_generator = DataLoader(test_iSarcasm, **params)

#On regarde que tout soit bien chargé
for i, (seq, labels) in enumerate(test_generator):
    if i > 0:
        break
    print("Batch size : "+str(len(seq)))
    
print("--- Chargement des datasets terminé ---")


# On créé le vocabulaire

# print("--- Traitement sur notre vocabulaire ---")

# list_of_words = []
# for phrase in X_train:
#     list_of_words.extend(phrase.split())
    
# list_of_words = np.unique(list_of_words)
# len_Vocabulary = len(list_of_words)

# print("La longueur du vocabulaire est de : "+str(len_Vocabulary))

print("--- Création du Transformer basique ---")


"""
La classe PositionalEncoding est offerte par le tutoriel officiel de PyTorch.
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
torch.nn n'implemente pas cette classe a ce jour.
"""
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, len_vocab, embed_size, nhead, num_layers):
        super(Transformer, self).__init__()

        # 
        self.len_embedding = embed_size

        # Masque pour le transformeur
        self.mask = nn.Transformer.generate_square_subsequent_mask(max_sequence_length).to(device)
        
        # On créé la layer de l'embedding
        self.embedding = nn.Embedding(len_vocab+1, embed_size, padding_idx=len_vocab)

        # Positional encoding 
        self.positional_encoder = PositionalEncoding(embed_size,vocab_size=len_vocab)

        # Le transformer
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_size, nhead), num_layers)
        # self.transformer = nn.Transformer(len_embedding, nhead, num_layers)

        # La layer fully connected pour déterminer si la phrase est sarcastique ou non
        self.fc = nn.Linear(embed_size, 1)


    def forward(self, x, label):
        
        #Pass avant de notre modèle
        x = self.embedding(x) * math.sqrt(self.len_embedding)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x
    
print("--- Création du Transformer basique terminée ---")

def one_hot_encoding(voc, list_Of_Words):
    indices = voc.forward(list_Of_Words)
    matrixToReturn = np.zeros([voc.__len__(),len(list_Of_Words)])
    matrixToReturn[indices,np.arange(len(list_Of_Words))] = 1
    return matrixToReturn

# Paramètres de notre modèle
taille_embeddings = 100
nhead = 2
num_layers = 3
n_epoch = 10


# On va optimiser notre modèle
import torch.optim as optim

# On déclare notre modèle
model = Transformer(voc.__len__(),taille_embeddings,nhead,num_layers).to(device)
# print(voc.__len__())
# print(len_Vocabulary)

# On déclare notre loss
criterion = nn.BCEWithLogitsLoss()

# On se déclare un optimiseur qui effectuera la descente de gradient
optimizer = optim.Adam(model.parameters())

# L'historique pour print plus tard
loss_history, train_accuracy_history, valid_accuracy_history = [], [], []

# On réalise notre nombre d'epochs
for epoch in range(n_epoch):
    
    running_loss = 0
    # On loop sur le batch
    for i, (seq, labels) in enumerate(training_generator):
        
        sequences = torch.transpose(torch.cat([sequence[None] for sequence in  list(seq)]),0,1).to(torch.int64).to(device)
        labels = labels.to(device)  

        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = model(sequences,labels)
        # print(outputs)

        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    
        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0                                                      
            
#Entrainement terminé