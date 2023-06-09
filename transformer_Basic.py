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
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

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
batch_size = 5
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}


#On créé nos données de test
panda_Train = pd.read_csv(directory+train_name, index_col=[0])
X_train = panda_Train['tweet']
y_train = panda_Train['sarcastic']

#On va créer nos données de validation
panda_Test = pd.read_csv(directory+test_name, index_col=[0])
X_test = panda_Test.index
y_test = panda_Test['sarcastic']

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
tokenized_sequences_train = [tokenizer(doc) for doc in list(X_train)]
tokenized_sequences_test = [tokenizer(doc) for doc in list(X_test)]

tokenized_sequences = []
for i in tokenized_sequences_train: 
    tokenized_sequences.append(i)
for i in tokenized_sequences_test: 
    tokenized_sequences.append(i) 

max_sequence_length = max([len(i) for i in tokenized_sequences]) # dimension maximale des sequences
voc = build_vocab_from_iterator(tokenized_sequences)
indexed_sequences_train = [torch.tensor(voc.forward(sequence)) for sequence in tokenized_sequences_train]
X_train = [torch.cat([sequence,torch.tensor([voc.__len__()]).expand(max_sequence_length- len(sequence))]) for sequence in indexed_sequences_train]
y_train = torch.tensor(y_train.values).unsqueeze(1).float()

indexed_sequences_test = [torch.tensor(voc.forward(sequence)) for sequence in tokenized_sequences_test]
X_test = [torch.cat([sequence,torch.tensor([voc.__len__()]).expand(max_sequence_length- len(sequence))]) for sequence in indexed_sequences_test]
y_test = torch.tensor(y_test.values).unsqueeze(1).float()


#On créé notre dataset pyTorch
train_iSarcasm = iSarcasmDataset(X_train, y_train)
training_generator = DataLoader(train_iSarcasm, **params)


#On regarde que tout soit bien chargé
for i, (seq, labels) in enumerate(training_generator):
    if i > 0:
        break
    print("Batch size : "+str(len(seq)))
    


#On créé notre dataset pyTorch
test_iSarcasm = iSarcasmDataset(X_test, y_test)
test_generator = DataLoader(test_iSarcasm, **params)

#On regarde que tout soit bien chargé
for i, (seq, labels) in enumerate(test_generator):
    if i > 0:
        break
    print("Batch size : "+str(len(seq)))
    
print("--- Chargement des datasets terminé ---")

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

        # On créé la layer de l'embedding
        self.embedding = nn.Embedding(len_vocab+1, embed_size, padding_idx=len_vocab)

        # Le transformer
        self.pos_encoder = PositionalEncoding(
            d_model=embed_size,
            dropout=0.01,
            vocab_size=len_vocab,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dropout=0.01,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # La layer fully connected pour déterminer si la phrase est sarcastique ou non
        self.fc = nn.Linear(8832, 512)
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 1)


    def forward(self, x):
        
        #Pass avant de notre modèle
        x = self.embedding(x) * math.sqrt(self.len_embedding)
        # x = self.pos_encoder(x)
        # x = self.transformer_encoder(x)
        x = x.permute(1,0,2)
        # print(x.size())
        x = x.flatten(1,2)
        # print(x.size())
        x = self.fc(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
    
print("--- Création du Transformer basique terminée ---")

def one_hot_encoding(voc, list_Of_Words):
    indices = voc.forward(list_Of_Words)
    matrixToReturn = np.zeros([voc.__len__(),len(list_Of_Words)])
    matrixToReturn[indices,np.arange(len(list_Of_Words))] = 1
    return matrixToReturn

# Paramètres de notre modèle
taille_embeddings = 64
nhead = 8
num_layers = 6
n_epoch = 15


# On va optimiser notre modèle
import torch.optim as optim

# On déclare notre modèle
model = Transformer(voc.__len__(),taille_embeddings,nhead,num_layers).to(device)

# On déclare notre loss
criterion = nn.BCELoss()

# On se déclare un optimiseur qui effectuera la descente de gradient
optimizer = optim.Adam(model.parameters(), lr=0.001)

# L'historique pour print plus tard
loss_history, train_accuracy_history, valid_accuracy_history, test_accuracy_history = [], [], [], []
train_f1_history, test_f1_history = [], []
# On réalise notre nombre d'epochs
for epoch in tqdm(range(n_epoch)):
    
    loss_train = []
    accuracy_train = []
    f1_train = []
    # On loop sur le batch
    for i, (seq, labels) in enumerate(training_generator):
        
        sequences = torch.transpose(torch.cat([sequence[None] for sequence in  list(seq)]),0,1).to(torch.int64).cpu().to(device)
        labels = labels.cpu().to(device)  

        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = model(sequences)
        # print(np.unique(outputs.detach().numpy().round()))
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        # print statistics
        accuracy_train.append(100*(np.sum((outputs.detach().numpy().round() == labels.detach().numpy()))/len(labels.detach().numpy())))
        loss_train.append(loss.item())
        for i in outputs.detach().numpy().round():
            f1_train.append(i)

    loss_history.append(np.mean(loss_train))
    train_accuracy_history.append(np.mean(accuracy_train))
    train_f1_history.append(f1_score(y_train,f1_train))

    accuracy_test = []
    f1_test = []
    for i, (seq, labels) in enumerate(test_generator):

        sequences = torch.transpose(torch.cat([sequence[None] for sequence in  list(seq)]),0,1).to(torch.int64).cpu().to(device)
        labels = labels.cpu().to(device)  

        outputs_test = model(sequences)   

        loss = criterion(outputs_test, labels)

        accuracy_test.append(100*(np.sum((outputs_test.detach().numpy().round() == labels.detach().numpy()))/len(labels.detach().numpy())))
        
        for i in outputs_test.detach().numpy().round():
            f1_test.append(i)

    test_accuracy_history.append(np.mean(accuracy_test))
    test_f1_history.append(f1_score(y_test,f1_test, average = "binary", pos_label = 1))


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

            
#Entrainement terminé
