# Projet_NLP

## Executer le code

Pour executer le code, il suffit de run les fichiers suivants : Feature_engineering.py et HBC.py

Ces fichiers vont alors entrainer les modèles et sortir les courbes voulues.

## A quoi servent ces fichiers ?

Le fichier Feature_engineering.py est le fichier dans lequel les features ajoutées ont été testées. On a le choix de le run avec ou sans les nouvelles features pour comparer les résultats en sortie d'algorithme.

Le fichier HBC.py entraine les modèles dérivés de Bert donnés dans le rapport et montre en sortie les courbes d'entrainement

## Où se trouvent les datasets ?

Les datasets se trouvent dans le dossier *dataset*. Principalement, 3 datasets sont interessant : dataframe_merged.csv pour avoir un dataset d'entrainement plus conséquent ainsi que task_A_en_test.csv et train.En.csv qui sont les datasets de la compétition.

Les datasets sarcasm2.csv et Sarcasm_Headlines_Dataset.csv sont les datasets qui ont été ajoutés à train.En.csv pour créer dataframe_merged.csv.

Toute la concaténation peut être analysée dans le fichier mergedata.ipynb

## Comment entrainer notre modèle sur une autre dataset ?

Pour entrainer notre modèle sur d'autres datasets, il suffit de changer dans les codes de nos fichiers la variable *train_name* par le nom souhaité.

## Comment modifier les réseaux ?

Pour modifier les réseaux, il faut entrer dans les codes Feature_engineering.py et HBC.py et se rendre dans la partie PyTorch pour ajouter/modifier les couches. Il faut bien faire attention aux nombre de neuronnes à mettre dans les couches pleinement connectés pour éviter de faire planter le programme si vous souhaitez faire une modification dans ces fichiers.
