# Plan

- Abstract
- Introduction
    - Vectorisation
    - Classifieur
    - tâches 
        - classification de documents
- Méthodologie
- Résultats
- Analyses
- Discussion

# Tables

- Un tableau qui, pour chaque PipeLine, montrer le meilleur modèle
    - score moyen ?
    - meilleur score ?
- Tableau plus détaillé pour la meilleure PipeLine de toutes 

# Figures

- Figure de learning rate et/ou loss over time pour voir à quel rapidité chaque classifieur apprend (+ éventuellement voir si ça overfit ou underfit)
- Schéma qui explique le déroulement de deux pipelines (celle avec juste les trucs de sklearn et celle avec nos propres transformers)
- Figure de decisionTree pour montrer qu'on a regardé nos données avant de commencer à coder

# Misc

- [ ] nettoyer le git pour mettre un lien dans l'article


# Répartition du travail

## Julie Sucrée

- Méthodologie
- Discussion

## Aurélien Sucré

- Abstract
- Introduction

## Den Sucrée

- Relecture du premier jet
- Faire les figures et les tableaux

## A faire ensemble

- Résultats
- Analyse

# Notebook

# TODO

- [ ] faire glove, fastext et w2v pour pouvoir les utiliser dans les pipelines
- [ ] faire les pipelines des classfieurs : 
    - [x] SVM
    - [ ] knn
    - [ ] régression logistique
    - [ ] naives bayes
    - [ ] random forest
- [ ] (faire des transformers pour sentiment analysis et theme)