from lxml import etree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import classification_report
import sys
from collections import namedtuple

confParti=namedtuple("confParti","parti,confiance")

def get_root_from_file(file):
    tree = etree.parse(file)
    root = tree.getroot()
    return tree, root

def get_X_Y_from_root(root):
    X = []
    y = []

    for child in root:
        if child.tag=="doc":
            for c in child:
                if c.tag=="EVALUATION":
                    for c1 in c:
                        for parti in c1:
                            y.append(parti.get("valeur"))
                elif c.tag=="texte":
                    X.append(" ".join([p.text for p in c if p.text]))
    
    return X,y

def rewrite_from_pred(tree,confPartiList,output):
    i=0
    root=tree.getroot()
    for child in root:
        if child.tag=="doc":
            for c in child:
                if c.tag=="EVALUATION":
                    for eval_parti in c:
                        eval_parti.clear()
                        eval_parti.set("nombre","5")
                        for confParti in confPartiList[i]:
                            etree.SubElement(eval_parti,"PARTI")
                            for parti in eval_parti:
                                parti.set("valeur",confParti.parti)
                                parti.set("confiance",confParti.conf) 
    tree.write(output)
    return root

def from_y_pred_to_confPartiList(classes,y_pred):
    confPartiList=[]
    for pred in y_pred:
        tmp=[]
        for conf,parti in zip(pred,classes):
            tmp.append(confParti(parti,conf))
        confPartiList.append(tmp)
    return confPartiList


if __name__=="__main__":
    file_train=sys.argv[1]
    file_test=sys.argv[2]
    tree_train, root_train = get_root_from_file(file_train)
    X_train, y_train = get_X_Y_from_root(root_train)
    tree_test, root_test = get_root_from_file(file_test)
    X_test, y_test = get_X_Y_from_root(root_test)

    vectorizer = CountVectorizer(stop_words="english")

    X_train = vectorizer.fit_transform(X_train)
    y_train=np.array(y_train)

    clf=SVC(probability=True)

    clf.fit(X_train[:500],y_train[:500])

    y_pred=clf.predict_proba(X_test)

    classes=clf.classes_

    confPartiList=from_y_pred_to_confPartiList(classes,y_pred)

    rewrite_from_pred(tree_test,confPartiList,"truc_test.xml")