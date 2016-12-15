
import os
import pickle


def excute(alogrithm, text_extraction, content):
    if alogrithm == 'KNN':
        if text_extraction == 'Bag of word':
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/K-NN_Bag Of Words_eng.pickle"
        else:
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/K-NN_TF IDF_eng.pickle"
    else:
        if text_extraction == 'Bag of word':
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_Bag Of Words_eng.pickle"
        else:
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_TF IDF_eng.pickle"
    a = pickle.load(open(path, 'rb'))
    result = a.predict([content])
    return result[0]
