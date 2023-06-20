import torchaudio
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cdist
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np 
import os

queries = {}
test = {}
results = {}
#loading of the model
root = "/home/getalp/leferrae/post_doc/classifier_zing/"
classifier = EncoderClassifier.from_hparams(source=root+"content/best_model/", hparams_file=root+'hparams_inference.yaml', savedir=root+"content/best_model/")

print("extraction of the embeddings for the queries")
#extraction of the embeddings for the queries
rootQueries = "/home/getalp/leferrae/post_doc/corpora/guinee_casa/queries/"
for elt in os.listdir(rootQueries):
    if ".wav" in elt:
        signal, fs = torchaudio.load(rootQueries+elt)
        embedding = classifier.encode_batch(signal)
        queries[elt.replace(".wav", "")] = embedding

print("extraction of the embeddings for the test set")
#extraction of the embeddings for the test set
rootTest = "./wordsTest/"
for i, elt in enumerate(os.listdir(rootTest)):
    signal, fs = torchaudio.load(rootTest+elt)
    embedding = classifier.encode_batch(signal)
    name = elt.split("_")[0]
    test[elt.replace(".wav", "")] = {"embedding" : embedding, "label" : name}
print("evaluation...")
#computation of the distance between queries and test set
for testWord in test:
    scores = []
    for query in queries:
        score = cdist(test[testWord]["embedding"][0], queries[query][0], metric='minkowski')
        scores.append((query, score[0][0]))
    scores.sort(key=lambda x : x[1])
    results[testWord] = scores


#sorting and evaluation of the top 1 5 10
top1 = 0
top2 = 0
top5 = 0
tot = 0
for elt in results:
    tot+=1
    name = elt.split("_")[0]
    list2 = [x[0] for x in results[elt][:2]]
    list5 = [x[0] for x in results[elt][:5]]

    if name == results[elt][0][0]:
        top1+=1
    if name in list2:
        top2+=1
    if name in list5:
        top5+=1
print(top1/tot*100, top2/tot*100, top5/tot*100)
print(tot)