# -*- coding: utf-8 -*-

import math
import argparse
from collections import defaultdict
def loaddata(corpus_file):
    f=open(corpus_file)
    labels=defaultdict(int)
    label_words={}
    docs=[]
    words=set()
    total=0
    for line in f.readlines():
        arr=line.strip().split('/')
        if len(arr)< 2:continue
        tokenizer=list(arr[1])
        if len(tokenizer)==0:continue
        labels[arr[0]] +=1
        total+=1
        docs.append((arr[0],tokenizer))
        words.update(tokenizer)
    label_words = dict(zip(words, range(len(words))))
    f.close()
    return labels,label_words,total,docs
def model(labels,labels_words,total,docs):
    bw=np.zeros([total, len(labels_words)])
    df=np.zeros([1, len(labels_words)])
    for idx,(label,doc) in  enumerate(docs):
         tf[idx] /= np.max(tf[idx])
        for wd in doc:
            df[0, self.label_words[wd]] += 1
            bw=[idx,label_words[wd]]+=1
    idf = np.log(float(total)) - np.log(df)
    res=np.multiply(bw, idf)
    return res
def calculateprob(labels,label_words,docs,total)
    for label in labels:
        y_prob[label] = float(labels[label]) / total
    c_prob = np.zeros([len(label), len(label_words)])
    T = np.zeros([len(labels), 1])
    for idx in range(len(docs)):
        tid = labels.keys().index(docs[idx][0])
        c_prob[tid] += feature[idx]
        T[tid] = np.sum(c_prob[tid])
    c_prob /= T
    return y_prob,c_prob
def get_vec(query,label_words):
    vec = np.zeros([1, len(label_words)])
    words=list(query)
    for word in words:
        if word in label_words:
            vec[0, label_words[word]] += 1
    return vec
def run(args):
    f=open(args.input_file)
    ff=open(args.output,'w')
    labels,labels_words,total,docs=loaddata(args.corpus)
    for line in f.readlines():
        query=line.strip()
        feature=model(labels,labels_words,total,docs)
        y_prob,c_prob=calculateprob(labels,label_words,docs,total)
        sen=get_vec(query,label_words)
        max_score=0
        rs=0
        for y, pc in zip(y_prob, c_prob):
            score = np.sum(vec * pc * y_prob[y])
            if score > max_score:
                max_score = score
                ret = y
        ff.write(query+'\t'+label+'\t'+str(score)+'\n')
    f.close()
    ff.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--corpus",help="corpus")
    parser.add_argument("-i","--input_file",help="input_file")
    parser.add_argument("-o","--output",help="output")
    args = parser.parse_args()
    run(args)
