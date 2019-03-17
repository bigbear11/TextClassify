# -*- coding: utf-8 -*-

import math
import argparse
from collections import defaultdict
def loaddata(corpus_file):
    f=open(corpus_file)
    labels=defaultdict(int)
    labels_words=defaultdict(int)
    total=0
    for line in f.readlines():
        arr=line.strip().split('/')
        if len(arr)< 2:continue
        tokenizer=list(arr[1])
        for item in tokenizer:
            labels[arr[0]] +=1
            labels_words[(arr[0],item)] += 1
            total+=1
        #print arr[0],arr[1]
    #print  total
    f.close()
    return labels,labels_words,total
def model(labels,labels_words,total,text):
    words = list(text)

    temp = {}
    #print text
    for tag in labels.keys():
        temp[tag] = math.log(labels[tag]) -  math.log(total)
        for word in words:
            temp[tag] += math.log(labels_words.get((tag, word), 1.0)) - math.log(labels[tag])
    label=0
    grade = 0.0
    for t in labels.keys():
        cnt = 0.0
        for tt in labels.keys():
            cnt += math.exp(temp[tt] - temp[t])
        cnt = 1.0 / cnt
        if cnt > grade:
            label, grade = t, cnt
    return label, grade
def run(args):
    f=open(args.input_file)
    ff=open(args.output,'w')
    labels,labels_words,total=loaddata(args.corpus)
    for line in f.readlines():
        query=line.strip()
        label,grade=model(labels,labels_words,total,query)
        ff.write(query+'\t'+label+'\t'+str(grade)+'\n')
    f.close()
    ff.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--corpus",help="corpus")
    parser.add_argument("-i","--input_file",help="input_file")
    parser.add_argument("-o","--output",help="output")
    args = parser.parse_args()
    run(args)
