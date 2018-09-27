# coding: utf-8

import os
import jieba
import jieba.posseg
import pandas as pd
import numpy as np
import multiprocessing
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


class word2Vec(object):
    def __init__(self, path, stopWordsPath, questionPath):
        self.path = path
        self.completePath = self.path + "/complete.txt"
        self.ParticiplePath = self.path + "/participle.txt"
        self.stopWords = pd.read_excel(stopWordsPath)
        self.outp1 = self.path + '/wiki.zh.text.model'
        self.outp2 = self.path + '/wiki.zh.text.vector'
        self.question = pd.read_excel(questionPath)[u'题目'].tolist()
        self.category = pd.read_excel(questionPath)[u'知识点编号'].tolist()

    def Merge(self):
        """实现目录下多个txt文件合并为一个txt文件"""
        txt = ''
        # 打印段落数
        for filename in os.listdir(self.path):
            if filename.endswith('.txt'):
                file = open(path + "/" + filename, 'r', encoding='utf-8')
                txt += " ".join([str(i) for i in file.readlines()])
        # 合并文件
        with open(self.completePath, 'w') as file:
            file.write(txt)

    def Participle(self):
        """实现分词"""
        text = ""

        with open(self.completePath, 'r', encoding='utf-8') as file:
            tmp = self.question
            tmp.extend(file.readlines())

            for line in tmp:
                # 定义选取的词性
                pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
                index = 0
                for word in jieba.posseg.cut(str(line)):
                    if len(word.word) > 1 and word.flag in pos and word.word not in self.stopWords.value.tolist():
                        text += " " + word.word
                        index = 1
                if index == 1:
                    text += '\n'

        with open(self.ParticiplePath, "w") as file:
            file.write(text)

    def word2Vector(self):
        """训练词向量模型"""
        # 训练skip-gram模型
        model = Word2Vec(LineSentence(self.ParticiplePath), size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())
        # 保存模型
        model.save(self.outp1)
        model.wv.save_word2vec_format(self.outp2, binary=False)

    def calMean(self):
        """计算平均值"""
        count = 0
        for i in self.question:
            count += len(i.split(" "))
        # 返回平均值
        return int(count / len(self.question))

    def Main(self):
        # 加载word2Vec模型
        model = Word2Vec.load(self.outp1)

        result = []
        for i in self.question:
            length = self.calMean()
            text = i.split(" ")
            if length <= len(text):
                text = text[0:length]
            else:
                text.extend(["you"]*(length - len(text)))

            tmp = np.array([])
            for word in text:
                if word in model:
                    tmp = np.append(tmp, model[word])
                else:
                    tmp = np.append(tmp, np.array([float(0)]*200))

            result.append(tmp)

        dataset = pd.DataFrame()
        dataset[u"词向量"] = result
        dataset[u"知识点编号"] = self.category
        # 对数据进行打乱处理, 重置index
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        # 目标类别
        target = list(dataset[u'知识点编号'])
        # 删除知识点编号列
        del dataset[u'知识点编号']
        dataset = pd.DataFrame(dataset[u'词向量'].tolist())
        # rbfSVM
        classifier = OneVsOneClassifier(LinearSVC(random_state=0))
        # 这里的cross_val_score将交叉验证的整个过程连接起来，不用再进行手动的分割数据
        # cv参数用于规定将原始数据分成多少份
        scores = cross_val_score(classifier, dataset, target, cv=5, scoring='accuracy')
        # 计算平均准确率
        return scores.mean()


if __name__ == '__main__':
    path = u"/".join(os.getcwd().split('/')[:-1])+"/data/语料库-材料"
    questionPath = u"../data/tfidfQuestionResult.xls"
    stopWordsPath = u"/".join(os.getcwd().split('/')[:-1])+"/data/stopWord.xlsx"
    wc = word2Vec(path, stopWordsPath, questionPath)
    # 合并txt文件
    # wc.Merge()
    # 分词
    # wc.Participle()
    # 训练词向量模型
    # wc.word2Vector()
    # print(wc.Main())
