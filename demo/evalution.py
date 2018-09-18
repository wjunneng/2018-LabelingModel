# coding: utf-8
import pandas as pd


class evalution(object):
    def __init__(self, questionPath, knowledgePath):
        self.question = pd.read_excel(questionPath)
        self.knowledge = pd.read_excel(knowledgePath)

    def Main(self):
        # 对于每个文本
        for i in self.question[u'权重'].values:
            print(i)


if __name__ == '__main__':
    # 题目路径
    questionPath = '../data/tfidfQuestionResult.xls'
    # 知识点路径
    knowledgePath = '../data/tfidfKnowledgeResult.xls'
    evalution(questionPath, knowledgePath).Main()