# coding: utf-8
import pandas as pd

class textRank(object):
    def __init__(self, path, stopWordPath, savePath):
        self.data = pd.read_excel(path)
        self.stopWordPath = stopWordPath
        self.savePath = savePath

    def Main(self):



if __name__ == '__main__':
    path = './data/题目-类别.xls'
    stopWordPath = u'./data/stopWord.txt'
    savePath = u'./data/textRankQuestionResult.xls'
    textRank(path, stopWordPath, savePath).Main()
