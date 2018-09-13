# coding: utf-8
import pandas as pd
import xlrd
import jieba

class DealData:
    def __init__(self, path, savePath, stopWordsPath):
        self.path = path
        self.savePath = savePath
        self.stopWordsPath = stopWordsPath

    # 读取停用词文件
    def readStopWords(self):
        data = pd.read_excel(self.stopWordsPath)

        return data.value.tolist()

    # 题目读取
    def dealTitle(self):
        # 读取文件
        wb = xlrd.open_workbook(self.path)
        # 获取所有的sheet名字
        sheets = wb.sheet_names()
        # 遍历
        data = None
        for i in sheets:
            df = pd.read_excel(self.path, sheet_name=i, encoding='uttf-8')
            if data is not None:
                data = pd.concat([data, df], axis=0)
            else:
                data = df

        data.reset_index(inplace=True)

        # 加载停用词
        stopWords = self.readStopWords()
        for i in range(data.shape[0]):
            txt = ''

            # 加空格　
            tmp = data.ix[i, u'题干'] + " " + data.ix[i, u'A'] + " " + data.ix[i, u'B'] + " " + data.ix[i, u'C'] + " " + \
                  data.ix[i, u'D']
            # 结巴分词
            for j in jieba.cut(tmp):
                print(j)
                if j not in stopWords and len(j) > 1:
                    txt += j + " "

            data.ix[i, u'题目'] = txt

        data = data.loc[:, [u'题目', u'知识点编号']]
        data.to_excel(self.savePath, index=None)

    # 知识点文件处理
    def DealKnolwledge(self):
        data = pd.read_excel(self.path)

        # 加载停用词
        stopWords = self.readStopWords()
        for i in range(data.shape[0]):
            txt = ''
            for word in jieba.cut(data.ix[i, u'知识点']):
                if word not in stopWords and len(word) > 1:
                    txt += word + ' '
            data.ix[i, u'词语'] = txt

        data = data.loc[:, [u'类别', u'词语']]

        data.to_excel(self.savePath, index=None)


if __name__ == '__main__':
    # path = "../data/大信-题目-全部.xls"
    # savePath = "../data/题目-类别.xls"
    # stopWordsPath = "../data/stopWord.xlsx"
    # DealData(path, savePath, stopWordsPath).DealKnolwledge()

    path = "../data/一级（信息技术基础）-Revision-知识点.xlsx"
    savePath = "../data/知识点-类别.xlsx"
    stopWordsPath = "../data/stopWord.xlsx"
    DealData(path, savePath, stopWordsPath).DealKnolwledge()