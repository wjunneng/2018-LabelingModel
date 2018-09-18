# coding: utf-8
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class tfidf:
    def __init__(self, path, savePath, topK):
        self.path = path
        self.savePath = savePath
        self.data = pd.read_excel(path)
        self.data[u'关键词'] = ''
        self.data[u'权重'] = ''
        self.topK = topK

    def titleTfidf(self):
        corpus = list(self.data[u'题目'])

        # 构建词频矩阵,将文本中的词语转化为次频矩阵
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        # 统计每个词的tf-idf
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)
        # 获取词袋模型中的关键字
        word = vectorizer.get_feature_names()
        # 获取tf-idf矩阵,a[i][j]为词j在文本i中的权重
        weight = tfidf.toarray()

        for i in range(len(weight)):
            # 每篇文本
            text = pd.DataFrame(index=word, data=weight[i], columns=[u'权重'])
            # 根据权重进行倒序排列
            text = text.sort_values(by=u'权重', ascending=False)
            # 题目分完词后的结果进行保存
            self.data.ix[i, u'关键词'] = " ".join(text.index.tolist()[0:self.topK])
            # 题目分完词后的权重结果进行保存
            self.data.ix[i, u'权重'] = " ".join([str(round(i, 3)) for i in text[u'权重'].tolist()[0:self.topK]])
        # 保存
        self.data.to_excel(self.savePath, index=None)

    def knowledgeTfidf(self):
        corpus = list(self.data[u'词语'])

        # 构建词频矩阵,将文本中的词语转化为次频矩阵
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        # 统计每个词的tf-idf
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)
        # 获取词袋模型中的关键字
        word = vectorizer.get_feature_names()
        # 获取tf-idf矩阵,a[i][j]为词j在文本i中的权重
        weight = tfidf.toarray()

        for i in range(len(weight)):
            # 每篇文本
            text = pd.DataFrame(index=word, data=weight[i], columns=[u'权重'])
            # 根据权重进行倒序排列
            text = text.sort_values(by=u'权重', ascending=False)
            # 题目分完词后的结果进行保存
            self.data.ix[i, u'关键词'] = " ".join(text.index.tolist()[0:self.topK])
            # 题目分完词后的权重结果进行保存
            self.data.ix[i, u'权重'] = " ".join([str(round(i, 3)) for i in text[u'权重'].tolist()[0:self.topK]])
        # 保存
        self.data.to_excel(self.savePath, index=None)


if __name__ == '__main__':
    # 题目
    # path = '../data/题目-类别.xls'
    # stopKeyPath = '../'
    # savePath = '../data/tfidfQuestionResult.xls'
    # topK = 10
    # tfidf(path, savePath, topK).titleTfidf()

    # 知识点
    path = '../data/知识点-类别.xls'
    stopKeyPath = '../'
    savePath = '../data/tfidfKnowledgeResult.xls'
    topK = 10
    tfidf(path, savePath, topK).knowledgeTfidf()
