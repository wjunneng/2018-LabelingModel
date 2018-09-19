# coding: utf-8
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

class tfidf:
    def __init__(self, path, savePath):
        self.path = path
        self.savePath = savePath
        self.data = pd.read_excel(path)
        self.data[u'关键词'] = ''
        self.data[u'权重'] = ''

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
            # 题目分完词后的权重结果进行保存
            self.data.ix[i, u'权重'] = " ".join([str(round(i, 3)) for i in text[u'权重'].tolist()])

        # 打乱数据
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        # 权重列
        weights = []
        for i in self.data[u'权重'].tolist():
            tmp = [float(j) for j in i.split(' ')]
            weights.append([i/sum(tmp) for i in tmp])

        # 知识点列
        categorys = [int(str(i).split(' ')[0]) for i in self.data[u'知识点编号'].tolist()]
        # SVC
        classifier = OneVsOneClassifier(LinearSVC(random_state=0))
        # 这里的cross_val_score将交叉验证的整个过程连接起来，不用再进行手动的分割数据
        # cv参数用于规定将原始数据分成多少份
        scores = cross_val_score(classifier, weights, categorys, cv=10, scoring='accuracy')
        # 计算平均准确率
        print(scores.mean())


if __name__ == '__main__':
    # 题目
    path = '../data/题目-类别.xls'
    stopKeyPath = '../'
    savePath = '../data/tfidfQuestionResult.xls'
    tfidf(path, savePath).titleTfidf()
