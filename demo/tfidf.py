# coding: utf-8
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.svm as svm

class tfidf:
    def __init__(self, path, savePath, topK):
        self.path = path
        self.savePath = savePath
        self.data = pd.read_excel(path)
        self.data[u'关键词'] = ''
        self.data[u'权重'] = ''
        self.topK = topK

    def Accuracy(self, classifier, X, Y):
        print(X, len(Y))
        # 返回准确率
        result = classifier.predict(X)
        print(result)
        num = len(X)
        num_error = 0
        for i in range(0, num):
            if result[i] != Y[i]:
                num_error += 1

        print("总测试个数：" + str(num))
        print("错误个数: " + str(num_error))
        print("正确率：" + "  " + str((num - num_error) / num))

        return (num - num_error) / num

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
        # 训练集数据数目
        trainTextNumber = int(self.data.shape[0] * 0.6)
        # 权重列
        weights = []
        for i in self.data[u'权重'].tolist():
            weights.append([float(j) for j in i.split(' ')])

        # 知识点列
        categorys = [int(str(i).split(' ')[0]) for i in self.data[u'知识点编号'].tolist()]
        # 训练集X
        trainX = weights[0:trainTextNumber]
        # 训练集类别Y
        trainY = categorys[0:trainTextNumber]
        # 测试集X
        testX = weights[trainTextNumber:]
        # 测试集类别Y
        testY = categorys[trainTextNumber:]
        # rbfSVM
        classifier = svm.SVC(kernel='rbf')
        # 训练分类器
        classifier.fit(trainX, trainY)
        # 返回准确率
        tmp = self.Accuracy(classifier, testX, testY)


if __name__ == '__main__':
    # 题目
    path = '../data/题目-类别.xls'
    stopKeyPath = '../'
    savePath = '../data/tfidfQuestionResult.xls'
    topK = 10
    tfidf(path, savePath, topK).titleTfidf()
