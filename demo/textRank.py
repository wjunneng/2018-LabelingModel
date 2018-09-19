# coding: utf-8
import pandas as pd
import jieba.analyse
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


class textRank(object):
    def __init__(self, path, stopWordPath):
        self.data = pd.read_excel(path)
        self.stopWordPath = stopWordPath

    def Main(self):
        dataset = None
        for i in range(self.data.shape[0]):
            text = self.data.ix[i, u'题目']
            jieba.analyse.set_stop_words(self.stopWordPath)
            keywords_weights = jieba.analyse.textrank(text, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'),
                                                       withWeight=True)
            tmp = pd.DataFrame(data={"知识点编号": [str(self.data.ix[i, '知识点编号']).split(' ')[0]]})
            # 添加关键词和对应的权重
            for key, weight in keywords_weights:
                tmp[key] = weight
            if dataset is None:
                dataset = tmp.copy()
            else:
                dataset = pd.concat([dataset, tmp])

        # 对NaN数据进行填充0
        dataset = dataset.fillna(0)
        # 对数据进行打乱处理, 重置index
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        # 目标类别
        target = list(dataset[u'知识点编号'])
        # 删除知识点编号列
        del dataset[u'知识点编号']
        # rbfSVM
        classifier = OneVsOneClassifier(LinearSVC(random_state=0))
        # 这里的cross_val_score将交叉验证的整个过程连接起来，不用再进行手动的分割数据
        # cv参数用于规定将原始数据分成多少份
        scores = cross_val_score(classifier, dataset, target, cv=10, scoring='accuracy')
        # 计算平均准确率
        print(scores.mean())


if __name__ == '__main__':
    path = u'../data/题目-类别.xls'
    stopWordPath = u'../data/stopWord.txt'
    textRank(path, stopWordPath).Main()
