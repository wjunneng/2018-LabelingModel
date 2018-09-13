# coding: utf-8
import pandas as pd

class tfidf:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_excel(path)

    def tf(self):
        # 保存所有的词
        all_words_set = set()
        # 记录总词数
        n = 0
        # 保存所有文档的tf
        all_tfs_list = list()
        for i in range(self.data.shape[0]):
            # 每行的词语
            words_list = self.data.ix[i, u'题目'].strip().split(' ')
            # 每行不同的词语
            words_set = set(words_list)
            # 记录每行中（词，频数）
            tf_dict = dict.fromkeys(words_set, 0)
            # 每行的词数
            length = len(words_list)

            for i in words_set:
                tf_dict[i] = round(words_list.count(i) / length, 3)

            # 计算所有的词
            all_words = all_words | words_set
            # 计算总词数
            n += length
            # 计算所有文档的词频
            all_tfs_list.append(tf_dict)

            print(tf_dict)


if __name__ == '__main__':
    path = '../data/题目-类别.xls'

    tfidf(path).tf()
