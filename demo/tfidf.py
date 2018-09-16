# coding: utf-8
import pandas as pd
import math
import collections


class tfidf:
    def __init__(self, path, savePath):
        self.path = path
        self.savePath = savePath
        self.data = pd.read_excel(path)
        self.data[u'关键词'] = ''

    def tf(self):
        # 保存所有的词(不存在重复)
        all_words_set = set()
        # 保存不同的词(存在重复)
        all_words_list = list()
        # 记录总词数
        n = 0
        # 记录总文本数
        n_text = self.data.shape[0]
        # 保存所有文档的tf
        all_tfs_list = list()

        for i in range(n_text):
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

            # 计算所有的词(不重复)
            all_words_set = all_words_set | words_set
            # 计算所有的词(存在重复)
            all_words_list.extend(all_words_list)
            # 计算总词数
            n += length
            # 计算所有文档的词频
            all_tfs_list.append(tf_dict)

        # 保存所有词的idf
        all_idf_dict = dict.fromkeys(all_words_set, 0)
        for i in all_words_set:
            # 该单词在文本中出现的次数
            count = 0
            for j in all_tfs_list:
                if i in j.keys():
                    count += 1
            all_idf_dict[i] = math.log(n_text / (count + 1))

        tf_idf_list = list()
        for i in all_tfs_list:
            # 记录每行中（词，tf_idf）
            ti_dict = dict.fromkeys(i.keys(), 0)
            for key, value in i.items():
                ti_dict[key] = round(value * all_idf_dict[key], 3)

            ti_dict = collections.OrderedDict(sorted(ti_dict.items(), key=lambda t: t[1], reverse=True))
            tf_idf_list.append(ti_dict)

        for i in range(n_text):
            k = len(tf_idf_list[i])
            # topN
            if 10 < k:
                k = 10
                # 截取前tmp个关键词
            tmp = []
            for j in range(k):
                tmp.append(list(tf_idf_list[i].keys())[j])

            self.data.ix[i, u'关键词'] = ' '.join([i for i in tmp])

        # 保存
        self.data.to_excel(self.savePath, index=None)


if __name__ == '__main__':
    path = '../data/题目-类别.xls'
    savePath = '../data/tfidfResult.xls'

    tfidf(path, savePath).tf()
