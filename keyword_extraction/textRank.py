# coding: utf-8
import pandas as pd
import jieba.analyse

class textRank(object):
    def __init__(self, datafile, topK):
        self.data = pd.read_csv(datafile)
        self.topK = topK

    def Main(self):
        idList, titleList, abstractList = self.data[u'id'], self.data[u'title'], self.data[u'abstract']
        ids, titles, abstracts = [], [], []
        for i in range(self.data.shape[0]):
            text = "%s %s" % (titleList[i], abstractList[i])
            # 加载停用词
            jieba.analyse.set_stop_words(u"./data/stopWord.txt")
            # TextRank关键词提取,词性筛选
            keyWords = jieba.analyse.textrank(text, topK=self.topK, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'))
            word_split = " ".join(keyWords)
            ids.append(idList[i])
            titles.append(titleList[i])
            abstracts.append(word_split)

        return pd.DataFrame({'id': ids, 'title': titles, 'key': abstracts}, columns=[u'id', u'title', u'key'])


if __name__ == '__main__':
    datafile = u'./data/sample_data.csv'
    result = textRank(datafile, topK=10).Main()
    result.to_csv("./result/keys_TextRank.csv", index=False)