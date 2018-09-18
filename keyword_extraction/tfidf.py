# coding: utf-8
"""采用TF-IDF"""
import sys, codecs
import numpy as np
import pandas as pd
import jieba.posseg
import jieba.analyse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

"""
    TF-IDF:
    1、CountVectorizer 构建词频矩阵
    2、TfidfTransformer构建tf-idf权值计算
    3、文本的关键字
    4、对应的tfidf矩阵
"""
"""
（1）读取样本源文件sample_data.csv;

（2）获取每行记录的标题和摘要字段，并拼接这两个字段；

（3）加载自定义停用词表stopWord.txt，并对拼接的文本进行数据预处理操作，包括分词、筛选出符合词性的词语、去停用词，用空格分隔拼接成文本;

（4）遍历文本记录，将预处理完成的文本放入文档集corpus中；

（5）使用CountVectorizer()函数得到词频矩阵，a[j][i]表示第j个词在第i篇文档中的词频；

（6）使用TfidfTransformer()函数计算每个词的tf-idf权值；

（7）得到词袋模型中的关键词以及对应的tf-idf矩阵；

（8）遍历tf-idf矩阵，打印每篇文档的词汇以及对应的权重；

（9）对每篇文档，按照词语权重值降序排列，选取排名前topN个词最为文本关键词，并写入数据框中；

（10）将最终结果写入文件keys_TFIDF.csv中。

"""


# 数据预处理操作：分词、去停用词、词性筛选
def dataPrepos(text, stopkey):
    result = []
    # 定义选取的词性
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
    # 分词
    wordsList = jieba.posseg.cut(text)
    # 去停用词和词性筛选
    for i in wordsList:
        if i.word not in stopkey and i.flag in pos:
            result.append(i.word)
    # 返回结果
    return result


# tf-idf获取文本top10关键词
def getKeyWords_tfidf(data, stopkey, topK):
    idList, titleList, abstractList = data[u'id'], data[u'title'], data[u'abstract']
    corpus = []
    for i in range(len(idList)):
        # 拼接标题和摘要
        text = '%s %s' % (titleList[i], abstractList[i])
        text = dataPrepos(text, stopkey)
        text = " ".join(text)
        corpus.append(text)

    # 构建词频矩阵，将文本中的词语转化为词频矩阵
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    # 统计每个词的tf-idf值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # 获取词袋模型中的关键字
    word = vectorizer.get_feature_names()
    # 获取tf-idf矩阵，a[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()
    # 打印词语权重
    ids, titles, keys = [], [], []
    for i in range(len(weight)):
        print(u'--这里输出第', i + 1, u'篇文本的词语tf-idf--')
        ids.append(idList[i])
        titles.append(titleList[i])
        df_word, df_weight = [], []  # 当前文章的所有词汇列表，词汇对应权重列表
        for j in range(len(word)):
            df_word.append(word[j])
            df_weight.append(weight[i][j])

        df_word = pd.DataFrame(df_word, columns=['word'])
        df_weight = pd.DataFrame(df_weight, columns=['weight'])

        # 拼接词汇列表和权重列表
        word_weight = pd.concat([df_word, df_weight], axis=1)
        # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by='weight', ascending=False)

        # 选择词汇列并转成数组形式
        keyword = np.array(word_weight['word'])
        # 抽取前topK个词汇作为关键词
        word_split = [keyword[k] for k in range(0, topK)]
        word_split = " ".join(word_split)
        keys.append(word_split)
    result = pd.DataFrame({"id": ids, "title": titles, "key": keys}, columns=['id', 'title', 'key'])

    return result


# 主函数
def main():
    # 读取数据集
    dataFile = "./data/sample_data.csv"
    data = pd.read_csv(dataFile)
    # 停用词表
    stopkey = [w.strip() for w in codecs.open('./data/stopWord.txt', 'r', 'utf-8').readlines()]
    # tf-idf关键词提取
    result = getKeyWords_tfidf(data, stopkey, 10)
    result.to_csv("./result/keys_tfidf.csv", encoding='utf-8', index=False)


if __name__ == '__main__':
    main()