# -*- coding: utf-8 -*-
"""
Author : videorighter
"""

from konlpy.tag import Komoran, Okt, Mecab, Kkma, Hannanum
import pandas as pd


def tokenize_sents(sents, tokenizer):
    """
    sent of list to tokenized sent of list
    :param sents: list of str
        sentence list
    :param tokenizer: callable
        tokenize(sent) returns list of str (word sequence)
    :return: list of list of str
        tokenized sentence list
    """
    return [tokenizer(sent) for sent in sents]


def komoran_tokenizer(sent):
    """
    :param sent: sentence; srt
    :return: list of words including Noun, Verb, Adjective, Stem
    """
    komoran = Komoran()
    words = komoran.pos(sent, join=True)
    print(sent)
    words = [word for word in words if ('/NN' in word or '/XR' in word or '/VA' in word or '/VV' in word)]
    return words


def mecab_tokenizer(sent):
    """
    :param sent: sentence; srt
    :return: list of words including Noun, Verb, Adjective, Stem
    """
    mecab = Mecab()
    words = mecab.pos(sent, join=True)
    words = [word for word in words if ('/NN' in word or '/XR' in word or '/VA' in word or '/VV' in word)]
    return words


if __name__ == "__main__":
    comment_all = pd.read_csv("/Users/oldman/PycharmProjects/nlp/data/glowpick_review.csv", encoding='UTF8')
    keyword_lst = pd.unique(comment_all.keyword)

    # 키워드 별 comment 개수
    num_dict = {}
    for keyword in keyword_lst:
        num_dict[keyword] = len(comment_all[comment_all['keyword'] == keyword])

    # 파운데이션 comment 수
    foundation = ['쿠션 파운데이션', '쿠션파운데이션', '리퀴드파운데이션', '스틱파운데이션', '크림파운데이션', '팩트파운데이션', '파우더파운데이션']
    fd_df = pd.concat([comment_all[comment_all['keyword'] == i] for i in foundation])
    print("glowpick foundation total comment length: ", len(fd_df))

    # 중복 제거
    fd_df_dp = fd_df.drop_duplicates(['cmt_body'])
    print("glowpick foundation duplicated comment length: ", len(fd_df_dp))

    # input
    sents = list(fd_df_dp.cmt_body)
    komoran_tokenizer(sents)
