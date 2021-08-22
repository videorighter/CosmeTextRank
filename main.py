# -*- coding: utf-8 -*-

import pandas as pd
from textrank import summarizer
from textrank.tokenizers import tokenize_sents, komoran_tokenizer
import csv
import time
import pprint


def main():

    comment_list = []
    with open("/Users/oldman/PycharmProjects/textrank/data/glowpick_review.csv", "rt", encoding='cp949') as f:
        comments = csv.reader(f, delimiter=',')
        for i in comments:
            comment_list.append(i)
    comment_all = pd.DataFrame(comment_list[1:], columns=comment_list[0])
    comment_all.set_index('idx', inplace=True)
    keyword_lst = pd.unique(comment_all.keyword)

    # 키워드 별 comment 개수
    num_dict = {}
    for keyword in keyword_lst:
        num_dict[keyword] = len(comment_all[comment_all['keyword'] == keyword])

    # 파운데이션 comment 수
    # '쿠션 파운데이션', '쿠션파운데이션', '리퀴드파운데이션', '스틱파운데이션', '크림파운데이션', '팩트파운데이션', '파우더파운데이션'
    foundation = ['립라커']
    fd_df = pd.concat([comment_all[comment_all['keyword'] == i] for i in foundation])
    print("glowpick foundation total comment length: ", len(fd_df))

    # 중복 제거
    fd_df_dp = fd_df.drop_duplicates(['cmt_body'])
    print("glowpick foundation duplicated comment length: ", len(fd_df_dp))

    start = time.time()
    keyword_extractor = summarizer.KeywordSummarizer(tokenizer=komoran_tokenizer, window=-1, verbose=True)
    print(keyword_extractor)
    keyword = keyword_extractor.summarize(list(fd_df_dp.cmt_body), top_keywords=30)
    pprint.pprint(keyword)
    print("total time: ", time.time() - start)


if __name__ == "__main__":
    main()