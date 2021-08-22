# -*- coding: utf-8 -*-
"""
Author : videorighter
"""

from collections import defaultdict

from textrank.tokenizers import komoran_tokenizer, tokenize_sents
from textrank.utils import vocab_indexing, dict2mat


def cooccurrence(tokens, vocab2idx, window=2, min_cooccurrence=2, verbose=True):
    """
    단어 간 / 문장 간 유사도 정의하기 위한 co-occurrence
    :param tokens: list of list of str
        list of (word+pos) lists
    :param vocab2idx: dict
        word-index dictionary
    :param window: int
        interval between a and b
    :param min_cooccurrence: int
        minimum co-ocurrence frequency
    :param verbose: bool
        if True, verbose mode
    :return dict2mat(counter, n_vocabs, n_vocabs): csr_matrix
        (n_vocabs, n_vocabs)
    """
    counter = defaultdict(int)
    for s, token_idx in enumerate(tokens):
        if verbose and s % 1000 == 0:
            print(f'\rword cooccurrence counting {s+1}', end='')
        vocabs = [vocab2idx[word] for word in token_idx if word in vocab2idx]
        num_word = len(vocabs)
        for i, vocab_idx in enumerate(vocabs):
            if window <= 0:  # -1이라면 모든 단어 고려
                start, end = 0, num_word
            else:
                start = max(0, i - window)
                end = min(i + window, num_word)
            for j in range(start, end):
                if i == j:
                    continue
                counter[(vocab_idx, vocabs[j])] += 1
                counter[(vocabs[j], vocab_idx)] += 1
    counter = {key: value for key, value in counter.items() if value >= min_cooccurrence}  # 최소 동시발생횟수 이상만
    n_vocabs = len(vocab2idx)
    if verbose:
        print(f'\rword cooccurrence counting from {len(tokens)+1} sents was done.')
    return dict2mat(counter, n_vocabs, n_vocabs)


def word_graph(sents, tokenizer=None, min_count=2, window=2, min_cooccurrence=2, vocab2idx=None, verbose=True):
    """
    make word graph
    :param vocab_to_idx: dict
        Vocabulary to index mapper.
        If None, this function scan vocabulary first.
    :param sents: list of str
        sentence list
    :param tokenizer: callable
        tokenize(str); list of str
    :param min_count: int
        minimum term frequency
    :param window: int
        Co-occurrence window size
    :param min_cooccurrence: int
    :param verbose: boolean
        if True, verbose mode
    :return:
    """
    if vocab2idx is None:
        idx2vocab, vocab2idx = vocab_indexing(sents, tokenizer, min_count)
    else:
        idx2vocab = [vocab for vocab, _ in sorted(vocab2idx.items(), key=lambda x: x[1])]
    tokens = tokenize_sents(sents, tokenizer)
    g = cooccurrence(tokens, vocab2idx, window, min_cooccurrence, verbose)
    return g, idx2vocab


if __name__ == "__main__":
    sents = ['안녕하세요 반가워요 헤헤헤.', '오, 안녕하세요 저도 반가워요 호호호.', '오늘 날씨가 참 좋네요.', '카드캡터 체리가 좋아하는 음식은?', '게찜이~ 겠지유~']
    min_count = 1
    idx2vocab, vocab2idx = vocab_indexing(sents, komoran_tokenizer, min_count)

    tokens = [komoran_tokenizer(sent) for sent in sents]

    window = 2
    min_cooccurrence = 2
    dict2mat = cooccurrence(tokens=tokens,
                            vocab2idx=vocab2idx,
                            window=window,
                            min_cooccurrence=min_cooccurrence,
                            verbose=True)