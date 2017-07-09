# coding=utf-8
"""util for data processing"""
import re
import codecs
import string
import random
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from utils.cache import RandomSet
from enum import Enum, unique
import numpy as np
from itertools import chain
from config.config import end_token

wn_lemmatizer = WordNetLemmatizer()

_WORD_SPLIT = re.compile(b"([.,!?\"';-@#)(])")
_DIGIT_RE = re.compile(br"\d")


@unique
class aksis_data_label(Enum):
    negative_label = 0
    positive_label = 1


def sentence_gen(files):
    """Generator that yield each sentence in a line.
    Parameters
    ----------
        files: list
            data file list
    """
    if not isinstance(files, list):
        files = [files]
    for filename in files:
        with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip().lower()
                if len(line):
                    yield line


def stem_tokens(tokens, lemmatizer):
    """lemmatizer
    Parameters
    ----------
        tokens: list
            token for lemmatizer
        lemmatizer: stemming model
            default model is wordnet lemmatizer
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def tokenize(text, lemmatizer=wn_lemmatizer):
    """tokenize and lemmatize the text"""
    text = clean_html(text)
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, lemmatizer)
    return stems


def extract_raw_query_title_score_from_aksis_data(sentence):
    """extract the query, title and score from aksis raw data, this function is to generate training data
    score gives a rough idea about specificness of a query. For example query1: "iphone" and query2: "iphone 6s 64GB".
    In both the query customer is looking for iphone but query2 is more specific.
    Query specificity score is number which ranges from 0.0 to 1.0.
    Aksis data format: MarketplaceId\tAsin\tKeyword\t Score\tActionType\tDate
    ActioType: 1-KeywordsByAdds, 2-KeywordsBySearches, 3-KeywordsByPurchases, 4-KeywordsByClicks
    """
    sentence = sentence.strip().lower()
    items = re.split(r'\t+', sentence)
    if len(items) == 7 and len(items[2]) and len(items[3]) and len(items[6]):
        return items[2], items[6], items[3]
    else:
        return None, None, None


def query_title_score_generator_from_aksis_data(files, dropout=-1):
    """Generator that yield query, title, score in aksis corpus"""
    for line in sentence_gen(files):
        query, title, score = extract_raw_query_title_score_from_aksis_data(line)
        if query and title and score:
            if not is_hit(score, dropout):
                continue
            yield query, title


def negative_sampling_train_data_generator(files, neg_number, dropout=-1):
    rs = RandomSet()
    for query, title in query_title_score_generator_from_aksis_data(files, dropout):
        rs.add(title)
        yield query, title, aksis_data_label.positive_label.value
        for neg_title in rs.get_n_items(neg_number):
            yield query, neg_title, aksis_data_label.negative_label.value


def is_hit(score, dropout):
    """sample function to decide whether the data should be trained,
    not sample if dropout less than 0"""
    return dropout < 0 or float(score) > random.uniform(dropout, 1)


def clean_html(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    Parameters
    ----------
        html: str
            the HTML string to be cleaned
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    return cleaned.strip()


def text_normalize(rawstr):
    tnstring = rawstr.lower()
    tnstring = re.sub("[^a-z0-9':#,$-]", " ", tnstring)
    tnstring = re.sub("\\s+", " ", tnstring).strip()
    return tnstring


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    sentence_normed = text_normalize(sentence)
    # sentence_normed = sentence.lower()
    for space_separated_fragment in sentence_normed.split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def trigram_encoding(data, trigram_dict, max_len):
    if not data:
        return None, []
    data = re.sub('[^a-z0-9.&\\ ]+', '', data.lower())
    data_seq = data.split()
    if len(data_seq) <= 2:  # remove query that are shorter than 3 words
        # raise Exception("trigram_encoding: the length of data should larger than 2 words, error data: %s" % data)
        return None, []

    data_triagrams = list(chain(*[find_ngrams("#" + qw + "#", 3) for qw in data_seq]))
    data_triagrams_index = [trigram_dict[d] if d in trigram_dict else len(trigram_dict) + 1 for d in data_triagrams]
    data_triagrams_index = data_triagrams_index[0:max_len]
    return data, data_triagrams_index


def trigram_sentence_to_padding_index(sentence, trigram_dict, maxlen):
    try:
        index = trigram_encoding(sentence, trigram_dict)
    except Exception as e:
        return 0, []
    original_len = min(len(index), maxlen)
    index = pad_sequences(np.array([index]), padding='post', truncating='post', maxlen=maxlen)
    return original_len, index[0]


# batch preparation of a given sequence pair for training
def prepare_train_batch(seqs_x, seqs_y, maxlen=None):
    # seqs_x, seqs_y: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x <= maxlen and l_y <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    batch_size = len(seqs_x)

    x_lengths = np.array(lengths_x)
    y_lengths = np.array(lengths_y)

    maxlen_x = np.max(x_lengths)
    maxlen_y = np.max(y_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * end_token
    y = np.ones((batch_size, maxlen_y)).astype('int32') * end_token

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[idx, :lengths_x[idx]] = s_x
        y[idx, :lengths_y[idx]] = s_y
    return x, x_lengths, y, y_lengths


def prepare_decode_batch(seqs_x, maxlen=None):
    # seqs_x, seqs_y: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x):
            if l_x <= maxlen and l_y <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    batch_size = len(seqs_x)

    x_lengths = np.array(lengths_x)

    maxlen_x = np.max(x_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * end_token

    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, x_lengths