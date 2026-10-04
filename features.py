"""
Feature engineering for Quora Question Pairs duplicate detection.
Implements:
  - Basic length-based features
  - Token-based features (common words, stopwords, ratios)
  - Fuzzy features (using fuzzywuzzy ratios)
"""
import re
import distance
import numpy as np
from fuzzywuzzy import fuzz

STOP_WORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",
    'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers',
    'herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been',
    'being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if',
    'or','because','as','until','while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to','from','up','down','in','out',
    'on','off','over','under','again','further','then','once','here','there','when','where','why',
    'how','all','any','both','each','few','more','most','other','some','such','no','nor','not',
    'only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",'should',
    "should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't"
])

SAFE_DIV = 0.0001


def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
         .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
         .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
         .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
         .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
         .replace("%", " percent ").replace("$", " dollar ").replace("₹", " rupee ")\
         .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def get_token_features(q1, q2):
    """Token-based features: overlap of content words, stopwords, and raw tokens."""
    token_features = [0.0] * 10
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([w for w in q1_tokens if w not in STOP_WORDS])
    q2_words = set([w for w in q2_tokens if w not in STOP_WORDS])

    q1_stops = set([w for w in q1_tokens if w in STOP_WORDS])
    q2_stops = set([w for w in q2_tokens if w in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens)) / 2
    return token_features


def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0.0
    return len(strs[0]) / (min(len(a), len(b)) + 1)


def get_fuzzy_features(q1, q2):
    """Fuzzy string matching features from the fuzzywuzzy library."""
    fuzzy_features = [0.0] * 4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features


FEATURE_NAMES = [
    'q1_len', 'q2_len', 'q1_n_words', 'q2_n_words',
    'word_common', 'word_total', 'word_share',
    'cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max',
    'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len',
    'longest_substr_ratio',
    'fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio',
]


def extract_features(q1_raw, q2_raw):
    q1 = preprocess(q1_raw)
    q2 = preprocess(q2_raw)

    feats = []
    # --- Length-based features ---
    feats.append(len(q1))
    feats.append(len(q2))
    feats.append(len(q1.split()))
    feats.append(len(q2.split()))

    w1 = set(q1.split())
    w2 = set(q2.split())
    word_common = len(w1 & w2)
    word_total = len(w1) + len(w2)
    word_share = word_common / (word_total + SAFE_DIV)
    feats.append(word_common)
    feats.append(word_total)
    feats.append(word_share)

    # --- Token-based features ---
    feats.extend(get_token_features(q1, q2))

    # --- Longest common substring ratio (length-based) ---
    feats.append(get_longest_substr_ratio(q1, q2))

    # --- Fuzzy features ---
    feats.extend(get_fuzzy_features(q1, q2))

    return feats


def extract_features_batch(df, q1_col='question1', q2_col='question2'):
    rows = []
    for q1, q2 in zip(df[q1_col].fillna(""), df[q2_col].fillna("")):
        rows.append(extract_features(q1, q2))
    return np.array(rows)
