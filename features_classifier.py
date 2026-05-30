"""
AI-likelihood classifier based on 7 features described by the user.

Usage:
- compute_features(text) -> dict of feature values
- fit_human_stats(human_texts, use_rttr=False) -> dict of mu (means) and sigma (stds) for each feature
- classify(text, stats, weights=None, ks=None, feature_sides=None, threshold=0.5, use_rttr=False)
    -> dict with per-feature scores, aggregated score, classification

Notes:
- If you don't have human-corpus stats, call fit_human_stats() with a list of human-written texts.
- feature_sides determines whether higher values indicate AI-likeness (+1), lower indicate AI-likeness (-1),
  or both directions (0, meaning "deviation from human norm" -> use absolute z-score).
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

# -----------------------
# Utilities: tokenization
# -----------------------
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_SPLIT_RE = re.compile(r'[.!?]+[\s"\']*')  # naive sentence splitter
_CONTRACTION_RE = re.compile(r"\b(?:[A-Za-z]+n't|[A-Za-z]+'[a-z]{1,3})\b", re.IGNORECASE)

# Minimal stopword list (English) - replace or extend as needed
_DEFAULT_STOPWORDS = {
    "the","a","an","in","on","and","or","but","if","then","so","is","are","was","were","be","to","of","for",
    "with","as","by","that","this","these","those","it","its","at","from","they","we","you","he","she","I","me","my",
    "your","our","their","not","have","has","had","do","does","did","which","what","when","where","who","whom"
}

# -----------------------
# Syllable counting (simple heuristic)
# -----------------------
def count_syllables(word: str) -> int:
    """
    Heuristic syllable counter:
    Counts vowel groups (aeiouy), with common adjustments.
    Not perfect but usually adequate for aggregate FRE calculations.
    """
    w = word.lower()
    if len(w) == 0:
        return 0
    # remove non-alpha tail
    w = re.sub(r'[^a-z]', '', w)
    if not w:
        return 0
    vowels = "aeiouy"
    groups = re.findall(r'[aeiouy]+', w)
    count = len(groups)
    # adjustments
    if w.endswith("e"):
        # silent e heuristic: don't subtract if word is just one vowel group
        if count > 1:
            count -= 1
    if count == 0:
        count = 1
    return count

# -----------------------
# Feature computation
# -----------------------
def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text)

def split_sentences(text: str) -> List[str]:
    # naive split that still keeps things if empty
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
    if not parts:
        # fallback: consider whole text as one sentence
        return [text.strip()] if text.strip() else []
    return parts

def compute_features(text: str, stopwords: set = None, use_rttr: bool = False) -> Dict[str, float]:
    """
    Compute the 7 features:
    f1: Lexical diversity (type-token ratio) or RTTR
    f2: Sentence length variance (sigma of words-per-sentence)
    f3: N-gram repetition score for n in 3..5 combined
    f4: Punctuation ratio (# punctuation marks / N_words)
    f5: Contraction ratio (# contractions / N_words)
    f6: Stopword ratio (# stopwords / N_words)
    f7: Flesch Reading Ease (FRE)
    """
    if stopwords is None:
        stopwords = _DEFAULT_STOPWORDS

    words = tokenize_words(text)
    N_words = len(words)
    sentences = split_sentences(text)
    N_sentences = max(1, len(sentences))  # avoid divide-by-zero; treat no-sentence as 1
    # 1. Lexical diversity
    unique_words = set(w.lower() for w in words)
    if use_rttr:
        f1 = len(unique_words) / math.sqrt(N_words) if N_words > 0 else 0.0
    else:
        f1 = len(unique_words) / N_words if N_words > 0 else 0.0

    # 2. Sentence length variance (sigma)
    L = []
    for s in sentences:
        w = tokenize_words(s)
        L.append(len(w))
    if len(L) == 0:
        mu = 0.0
        sigma = 0.0
    else:
        mu = sum(L) / len(L)
        sigma2 = sum((li - mu) ** 2 for li in L) / len(L)
        sigma = math.sqrt(sigma2)
    f2 = sigma

    # 3. N-gram repetition score (n=3..5)
    # compute counts for all n-grams (word-level)
    ngram_counts = Counter()
    for n in (3, 4, 5):
        if N_words >= n:
            for i in range(N_words - n + 1):
                ngram = tuple(words[i:i+n])
                ngram_counts[(n, ngram)] += 1
    # f3 = sum(max(0, c_j - 1)) / total_n_grams
    total_n_grams = sum(count for (n,gram),count in ngram_counts.items())
    rep_extra = sum(max(0, count - 1) for count in ngram_counts.values())
    f3 = rep_extra / total_n_grams if total_n_grams > 0 else 0.0

    # 4. Punctuation ratio (# punctuation marks / N_words)
    punctuation_marks = re.findall(r'[^\w\s]', text)  # every non-alnum/space char
    # But treat apostrophes inside words as not punctuation? already included above; keep simple.
    f4 = len(punctuation_marks) / N_words if N_words > 0 else 0.0

    # 5. Contraction ratio (# contractions / N_words)
    contractions = _CONTRACTION_RE.findall(text)
    f5 = len(contractions) / N_words if N_words > 0 else 0.0

    # 6. Stopword ratio (# stopwords / N_words)
    stopword_count = sum(1 for w in words if w.lower() in stopwords)
    f6 = stopword_count / N_words if N_words > 0 else 0.0

    # 7. Readability (Flesch Reading Ease)
    # FRE = 206.835 - 1.015*(N_words/N_sentences) - 84.6*(N_syllables/N_words)
    N_syllables = sum(count_syllables(w) for w in words)
    avg_words_per_sentence = N_words / N_sentences if N_sentences > 0 else N_words
    syllables_per_word = N_syllables / N_words if N_words > 0 else 0.0
    FRE = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * syllables_per_word
    f7 = FRE

    return {
        "f1_lex_div": f1,
        "f2_sent_len_sigma": f2,
        "f3_ngram_rep": f3,
        "f4_punct_ratio": f4,
        "f5_contraction_ratio": f5,
        "f6_stopword_ratio": f6,
        "f7_flesch": f7,
        # extras:
        "N_words": N_words,
        "N_sentences": N_sentences,
    }

# -----------------------
# Fit human-corpus stats
# -----------------------
def fit_human_stats(human_texts: List[str], stopwords: set = None, use_rttr: bool = False) -> Tuple[Dict[str,float], Dict[str,float]]:
    """
    Given a list of human-written texts, compute mean (mu) and std (sigma) for each f_i.
    Returns (mu_dict, sigma_dict)
    """
    if stopwords is None:
        stopwords = _DEFAULT_STOPWORDS
    feature_lists = defaultdict(list)
    for t in human_texts:
        feats = compute_features(t, stopwords=stopwords, use_rttr=use_rttr)
        # only keep the seven primary features
        for k in ["f1_lex_div","f2_sent_len_sigma","f3_ngram_rep","f4_punct_ratio","f5_contraction_ratio","f6_stopword_ratio","f7_flesch"]:
            feature_lists[k].append(feats[k])
    mu = {}
    sigma = {}
    for k, vals in feature_lists.items():
        if not vals:
            mu[k] = 0.0
            sigma[k] = 1.0
        else:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var) if var > 0 else 1.0
            mu[k] = mean
            sigma[k] = std
    return mu, sigma

# -----------------------
# Classification pipeline
# -----------------------
def safe_z(value: float, mu: float, sigma: float) -> float:
    if sigma == 0:
        return 0.0
    return (value - mu) / sigma

def logistic(x: float, k: float = 1.0) -> float:
    # logistic mapping to (0,1)
    return 1.0 / (1.0 + math.exp(-k * x))

def classify(
    text: str,
    stats: Dict[str, Dict[str, float]],
    weights: Dict[str, float] = None,
    ks: Dict[str, float] = None,
    feature_sides: Dict[str, int] = None,
    threshold: float = 0.5,
    stopwords: set = None,
    use_rttr: bool = False
) -> Dict:
    """
    stats: {"mu": {...}, "sigma": {...}} produced by fit_human_stats or provided manually
    feature_sides: mapping feature -> {+1, -1, 0}
      +1 => higher value indicates AI-likeness
      -1 => lower value indicates AI-likeness
       0 => deviation (both directions) indicates AI-likeness (we use abs(z))
    ks: steepness parameter per feature for logistic mapping (default 1.0)
    weights: per-feature aggregation weights (default uniform)
    """
    if stopwords is None:
        stopwords = _DEFAULT_STOPWORDS
    feats = compute_features(text, stopwords=stopwords, use_rttr=use_rttr)
    keys = ["f1_lex_div","f2_sent_len_sigma","f3_ngram_rep","f4_punct_ratio","f5_contraction_ratio","f6_stopword_ratio","f7_flesch"]
    # default sides based on description:
    default_feature_sides = {
        "f1_lex_div": -1,   # lower lexical diversity -> AI-like
        "f2_sent_len_sigma": -1, # lower variance -> AI-like
        "f3_ngram_rep": +1, # higher repetition -> AI-like
        "f4_punct_ratio": 0, # deviation (both directions) -> AI-like
        "f5_contraction_ratio": -1, # lower contractions -> AI-like
        "f6_stopword_ratio": 0, # deviation -> AI-like
        "f7_flesch": 0, # deviation from human FRE -> AI-like
    }
    if feature_sides is None:
        feature_sides = default_feature_sides
    # default weights: equal
    if weights is None:
        weights = {k: 1.0 for k in keys}
    # default ks
    if ks is None:
        ks = {k: 1.0 for k in keys}
    # ensure stats contain mu and sigma
    mu = stats.get("mu", {})
    sigma = stats.get("sigma", {})
    per_feature = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for k in keys:
        val = feats[k]
        muv = mu.get(k, 0.0)
        sigv = sigma.get(k, 1.0)
        z = safe_z(val, muv, sigv)
        side = feature_sides.get(k, default_feature_sides.get(k, 0))
        if side == 0:
            r = abs(z)
        else:
            r = side * z
        s = logistic(r, ks.get(k, 1.0))
        w = weights.get(k, 1.0)
        per_feature[k] = {
            "value": val,
            "mu": muv,
            "sigma": sigv,
            "z": z,
            "r": r,
            "s": s,
            "w": w,
            "contrib": w * s
        }
        weighted_sum += w * s
        weight_total += w
    final_score = weighted_sum / weight_total if weight_total != 0 else 0.0
    classification = "AI" if final_score >= threshold else "Human"
    out = {
        "features": per_feature,
        "final_score": final_score,
        "threshold": threshold,
        "classification": classification,
        "meta": {
            "N_words": feats.get("N_words"),
            "N_sentences": feats.get("N_sentences")
        }
    }
    return out

# -----------------------
# Example / demo
# -----------------------
if __name__ == "__main__":
    # Example human corpus (very small — replace with large human-written corpus for realistic stats)
    human_corpus = [
        "I went to the store earlier today to buy some milk and eggs. The cashier smiled and wished me a nice day.",
        "The quick brown fox jumps over the lazy dog. It was a sunny afternoon, and children were playing in the park.",
        "Please send the report by tomorrow. Let me know if you need any help preparing the slides.",
        "I'm excited about the trip — we've planned the itinerary carefully and booked all hotels.",
    ]
    # An example AI-like paragraph (toy)
    ai_text = ("This document outlines the objectives. The model generates coherent sentences that maintain topical cohesion. "
               "Each paragraph contains sentence sequences that demonstrate a consistent formal tone and neutral affect. "
               "Please refer to the sections below for methodology and results.")

    # Fit stats
    mu, sigma = fit_human_stats(human_corpus, use_rttr=False)
    stats = {"mu": mu, "sigma": sigma}

    print("Human-corpus stats (mu):")
    for k,v in mu.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassifying example texts...\n")
    res_human = classify(human_corpus[0], stats, threshold=0.5)
    res_ai = classify(ai_text, stats, threshold=0.5)
    print("Sample human text classification:", res_human["classification"], "score=", res_human["final_score"])
    print("Sample AI-like text classification:", res_ai["classification"], "score=", res_ai["final_score"])
    # print per-feature breakdown for the AI-like example
    print("\nPer-feature breakdown (AI-like sample):")
    for k, info in res_ai["features"].items():
        print(f"{k}: value={info['value']:.4f}, z={info['z']:.4f}, r={info['r']:.4f}, s={info['s']:.4f}, contrib={info['contrib']:.4f}")