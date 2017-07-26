# coding: utf-8
#
# Copyright 2017 The Oppia Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for using winnowing as pre-processing strp."""


def hash_generator(token_to_id, tokens):
    """Generate hash for tokens in 'tokens' using 'token_to_id'.

    Args:
        token_to_id: dict. A dictionary which maps each token to a unique ID.
        tokens: list(str). A list of tokens.

    Returns:
        int. Hash value generated for tokens in 'tokens' using 'token_to_id'.
    """
    hash_val = 0
    n = len(tokens) - 1
    for x in tokens:
        hash_val += token_to_id[x] * (len(token_to_id) ** n)
        n -= 1
    return hash_val


def k_gram_hash_generator(tokens, token_to_id, K):
    """Generate all k-gram hashes for tokenized program.

    Args:
        tokens: list(str). A list of tokens.
        token_to_id: dict. A dictionary which maps each token to a unique ID.
        K: int. Model parameter 'K' of winnowing.

    Returns:
        list(int). k-gram hashes generated from 'tokens' using
        'token_to_id'.

    """
    generated_hashes = [
        hash_generator(token_to_id, tokens[i: i+K])
        for i in xrange(0, len(tokens) - K + 1)]
    return generated_hashes


def get_fingerprint_from_hashes(k_gram_hashes, window_size):
    """Generate document fingerprint from k-gram hashes of given program.

    Args:
        k_gram_hashes: list(int). k-gram hashes of program from which
            fingerprint is to be obtained.
        window_size: int. Size of the window from which fingerprints are to be
            obtained.

    Returns:
        list(int). Fingerprint obtained from k-gram hashes over 'window_size' of
        window.
    """
    generated_fingerprint = set()
    for i in xrange(0, len(k_gram_hashes) - window_size + 1):
        window_hashes = k_gram_hashes[i: i + window_size]
        min_hash_index = i + min(
            xrange(window_size), key=window_hashes.__getitem__)
        min_hash = k_gram_hashes[min_hash_index]
        generated_fingerprint.add((min_hash, min_hash_index))

    return list(generated_fingerprint)
