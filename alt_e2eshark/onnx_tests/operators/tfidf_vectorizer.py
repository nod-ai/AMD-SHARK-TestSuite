# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Dict, Any

from onnx import TensorProto
from onnx.helper import make_node, make_tensor_value_info

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test

TEST_PARAMS = {
    "only_bigram": {
        "n": 2,
        "min_gram_length": 2,
        "max_gram_length": 2,
        "max_skip_count": 1,
        "ngram_counts": [0, 1],
        "pool_int64s": [1, 2, 3, 5, 4, 5, 6, 7, 8, 6, 7],
        "ngram_indexes": [0, 1, 2, 3, 4, 5, 6],
        "weights": [0.5, 0.0, 1.0, 1.0, 7.0, 2.0, 0.0],
    },
    "unigram_bigram": {
        "n": 4,
        "min_gram_length": 1,
        "max_gram_length": 2,
        "max_skip_count": 0,
        "ngram_counts":[0,4],
        "pool_int64s": [2, 3, 5, 4, 5, 6, 7, 8, 6, 7],
        "ngram_indexes": [0, 2, 4, 5, 6, 7, 8, 9],
        "weights": [0.491, 0.15, 0.1, 1.0, 1.0, 7.0, 2.0, 0.0],
    },
    "unigram_bigram_skip5": {
        "n": 3,
        "min_gram_length": 1,
        "max_gram_length": 2,
        "max_skip_count": 5,
        "ngram_counts":[0,4],
        "pool_int64s": [2, 3, 5, 4, 5, 6, 7, 8, 6, 7],
        "ngram_indexes": [0, 2, 4, 5, 6, 7, 8, 9],
        "weights": [0.0, 0.15, 0.1, 1.0, 1.0, 7.0, 2.0, 0.0],
    },
    "unigram_bigram_no_weights": {
        "n": 4,
        "min_gram_length": 1,
        "max_gram_length": 2,
        "max_skip_count": 2,
        "ngram_counts":[0,2],
        "pool_int64s": [2, 3, 5, 4, 5, 6, 7, 8, 6, 7],
        "ngram_indexes": [0, 2, 4, 5, 6, 7, 8, 9],
        "weights": None,
    },
}

def create_vectorizer_node(
        mode: str,
        params: Dict[str, Any]
    ):
    return make_node(
        op_type="TfIdfVectorizer",
        inputs=["X"],
        outputs=["Y"],
        mode=mode,
        min_gram_length=params["min_gram_length"],
        max_gram_length=params["max_gram_length"],
        max_skip_count=params["max_skip_count"],
        ngram_counts=params["ngram_counts"],
        ngram_indexes=params["ngram_indexes"],
        pool_int64s=params["pool_int64s"],
        weights=params["weights"],
    )

def create_test_case(test_case, mode):
    class TFIDFModel(BuildAModel):
        def construct_i_o_value_info(self):
            # Input (ValueInfoProto)
            params = TEST_PARAMS[test_key]
            X = make_tensor_value_info(
                "X",
                TensorProto.INT32,
                shape=[params["n"], max(params["ngram_indexes"])]
            )
            # Output
            Y = make_tensor_value_info(
                "Y",
                TensorProto.FLOAT,
                shape=[params["n"], max(params["ngram_indexes"]) + 1])
            self.input_vi = [X]
            self.output_vi = [Y]

        def construct_nodes(self):
            params = TEST_PARAMS[test_key]
            vectorizer_node = create_vectorizer_node(
                mode=mode,
                params=params,
            )
            self.node_list = [vectorizer_node]

    return TFIDFModel

for mode in ["IDF", "TFIDF"]:
    for test_key in TEST_PARAMS.keys():
        test_case = create_test_case(test_key, mode)
        register_test(test_case, f'tfidfvectorizer_{mode.lower()}_{test_key}')
