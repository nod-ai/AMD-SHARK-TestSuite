# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from e2e_testing.framework import Test

GLOBAL_TEST_LIST = []

_SEEN_NAMES = set()


def register_test(test_class: type, test_name: str):
    '''After defining a ModelInfo class "MyModelInfo", you can register it as a test with register_test(MyModelInfo, "my_test_name")'''
    # Ensure that there are no duplicate names in the global test registry.
    if test_name in _SEEN_NAMES:
        raise ValueError(
            f"Duplicate test name: '{test_name}'. Please make sure that the function wrapped by `register_test` has a unique name."
        )
    _SEEN_NAMES.add(test_name)

    # Store the test in the registry.
    GLOBAL_TEST_LIST.append(
        Test(
            unique_name=test_name,
            model_constructor=test_class,
        )
    )

# a class decorator alternative to register_test for convenience
def register_with_name(name):
    '''Use @register_with_name("my_test_name") before defining a ModelInfo class to add that ModelInfo as a test.'''
    return lambda test_class : register_test(test_class, name)
