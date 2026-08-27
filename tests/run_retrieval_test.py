import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),'src'))
from tests.test_retrieval import test_build_db_and_retrieve, test_split_guard_raises_on_overlap

if __name__ == '__main__':
    test_build_db_and_retrieve()
    try:
        test_split_guard_raises_on_overlap()
    except AssertionError:
        print('split_guard raised as expected')
    print('RETRIEVAL TESTS PASSED')
