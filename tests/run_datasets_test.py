import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),'src'))
from tests.test_datasets import test_synthetic_dataset, test_factory_synthetic_config

if __name__ == '__main__':
    test_synthetic_dataset()
    test_factory_synthetic_config()
    print('DATASETS TESTS PASSED')
