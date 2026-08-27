import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),'src'))
from tests.test_models import test_transformer_projectors_forward

if __name__ == '__main__':
    test_transformer_projectors_forward()
    print('MODEL TEST PASSED')
