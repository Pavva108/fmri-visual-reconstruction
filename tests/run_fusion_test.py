import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),'src'))
from tests.test_fusion import test_fusion_concat, test_fusion_gated, test_fusion_attention

if __name__ == '__main__':
    test_fusion_concat()
    test_fusion_gated()
    test_fusion_attention()
    print('FUSION TESTS PASSED')
