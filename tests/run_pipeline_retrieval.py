import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),'src'))
from tests.test_pipeline_retrieval import test_pipeline_model_fusion_retrieval

if __name__ == '__main__':
    test_pipeline_model_fusion_retrieval()
    print('PIPELINE RETRIEVAL TEST PASSED')
