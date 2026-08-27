import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),'src'))
from tests.test_diffusion import test_dual_diffusion_forward_and_loss

if __name__ == '__main__':
    test_dual_diffusion_forward_and_loss()
    print('DIFFUSION TEST PASSED')
