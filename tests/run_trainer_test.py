import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),'src'))
from tests.test_trainer import test_trainer_one_step_and_checkpoint

if __name__ == '__main__':
    test_trainer_one_step_and_checkpoint()
    print('TRAINER TEST PASSED')
