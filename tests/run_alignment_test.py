import sys, os
import importlib.util

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

spec = importlib.util.spec_from_file_location('test_alignment', os.path.join(os.getcwd(), 'tests', 'test_alignment.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == '__main__':
    mod.test_aligners_api()
    print('ALIGNMENT TEST PASSED')
