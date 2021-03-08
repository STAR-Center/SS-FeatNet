import sys
import numpy as np
sys.path.append('../..')
from data.datagenerator import DataGenerator

if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print('Usage: python thisfile.py desc.bin desc.txt')
        sys.exit(1)

    pc_mat = DataGenerator.load_point_cloud(sys.argv[1], 3+32)

    np.savetxt(sys.argv[2], pc_mat[:,3:], delimiter=',', encoding=None)







