import time
from multiprocessing import Process

def coding(language):
    '''子进程要执行的程序'''
    for i in range(5):
        print('{} coding'.format(language), end='|')
        time.sleep(1)
        
if __name__ == '__main__':
    # 单进程
    start = time.time()
    coding('python')
    for i in range(5):
        print('main program', end='|')
        time.sleep(1)
    end = time.time()
    print('\nOne process cost time:', end - start)
    
    # 多进程
    multi_start = time.time()
    p = Process(target=coding, args=('python',))
    p.start()
    for i in range(5):
        print('main program', end='|')
        time.sleep(1)
    multi_end = time.time()
    print('\nMulti process cost time:', multi_end - multi_start)