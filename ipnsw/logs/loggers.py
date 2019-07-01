
import numpy as np
np.set_printoptions(linewidth=np.inf, threshold=32, edgeitems=10)

def log_array(msg, x):
    print(msg + ' (shape=%s)' % str(x.shape))
    print(x)
    print(' ')

def log_sparse_vector(msg, x, threshold=32):
    x = x.copy().squeeze()
    assert len(x.shape) == 1
    print(msg + ' (shape=%s)' % str(x.shape))
    
    idx = np.where(x != 0)[0]
    if len(idx) == 0:
        tmp = '... no nonzero elements...'
    elif len(idx) > threshold:
        tmp = (
            '(index:value) ' + 
            ' '.join(['%d:%f' % (a,b) for a,b in zip(idx[:threshold // 2], x[idx[:threshold // 2]])]) +
            ' ... %d more entries ... ' % (len(idx) - threshold) +
            ' '.join(['%d:%f' % (a,b) for a,b in zip(idx[-threshold // 2:], x[idx[-threshold // 2:]])])
        )
    else:
        tmp = '(index:value) ' + ' '.join(['%d:%f' % (a,b) for a,b in zip(idx, x[idx])])
    
    print(tmp)
    
    # print('nonzero_idx: ', idx)
    # print('nonzero_val: ', x[idx])
    print(' ')

def log_sparse_matrix(msg, x):
    print(msg + '(sparse matrix; shape=%s; # nonzero elements=%d)' % (x.shape, x.nnz))
    x.maxprint = 20
    print(x)
    print(' ')

def log_header(*args, n=1):
    print('=' * n)
    print(*args)