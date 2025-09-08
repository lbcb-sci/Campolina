def rf(kernels, strides):
    '''Compute receptive field given kernels and corresponding strides.'''
    result = 0
    product_strides = prev_stride = 1

    for kernel, stride in zip(kernels, strides):
        product_strides *= prev_stride
        rf = (kernel - 1) * product_strides
        result += rf
        prev_stride = stride

    return result

print('RF (at bottleneck) =', rf(
    [9, 
     3, 3, 2, 
     3, 3, 2, 
     3, 3, 2, 
     3, 3, 2, 
     5
     ], 
    [1, 
     1, 1, 2, 
     1, 1, 2, 
     1, 1, 2, 
     1, 1, 2, 
     1],
))
