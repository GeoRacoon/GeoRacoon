from landiv_blur.filters import gaussian as lbfgauss


def test_gaussian_():
    kernels = []
    default_sigma = 1.0
    default_size = lbfgauss.get_kernel_size(sigma=default_sigma)
    expected_ks = []
    for scale in [0.5, 1, 1.5, 2, 4]:
        sigma = scale * default_sigma
        kernels.append(lbfgauss.get_kernel_size(sigma))
        expected_ks.append(default_size*scale)
    assert kernels == expected_ks, 'Gaussian kernel size does not scale' \
           'linearly with sigma!'
