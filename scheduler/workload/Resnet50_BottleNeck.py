workload = {
    0: {
        'op_type': 'Conv',
        'op_source': {'I': None, 'W': None},
        'op_datatype': {'I': 'float32', 'W': 'float32'},
        'constant_op': ['I', 'W'],
        'input_size': {'N': 1, 'H': 56, 'W': 56, 'C': 64},
        'weight_size': {'K': 64, 'H': 1, 'W': 1, 'C': 64}, #for kernel: K, H, W, C
        'output_size': {'N': 1, 'H': 56, 'W': 56,'C': 64},
        'padding': {'IX': 0, 'IY': 0},
        'stride': (1, 1)
    },

    1: {
        'op_type': 'BatchNorm',
        'op_source': {'I': 0, 'W': None},
        'op_datatype': {'I': 'float32', 'W': 'float32'},
        'constant_op': ['W'],
        'input_size': {'N': 1,'H': 56, 'W': 56,'C': 64},
        'weight_size': {'mean': 64,'gamma': 64, 'beta': 64}, #for BN: mean, gamma, beta
        'output_size': {'N': 1,'H': 56,'W': 56,'C': 64},
        'padding': {'IX': 0, 'IY': 0},
        'stride': ()
    },

    2: {
        'op_type': 'Relu',
        'op_source': {'I': 1, 'W': None},
        'op_datatype': {'I': 'float32'},
        'constant_op': [],
        'input_size': {'N': 1,'H': 56, 'W': 56,'C': 64},
        'weight_size': {},
        'output_size': {'N': 1,'H': 56,'W': 56,'C': 64},
        'padding': {'IX': 0, 'IY': 0},
        'stride': ()
    },

    3: {
        'op_type': 'Conv',
        'op_source': {'I': 2, 'W': None},
        'op_datatype': {'I': 'float32', 'W': 'float32'},
        'constant_op': ['W'],
        'input_size': {'N': 1, 'H': 56, 'W': 56, 'C': 64},
        'weight_size': {'K': 64, 'H': 3, 'W': 3, 'C': 64}, #for kernel: K, H, W, C
        'output_size': {'N': 1, 'H': 56, 'W': 56,'C': 64},
        'padding': {'IX': 1, 'IY': 1},
        'stride': (1, 1)
    },

    4: {
        'op_type': 'BatchNorm',
        'op_source': {'I': 3, 'W': None},
        'op_datatype': {'I': 'float32', 'W': 'float32'},
        'constant_op': ['W'],
        'input_size': {'N': 1,'H': 56, 'W': 56,'C': 64},
        'weight_size': {'mean': 64,'gamma': 64, 'beta': 64}, #for BN: mean, gamma, beta
        'output_size': {'N': 1,'H': 56,'W': 56,'C': 64},
        'padding': {'IX': 0, 'IY': 0},
        'stride': ()
    },

    5: {
        'op_type': 'Relu',
        'op_source': {'I': 4, 'W': None},
        'op_datatype': {'I': 'float32'},
        'constant_op': [],
        'input_size': {'N': 1,'H': 56, 'W': 56,'C': 64},
        'weight_size': {},
        'output_size': {'N': 1,'H': 56,'W': 56,'C': 64},
        'padding': {'IX': 0, 'IY': 0},
        'stride': ()
    },

    6: {
        'op_type': 'Conv',
        'op_source': {'I': 5, 'W': None},
        'op_datatype': {'I': 'float32', 'W': 'float32'},
        'constant_op': ['W'],
        'input_size': {'N': 1, 'H': 56, 'W': 56, 'C': 64},
        'weight_size': {'K': 256, 'H': 1, 'W': 1, 'C': 64}, #for kernel: K, H, W, C
        'output_size': {'N': 1, 'H': 56, 'W': 56,'C': 256},
        'padding': {'IX': 0, 'IY': 0},
        'stride': (1, 1)
    },

    7: {
        'op_type': 'BatchNorm',
        'op_source': {'I': 6, 'W': None},
        'op_datatype': {'I': 'float32', 'W': 'float32'},
        'constant_op': ['W'],
        'input_size': {'N': 1,'H': 56, 'W': 56,'C': 256},
        'weight_size': {'mean': 256,'gamma': 256, 'beta': 256}, #for BN: mean, gamma, beta
        'output_size': {'N': 1,'H': 56,'W': 56,'C': 256},
        'padding': {'IX': 0, 'IY': 0},
        'stride': ()
    },

    8: {
        'op_type': 'Conv',
        'op_source': {'I': None, 'W': None},
        'op_datatype': {'I': 'float32', 'W': 'float32'},
        'constant_op': ['I', 'W'],
        'input_size': {'N': 1, 'H': 56, 'W': 56, 'C': 64},
        'weight_size': {'K': 256, 'H': 1, 'W': 1, 'C': 64}, #for kernel: K, H, W, C
        'output_size': {'N': 1, 'H': 56, 'W': 56,'C': 256},
        'padding': {'IX': 0, 'IY': 0},
        'stride': (1, 1)
    },

    9: {
        'op_type': 'BatchNorm',
        'op_source': {'I': 8, 'W': None},
        'op_datatype': {'I': 'float32', 'W': 'float32'},
        'constant_op': ['W'],
        'input_size': {'N': 1,'H': 56, 'W': 56,'C': 256},
        'weight_size': {'mean': 256,'gamma': 256, 'beta': 256}, #for BN: mean, gamma, beta
        'output_size': {'N': 1,'H': 56,'W': 56,'C': 256},
        'padding': {'IX': 0, 'IY': 0},
        'stride': ()
    },

    10: {
        'op_type': 'Add',
        'op_source': {'I': 7, 'W': 9},
        'op_datatype': {'I': 'float32', 'W': 'float32'},
        'constant_op': [],
        'input_size': {'N': 1,'H': 56, 'W': 56,'C': 256},
        'weight_size': {'N': 1,'H': 56, 'W': 56,'C': 256},
        'output_size': {'N': 1,'H': 56,'W': 56,'C': 256},
        'padding': {'IX': 0, 'IY': 0},
        'stride': ()
    },

    11: {
        'op_type': 'Relu',
        'op_source': {'I': 10, 'W': None},
        'op_datatype': {'I': 'float32'},
        'constant_op': [],
        'input_size': {'N': 1,'H': 56, 'W': 56,'C': 256},
        'weight_size': {},
        'output_size': {'N': 1,'H': 56,'W': 56,'C': 256},
        'padding': {'IX': 0, 'IY': 0},
        'stride': ()
    }
}