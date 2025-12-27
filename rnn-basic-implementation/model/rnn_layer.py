# model/rnn_layer.py
"""
RNN层实现（占位文件）
在实际的多层RNN中会用到
"""


class RNNDummyLayer:
    """简单的RNN层包装器"""

    def __init__(self, cell):
        self.cell = cell

    def forward(self, x, h_prev):
        return self.cell.forward(x, h_prev)

    def backward(self, dh_next, dy):
        return self.cell.backward(dh_next, dy)