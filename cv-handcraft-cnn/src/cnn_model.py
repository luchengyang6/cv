import numpy as np
from src.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax


class SimpleCNN:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        # 使用更简单的结构
        self.layers = [
            Conv2D(input_channels=1, output_channels=4, kernel_size=3, padding=1),  # 输出: (28, 28, 4)
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),  # 输出: (14, 14, 4)

            Conv2D(input_channels=4, output_channels=8, kernel_size=3, padding=1),  # 输出: (14, 14, 8)
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),  # 输出: (7, 7, 8)

            Flatten(),  # 输出: (7*7*8 = 392)
            Dense(input_size=7 * 7 * 8, output_size=64),
            ReLU(),
            Dense(input_size=64, output_size=num_classes)
        ]

        self.softmax = Softmax()
        self.debug_printed = False

    def forward(self, x, training=True):
        """前向传播"""
        # 只在第一个批次打印形状
        if training and not self.debug_printed:
            print(f"输入形状: {x.shape}")
            self.debug_printed = True

        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if training and not self.debug_printed and i < 6:  # 只打印前几层的形状
                print(f"第{i}层 {layer.__class__.__name__} 输出形状: {x.shape}")

        # 重置调试标志，以便下一个epoch的第一个批次可以打印
        if not training:
            self.debug_printed = False

        return self.softmax.forward(x)

    def backward(self, dout):
        """反向传播"""
        grads = []
        dout = self.softmax.backward(dout)

        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                if isinstance(layer, (Conv2D, Dense)):
                    result = layer.backward(dout)
                    if result is not None:
                        if len(result) == 3:
                            dout, dw, db = result
                            grads.append((dw, db))
                        else:
                            dout = result
                else:
                    dout = layer.backward(dout)

        return list(reversed(grads))

    def get_parameters(self):
        """获取所有可训练参数"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.append((layer.weights, layer.bias))
        return params

    def set_parameters(self, params):
        """设置参数"""
        for i, (weights, bias) in enumerate(params):
            layer = self.layers[i]
            if hasattr(layer, 'weights'):
                layer.weights = weights
                layer.bias = bias