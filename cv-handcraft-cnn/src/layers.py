import numpy as np


class Conv2D:
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Xavier初始化权重 - 形状: (output_channels, input_channels, kernel_size, kernel_size)
        scale = np.sqrt(2.0 / (input_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(output_channels)

        self.cache = None

    def forward(self, x):
        """前向传播"""
        # 输入形状: (batch_size, height, width, channels)
        batch_size, in_height, in_width, in_channels = x.shape

        # 计算输出尺寸
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 添加padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (self.padding, self.padding),
                                  (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            x_padded = x

        # 输出形状: (batch_size, out_height, out_width, output_channels)
        output = np.zeros((batch_size, out_height, out_width, self.output_channels))

        # 实现卷积操作
        for b in range(batch_size):
            for out_ch in range(self.output_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # 提取局部区域
                        region = x_padded[b, h_start:h_end, w_start:w_end, :]

                        # 计算卷积 - 使用正确的权重形状
                        conv_sum = 0
                        for in_ch in range(self.input_channels):
                            # 权重形状: (output_channels, input_channels, kernel_size, kernel_size)
                            # 区域形状: (kernel_size, kernel_size, input_channels)
                            # 我们需要提取对应输入通道的权重和区域
                            weight_slice = self.weights[out_ch, in_ch, :, :]
                            region_slice = region[:, :, in_ch]
                            conv_sum += np.sum(weight_slice * region_slice)

                        output[b, i, j, out_ch] = conv_sum + self.bias[out_ch]

        self.cache = x
        return output

    def backward(self, dout):
        """反向传播"""
        x = self.cache
        batch_size, in_height, in_width, in_channels = x.shape

        # 初始化梯度
        dweights = np.zeros_like(self.weights)
        dbias = np.zeros_like(self.bias)

        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (self.padding, self.padding),
                                  (self.padding, self.padding), (0, 0)), mode='constant')
            dx_padded = np.zeros_like(x_padded)
        else:
            x_padded = x
            dx_padded = np.zeros_like(x)

        out_height, out_width = dout.shape[1], dout.shape[2]

        # 计算梯度
        for b in range(batch_size):
            for out_ch in range(self.output_channels):
                dbias[out_ch] += np.sum(dout[b, :, :, out_ch])

                for in_ch in range(self.input_channels):
                    for i in range(out_height):
                        for j in range(out_width):
                            h_start = i * self.stride
                            h_end = h_start + self.kernel_size
                            w_start = j * self.stride
                            w_end = w_start + self.kernel_size

                            region = x_padded[b, h_start:h_end, w_start:w_end, in_ch]
                            dweights[out_ch, in_ch] += dout[b, i, j, out_ch] * region

                            dx_padded[b, h_start:h_end, w_start:w_end, in_ch] += \
                                dout[b, i, j, out_ch] * self.weights[out_ch, in_ch]

        # 去除padding
        if self.padding > 0:
            dx = dx_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dx = dx_padded

        return dx, dweights, dbias


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x):
        """前向传播"""
        batch_size, in_height, in_width, channels = x.shape

        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, out_height, out_width, channels))
        mask = np.zeros_like(x)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = x[b, h_start:h_end, w_start:w_end, c]
                        output[b, i, j, c] = np.max(region)

                        # 记录最大值位置
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        mask[b, h_start + max_idx[0], w_start + max_idx[1], c] = 1

        self.cache = mask
        return output

    def backward(self, dout):
        """反向传播"""
        mask = self.cache
        dx = np.zeros_like(mask)

        batch_size, out_height, out_width, channels = dout.shape

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        dx[b, h_start:h_end, w_start:w_end, c] += \
                            mask[b, h_start:h_end, w_start:w_end, c] * dout[b, i, j, c]

        return dx


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        x = self.cache
        dx = dout * (x > 0)
        return dx


class Flatten:
    def __init__(self):
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.original_shape)


class Dense:
    def __init__(self, input_size, output_size):
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros(output_size)
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, dout):
        x = self.cache
        dx = np.dot(dout, self.weights.T)
        dweights = np.dot(x.T, dout)
        dbias = np.sum(dout, axis=0)
        return dx, dweights, dbias


class Softmax:
    def forward(self, x):
        # 数值稳定性改进
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, dout):
        # 通常在损失函数中直接计算
        return dout