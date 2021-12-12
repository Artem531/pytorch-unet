from adaptive_conv import adaConv2d
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__

    # for every Conv2d layer in a model..
    if classname.find('Conv2d') != -1:
        # get the number of the inputs
        n = m.out_channels * m.kernel_size[0] * m.kernel_size[1]

        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(1)

class adaModule(nn.Module):
    """
    paper module
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
    ):
        super(adaModule, self).__init__()

        self.conv = adaConv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

    def forward(self, input: Tensor, scales: Tensor) -> Tensor:
        return self.conv(input, scales=scales)

def get_inference_time(model, device):
    """
    calc mean inference time of model
    :param model: input model
    :param device:
    :return:
    """
    dummy_input = torch.randn(5, 3,256,256, dtype=torch.float).to(device)
    dummy_scales = torch.randn(5, 3, 256, 256, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))

    #GPU-WARM-UP
    for _ in range(10):
       _ = model(dummy_input, dummy_scales)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input, dummy_scales)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rand_t = torch.rand(5, 3, 7, 7).to(device)

    test_conv = adaModule(3, 64, kernel_size=3, dilation=1, padding=0, stride=1).to(device)
    print(get_inference_time(test_conv, device))
