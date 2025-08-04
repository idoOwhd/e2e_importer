import sys
import torch
import os
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"

sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)



import torch
import torch.nn as nn
import numpy as np

class Conv1D(nn.Module):
    def __init__(self, N, C, L, K, KL, stride, padding, dilation):
        super().__init__()
        self.N = N
        self.C = C
        self.L = L
        self.K = K
        self.KL = KL
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, A_torch, B_torch):
        # Perform 1D convolution
        C_torch = torch.nn.functional.conv1d(
            A_torch, B_torch, bias=None, stride=self.stride, 
            padding=self.padding, dilation=self.dilation
        )
        return C_torch

    def prepare(self, dtype=torch.float32, device=torch.cuda.current_device()):
        # Generate random input data
        A_np = np.random.uniform(-10, 10, [self.N, self.C, self.L]).astype("float32")
        B_np = np.random.uniform(-10, 10, [self.K, self.C, self.KL]).astype("float32")

        # Convert to PyTorch tensors and move to GPU
        A_torch = torch.tensor(A_np).type(dtype).to(device)
        B_torch = torch.tensor(B_np).type(dtype).to(device)

        ret = {
            'input': {
                'A': A_torch,
                'B': B_torch,
            },
            'output': ['C']
        }
        return ret


byte_net_shapes = [
  # (   C,   L,   K,  KL, stride,padding, dilation)
    ( 512, 892, 512,   3,       1,     2,        1),
    ( 512, 892,1024,   1,       1,     0,        1),
    (1024, 892, 512,   1,       1,     0,        1),
    ( 512, 892, 512,   3,       1,     4,        2),
    ( 512, 892, 512,   3,       1,     8,        4),
    ( 512, 892, 512,   3,       1,    16,        8),
    ( 512, 892, 512,   3,       1,    32,       16),
    (1024, 892, 250,   1,       1,     0,        1)
]
# batch = 1 / 8 / 16

model = Conv1D(
    N=1, C=512, L=892, K=1024, KL=1, 
    stride=1, padding=0, dilation=1)
input_data = model.prepare()
A = input_data['input']['A']
B = input_data['input']['B']
E2ESHARK_CHECK["input"] = [A, B]
E2ESHARK_CHECK["output"] = model(A, B)
print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])


  