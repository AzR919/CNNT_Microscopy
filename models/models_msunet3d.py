import torch
import torch.nn as nn
from functools import partial
from torch.profiler import profile, record_function, ProfilerActivity


@torch.jit.script
def channel_shuffle(x, groups: int):
    batchsize, num_channels, depth, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, depth, height, width).transpose(1, 2).contiguous()
    return x.view(batchsize, -1, depth, height, width)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

class MSDC3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dwconvs = nn.ModuleList([
            nn.Conv3d(self.in_channels, self.out_channels, kernel_size, stride, kernel_size // 2, groups=gcd(self.in_channels, self.out_channels), bias=False)
            for kernel_size in kernel_sizes
        ])

        #self.pointwise_conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        #self.batch_norm = nn.BatchNorm3d(out_channels)
        #self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.ReLU6(inplace=True)
        
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        # Initialize weights if necessary
        pass

    def forward(self, x):
        depthwise_outs = [dwconv(x) for dwconv in self.dwconvs]
        depthwise_outs = sum(depthwise_outs)
        #pointwise_out = self.pointwise_conv(summed)
        #output = self.batch_norm(pointwise_out)
        #output = self.activation(output)
        return depthwise_outs

class MSCB3D(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5], expansion_factor=1, dw_parallel=True, add=True, activation='relu6'):
        super(MSCB3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)

        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.msdc = MSDC3D(self.in_channels, self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)

        self.combined_channels = self.ex_channels*1
        #else:
        #    self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv3d(self.combined_channels, self.out_channels, 1, 1, 0, groups=gcd(self.combined_channels, self.out_channels), bias=False)#, #groups=gcd(self.combined_channels, self.out_channels), 
            #nn.InstanceNorm3d(self.out_channels)
            #nn.BatchNorm3d(self.out_channels),
        )

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        # Initialize weights if necessary
        pass

    def forward(self, x):
        out = self.msdc(x)
        out = channel_shuffle(out, gcd(self.in_channels,self.combined_channels))
        out = self.pconv2(out)
        return out

class MSCBLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, stride=1, kernel_sizes=[1,3,5], expansion_factor=1, dw_parallel=True, add=True, activation='relu6'):
        super(MSCBLayer3D, self).__init__()
        layers = [MSCB3D(in_channels, out_channels, stride, kernel_sizes, expansion_factor, dw_parallel, add, activation)]
        for _ in range(1, n):
            layers.append(MSCB3D(out_channels, out_channels, 1, kernel_sizes, expansion_factor, dw_parallel, add, activation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class MSUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512], kernel_sizes=[1, 3, 5]):
        super(MSUNet3D, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()

        for feature in features:
            self.encoder.append(MSCBLayer3D(in_channels, feature, n=1, kernel_sizes=kernel_sizes))
            in_channels = feature

        for c_idx in range (len(features)-1):
            self.decoder.append(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            )
            self.decoder.append(MSCBLayer3D(features[-(c_idx+1)], features[-(c_idx+2)], n=1, kernel_sizes=kernel_sizes))

        self.decoder.append(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        self.decoder.append(MSCBLayer3D(features[0], out_channels, n=1, kernel_sizes=kernel_sizes))
        #self.upsample3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.bottleneck = MSCBLayer3D(features[-1], features[-1], n=1, kernel_sizes=kernel_sizes)
        #self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = skip_connection + x #torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)

        return x #self.upsample3d(x) #self.final_conv(x)

# Example usage
if __name__ == "__main__":
    model = MSUNet3D(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 16, 128, 128)  # Example input tensor
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            output = model(x)
    print(output.shape)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_stacks("profiler_output.txt", metric="self_cuda_time_total")
