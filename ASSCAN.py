from math import ceil
import torch
import torch.nn.functional as F
from torch import nn
from HSDC import HSDCmodel
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        last_layer = m[-1]
        if hasattr(last_layer, 'weight'):
            nn.init.constant_(last_layer.weight, 0)
        if hasattr(last_layer, 'bias') and last_layer.bias is not None:
            nn.init.constant_(last_layer.bias, 0)
    else:
        if hasattr(m, 'weight'):
            nn.init.constant_(m.weight, 0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DCSA(nn.Module):
    def __init__(self, inplanes, planes, pool='att'):
        super(DCSA, self).__init__()
        assert pool in ['avg', 'att']
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        
        if self.pool == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            if hasattr(self.conv_mask, 'weight'):
                nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in', nonlinearity='relu')
            if hasattr(self.conv_mask, 'bias') and self.conv_mask.bias is not None:
                nn.init.constant_(self.conv_mask.bias, 0)
        
        last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width).unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = self.softmax(context_mask.view(batch, 1, height * width))
            context = torch.matmul(input_x, context_mask.unsqueeze(3))
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)          
        channel_add_term = self.channel_add_conv(context) 
        out = x + channel_add_term              
        return out
class FUSH(nn.Module):
    def __init__(self, in_channels):
        super(FUSH, self).__init__()
        self.in_channels = in_channels
        r = 16 
        planes = 32
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)
        
        self.CA = HSDCmodel(img_size=120, in_chans=64, embed_dim=192, 
                 depths=[2,2,6,2], num_heads=[6,12,12,12], 
                 patch_size=3, window_size=10, 
                 pretrained_window_sizes=[10,10,10,5],
                 mlp_ratio=4., qkv_bias=True)
        self.CAgap   = nn.AdaptiveAvgPool2d(output_size=1)
        self.CA2 = nn.Conv2d(in_channels=1536, out_channels=in_channels, kernel_size=1)

        self.SA = DCSA(inplanes=in_channels,planes=planes)
        self.conv5 = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        

        self.sigmoid = nn.Sigmoid()


        self.channels = in_channels
        d = 32
        self.fc = nn.Sequential(
            nn.Linear(2 * in_channels, d),
            nn.ReLU()
        )

        self.A = nn.Parameter(torch.randn(in_channels, d)) 
        self.B = nn.Parameter(torch.randn(in_channels, d))
    def forward(self, x):
        u       = self.conv2(F.relu(self.conv1(x)))
        M_CA    = self.sigmoid(self.CA2(self.CAgap(self.sigmoid(self.CA(u)[3]))))
        M_SA    = self.sigmoid(self.conv5(self.SA(u)))
        U_CA    = u*M_CA
        U_SA    = u*M_SA
 
        S1 = torch.mean(U_CA, dim=[2,3])
        S2 = torch.mean(U_SA, dim=[2,3])
        
        S_cat = torch.cat([S1, S2], dim=1)
        Z = self.fc(S_cat)
        
        score_A = torch.einsum("bd,cd->bc", Z, self.A)
        score_B = torch.einsum("bd,cd->bc", Z, self.B)
        
 
        exp_A = torch.exp(score_A) 
        exp_B = torch.exp(score_B)  
        denominator = exp_A + exp_B 
        
        W1 = exp_A / denominator  
        W2 = exp_B / denominator   
        
        W1 = W1.view(-1, self.channels, 1, 1) 
        W2 = W2.view(-1, self.channels, 1, 1)  
        Y = U_CA * W1 + U_SA * W2                  

        out     = Y + x
        return out



class ASSCAN(nn.Module):
    def __init__(self, inchannels=145):
        super(ASSCAN, self).__init__()
        self.in_channels    = inchannels
        self.out_channels   = inchannels
        self.N_Filters      = 64
        self.N_modules      = 1

        self.FEN    = nn.Conv2d(in_channels=self.in_channels+1, out_channels=self.N_Filters, kernel_size=3, padding=1)

        self.FUSH1   = FUSH(in_channels=self.N_Filters)
        self.FUSH2   = FUSH(in_channels=self.N_Filters)
        self.FUSH3   = FUSH(in_channels=self.N_Filters)
        self.FUSH4   = FUSH(in_channels=self.N_Filters)

        self.ReB    = nn.Conv2d(in_channels=self.N_Filters, out_channels=self.out_channels, kernel_size=3, padding=1)
        

    def forward(self, X_MS, X_PAN):
        X_MS_UP = X_MS
        x = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)


        x = self.FEN(x)


        x = self.FUSH1(x)
        x = self.FUSH2(x)
        x = self.FUSH3(x)
        x = self.FUSH4(x)

        x = self.ReB(x)

        x = x + X_MS_UP

        output = {"pred": x}
        return output

