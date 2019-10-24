import torch.nn as nn
import torch.nn.functional as F


class NetDMixture(nn.Module):
    def __init__(self, ndim, num_modes):
        super(NetDMixture, self).__init__()
        
        self.num_modes = num_modes
        self.main = nn.Sequential(
            nn.Linear(ndim, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1)
        )
        
    def forward(self, inp, mode_prob=None, real_flag=False, return_arr=False):
        if real_flag:
            output = self.main(inp)
            output = output.mean(0)
            return output.view(1)
        else:
            if return_arr:
                output_list = []
                for i in range(self.num_modes):
                    output_tmp = self.main(inp[i])
                    output_tmp = output_tmp.mean()
                    output_list.append(output_tmp.item())
                return output_list
            else:            
                for i in range(self.num_modes):
                    if i == 0:
                        output = mode_prob[i]*self.main(inp[i])
                    else:
                        output = output + mode_prob[i]*self.main(inp[i])
                output = output.mean(0)
                return output.view(1)


class NetDMixtureLearnPi(nn.Module):
    def __init__(self, ndim, num_modes):
        super(NetDMixtureLearnPi, self).__init__()
        
        self.num_modes = num_modes
        self.main = nn.Sequential(
            nn.Linear(ndim, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1)
        )

    def forward(self, inp, mode_prob=None, real_flag=False):
        if real_flag:
            output = self.main(inp)
            output = output.mean(0)
            return output.view(1)
        else:
            for i in range(self.num_modes):
                if i == 0:
                    output = F.softmax(mode_prob, dim=0)[i]*self.main(inp[i])
                else:
                    output = output + F.softmax(mode_prob, dim=0)[i]*self.main(inp[i])
            output = output.mean(0)
            return output.view(1)


class NetG(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(NetG, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(inp_dim, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, out_dim),
        )
        
    def forward(self, inp):
        output = self.main(inp)
        return output

