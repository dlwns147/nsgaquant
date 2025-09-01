import torch
import torch.nn as nn
from torch.nn import functional as F

# class JSD(nn.Module):
#     def __init__(self, tau=1., reduction='batchmean'):
#         super(JSD, self).__init__()
#         self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)
#         # self.kl = nn.KLDivLoss(reduction='sum', log_target=True)
#         self.tau = tau

#     def forward(self, p: torch.tensor, q: torch.tensor):
#         p, q = (p / self.tau).log_softmax(-1), (q / self.tau).log_softmax(-1)
#         m = (0.5 * (p + q))
#         return 0.5 * (self.kl(m, p) + self.kl(m, q))
    
class JSD(nn.Module):
    def __init__(self, reduction='batchmean', eps=1e-7):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)
        self.eps = eps

    def forward(self, p: torch.tensor, q: torch.tensor):
        m = (0.5 * (p.softmax(-1) + q.softmax(-1))).clamp_min(self.eps).log()
        return 0.5 * (self.kl(m, p.log_softmax(-1)) + self.kl(m, q.log_softmax(-1)))
    