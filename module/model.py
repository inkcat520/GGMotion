import torch
import math
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from module.modules import cosine_embedding


class GGMNet(nn.Module):
    def __init__(self, config, group, edges):
        super().__init__()
        self.group = group
        self.edges = torch.from_numpy(edges).long()
        self.act = nn.SiLU()
        self.key_n = config.key_point
        self.input_n = config.past_length
        self.output_n = config.future_length
        self.seq_n = self.input_n + self.output_n
        self.e_dim = config.e_dim
        self.h_dim = config.h_dim
        self.norm = config.norm
        self.eps = 1e-6
        self.fk = config.fk
        self.n_layer = config.n_layer

        # init
        self.x_emb = nn.Linear(self.input_n, self.e_dim, bias=False)
        self.v_emb = nn.Linear(self.input_n, self.e_dim, bias=False)

        self.blocks = nn.ModuleList(
            [Block(self.key_n, self.h_dim, self.e_dim, self.group,
                   self.edges, self.act, self.eps, self.norm, self.fk)
             for _ in range(self.n_layer)])

        self.center_mlp = nn.ModuleList(
            [nn.Linear(self.e_dim, self.output_n, bias=False)
             for _ in range(self.n_layer)])

        self.pro = nn.Linear(self.e_dim, self.output_n, bias=False)

    def forward(self, xco):  # (B,N,3,T)
        batch_size = xco.shape[0]

        vel = torch.zeros_like(xco, device=xco.device)
        vel[..., 1:] = xco[..., 1:] - xco[..., :-1]
        vel[..., 0] = vel[..., 1]

        x_center = torch.mean(xco, dim=(1, -1), keepdim=True)
        x = self.x_emb(xco - x_center) + x_center
        v = self.v_emb(vel)

        attr1 = cosine_embedding(self.edges[:, 2], self.h_dim).unsqueeze(0).expand(x.shape[0], -1, -1)
        attr2 = cosine_embedding(torch.arange(0, self.key_n), self.h_dim).unsqueeze(0).expand(x.shape[0], -1, -1)

        for i, block in enumerate(self.blocks):
            x, v = block(x, v, attr1, attr2, x_center)
            x_center = self.center_mlp[i](x)
            x_center = torch.mean(x_center, dim=(1, -1), keepdim=True)

        out = self.pro(x - x_center) + x_center  # (B,N,3,T)

        return out


class Block(nn.Module):
    def __init__(self, key_n, h_dim, e_dim, group, edges, act, eps, norm=False, fk=False):
        super().__init__()
        self.group = group
        self.edges = edges
        self.norm = norm
        self.key_n = key_n
        self.h_dim = h_dim
        self.eps = eps
        self.fk = fk

        self.spatio_mlp = nn.Linear(e_dim, e_dim, bias=False)
        self.temporal_mlp = nn.Linear(e_dim, e_dim, bias=False)

        self.spatio_scale = nn.Parameter(torch.zeros(self.key_n, 3))
        self.temporal_scale = nn.Parameter(torch.zeros(self.key_n, 3))

        self.spatio_att_mlp = nn.Sequential(
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

        self.spatio_h_mlp = nn.Sequential(
            nn.Linear(e_dim, h_dim),
            act,
            nn.Linear(h_dim, e_dim)
        )

        self.temporal_h_mlp = nn.Sequential(
            nn.Linear(e_dim, h_dim),
            act,
            nn.Linear(h_dim, e_dim)
        )

        # group force out
        self.group_force_emb = nn.ModuleList([
            nn.Linear(e_dim, e_dim, bias=False)
            for _ in range(3)])

        self.group_force_mlp = nn.Sequential(
            nn.Linear(len(self.group)**2, h_dim),
            act,
            nn.Linear(h_dim, len(self.group)**2),
        )
        self.group_force_out = nn.Linear(e_dim, e_dim, bias=False)

        # part force out
        self.part_force_emb = nn.ModuleList([
            nn.Linear(e_dim, e_dim, bias=False)
            for _ in range(3)])

        self.part_force_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(part) ** 2, h_dim),
                act,
                nn.Linear(h_dim, len(part) ** 2),
            ) for part in self.group.values()
        ])
        self.part_force_out = nn.Linear(e_dim, e_dim, bias=False)

        self.a_out_mlp = nn.Linear(e_dim, e_dim, bias=False)

        if self.fk == 1:
            self.acc_emb = nn.ModuleList([
                nn.Linear(e_dim, e_dim, bias=False)
                for _ in range(3)])

            self.n_basis = 3
            self.acc_mlp = nn.Sequential(
                nn.Linear(self.n_basis ** 2, h_dim),
                act,
                nn.Linear(h_dim, self.n_basis),
            )
            self.acc_out = nn.Linear(e_dim, e_dim, bias=False)

    def forward(self, x, v, attr1, attr2, x_center):  # (B,N,E) (B,N,3,E) (B,N,H) (N,E)
        batch_size = x.shape[0]

        spatio_x = x[:, self.edges[:, 0]] - x[:, self.edges[:, 1]]  # (B,N,3,E)
        spatio_att = self.spatio_att_mlp(attr1).unsqueeze(-2)
        scale = self.spatio_h_mlp(torch.norm(spatio_x, dim=-2, p=2)).unsqueeze(-2)
        spatio_x = spatio_att * scale * self.spatio_mlp(spatio_x)  # (B,G,3,E)

        spatio_force = torch.zeros(x.shape, device=x.device)  # (B,N,3,E)
        index = self.edges[:, 0][None, :, None, None].to(x.device)
        index = index.expand(spatio_x.shape[0], -1, spatio_x.shape[2], spatio_x.shape[3])
        spatio_force.scatter_add_(1, index, spatio_x)
        spatio_force = v + self.spatio_scale.unsqueeze(-1) * spatio_force

        temporal_x = x - x_center
        scale = self.temporal_h_mlp(torch.norm(temporal_x, dim=-2, p=2)).unsqueeze(-2)
        temporal_force = scale * self.temporal_mlp(temporal_x)
        temporal_force = v + self.temporal_scale.unsqueeze(-1) * temporal_force

        f = spatio_force + temporal_force

        # group force
        group_force = torch.zeros(f.shape[0], len(self.group), f.shape[2], f.shape[3], device=f.device)  # (B,P,3,E)
        for index, part in enumerate(self.group.values()):
            group_force[:, index] = torch.sum(f[:, part[:, 1]], dim=1)

        force_k = self.group_force_emb[0](group_force).permute(0, 3, 2, 1)  # (B,E,3,P)
        force_q = self.group_force_emb[1](group_force).permute(0, 3, 2, 1)
        force_v = self.group_force_emb[2](group_force).permute(0, 3, 2, 1)
        invar_force = torch.matmul(force_q.transpose(2, 3), force_k)  # (B,E,P,P)
        if self.norm:
            invar_force = F.normalize(invar_force, dim=-1, p=2, eps=self.eps)
        invar_force = self.group_force_mlp(invar_force.flatten(-2)).view(batch_size, -1, invar_force.shape[-2], invar_force.shape[-1])  # (B,E,P,P)
        group_force = self.group_force_out(torch.matmul(force_v, invar_force).permute(0, 3, 2, 1))  # (B,P,3,E)

        act_force = torch.zeros(f.shape, device=f.device)  # (B,N,3,E)
        for index, part in enumerate(self.group.values()):
            act_force[:, part[:, 1]] = group_force[:, index].unsqueeze(1).repeat(1, len(part[:, 1]), 1, 1)

        f = f + act_force

        x_out = torch.zeros_like(x, device=x.device)  # (B,N,3,E)
        v_out = torch.zeros_like(v, device=v.device)
        for index, part in enumerate(self.group.values()):
            # part force
            act_force = f[:, part[:, 1]]  # (B,G,3,E)
            force_k = self.part_force_emb[0](act_force).permute(0, 3, 2, 1)  # (B,E,3,P)
            force_q = self.part_force_emb[1](act_force).permute(0, 3, 2, 1)
            force_v = self.part_force_emb[2](act_force).permute(0, 3, 2, 1)
            invar_force = torch.matmul(force_q.transpose(2, 3), force_k)  # (B,E,P,P)
            if self.norm:
                invar_force = F.normalize(invar_force, dim=-1, p=2, eps=self.eps)
            invar_force = self.part_force_mlp[index](invar_force.flatten(-2)).view(batch_size, -1, invar_force.shape[-2], invar_force.shape[-1]) # (B,E,P,P)
            act_force = self.part_force_out(torch.matmul(force_v, invar_force).permute(0, 3, 2, 1))  # (B,P,3,E)

            r_diff = x[:, part[:, 1]] - x[:, part[:, 0]]  # (B,G,3,E)
            v_diff = v[:, part[:, 1]] - v[:, part[:, 0]]  # (B,G,3,E)
            f_diff = f[:, part[:, 1]] + act_force

            if self.fk == 1:
                a = torch.stack((f_diff, r_diff, v_diff), dim=-1)  # (B,G,3,E,P)
                a_k = self.acc_emb[0](a.transpose(3, 4)).permute(0, 1, 4, 2, 3)  # (B,E,3,F)
                a_q = self.acc_emb[1](a.transpose(3, 4)).permute(0, 1, 4, 2, 3)  # (B,E,3,F)
                a_v = self.acc_emb[2](a.transpose(3, 4)).permute(0, 1, 4, 2, 3)  # (B,E,3,F)
                invar_a = torch.matmul(a_q.transpose(3, 4), a_k)  # (B,G,E,H,H)
                if self.norm:
                    invar_a = F.normalize(invar_a, dim=-1, p=2, eps=1e-6)  # (B,G,E,H*H)
                invar_a = self.acc_mlp(invar_a.flatten(-2)).unsqueeze(-1)  # (B,G,E,H,1)
                a = self.acc_out(torch.matmul(a_v, invar_a).squeeze(-1).permute(0, 1, 3, 2))  # (B,G,3,E)
                a_out = f_diff - a   # (B,3,E)

            elif self.fk == 2:
                part_ind = pd.factorize(part[1:, :2].flatten(), sort=False)[0].reshape(-1, 2)
                a_out = torch.zeros_like(f_diff, device=x.device)  # (B,G,3,E)
                a_out[:, 0] = torch.sum(f_diff, dim=1)  # (B,3,E)

                for i, j in part_ind:
                    cur_a = a_out[:, i]
                    cur_r = r_diff[:, j]
                    cur_v = v_diff[:, j]
                    cur_f = f_diff[:, j]
                    w_a = torch.cross(cur_r, cur_f - cur_a, dim=1) / torch.sum(cur_r ** 2, dim=1, keepdim=True).clamp_min(self.eps)
                    w = torch.cross(cur_r, cur_v, dim=1) / torch.sum(cur_r ** 2, dim=1, keepdim=True).clamp_min(self.eps)
                    a_out[:, j] = cur_a + torch.cross(w_a, cur_r) + torch.cross(w, cur_v)

            else:
                a_out = f_diff   # (B,3,E)

            v_out[:, part[:, 1]] = v[:, part[:, 1]] + self.a_out_mlp(a_out)
            x_out[:, part[:, 1]] = x[:, part[:, 1]] + v_out[:, part[:, 1]]

        return x_out, v_out
