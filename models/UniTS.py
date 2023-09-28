# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.Uni_backbone import Uni_backbone
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
from utils.tools import RevIN
# from utils.tools import multi_RevIN as RevIN
import sys

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.target_window = configs.pred_len
        # jnject position coviriate or not
        n_layers = configs.nhl
        self.norm = configs.norm
        usepe = configs.usepe
        useatt = configs.useatt
        n_heads = configs.n_heads
        d_model = configs.ims

        self.usefreq = configs.usefreq
        self.usepembed = configs.pembed

        self.uselocal = configs.uselocal
        self.useglobal = configs.useglobal
        self.global_addresidual = configs.gaddres
        self.layer_addresidual = configs.laddres
        self.convkernel = configs.convkernel

        self.deconly = configs.deconly
        # dense_e = configs.edims
        # dense_d = configs.ddims
        d_ff = configs.ims
        # TODO 补充dropout参数
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        affine = True
        subtract_last = False
        
        decomposition = configs.d
        kernel_size = configs.decomp_kernel
        self.length_ratio = (self.seq_len + self.target_window)/self.seq_len
        # d_ff = 32
        
        # model
        if self.norm=='allrev':
            self.revin_layer = RevIN(c_in, affine=True, subtract_last= False)
            revin=False
        else:
            if self.norm=='revin':
                revin=True
            else:
                revin=False
        self.decomposition = decomposition
        # self.regression = nn.Linear(configs.seq_len, configs.pred_len)
        # dynamic_input_size = (configs.seq_len + configs.pred_len + configs.label_len)*5
        # print(dynamic_input_size)
        self.dynamic_proj = nn.Linear( 5 , 1)
        # self.linear_proj = nn.Linear(self.seq_len, self.target_window)
        if self.usefreq:
            self.dominance_freq= configs.dominance_freq
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat)
        if self.uselocal == 1 or self.useglobal == 1:
            if self.decomposition:
                self.decomp_module = series_decomp(kernel_size)
                # self.multi_decomp = [series_decomp(kernel) for kernel in kernel_size]
                self.model_trend = Uni_backbone(c_in=c_in, context_window = self.seq_len, target_window=self.target_window, patch_len=patch_len, stride=stride, 
                                    max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,  usepembed = self.usepembed, deconly = self.deconly,
                                    n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                    dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, uselocal=self.uselocal, useglobal=self.useglobal,
                                    addresidual = self.layer_addresidual, conv_kernel=self.convkernel,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch, usepe=usepe, useatt=useatt,
                                    pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                    subtract_last=subtract_last, verbose=verbose, **kwargs)
                self.model_res = Uni_backbone(c_in=c_in, context_window = self.seq_len, target_window=self.target_window, patch_len=patch_len, stride=stride, usepe=usepe, useatt=useatt,
                                    max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model, usepembed = self.usepembed, deconly = self.deconly,
                                    n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                    dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, uselocal=self.uselocal, useglobal=self.useglobal,
                                    addresidual = self.layer_addresidual, conv_kernel=self.convkernel,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                    pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                    subtract_last=subtract_last, verbose=verbose, **kwargs)
                if self.global_addresidual:
                    self.residualff = nn.Sequential(nn.Linear(self.seq_len, d_ff),
                                        nn.Dropout(dropout),
                                        nn.Linear(d_ff, self.target_window))
            else:
                self.model = Uni_backbone(c_in=c_in, context_window = self.seq_len, target_window=self.target_window, patch_len=patch_len, stride=stride, usepe=usepe, useatt=useatt,
                                    max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model, usepembed = self.usepembed, deconly = self.deconly,
                                    n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                    dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, uselocal=self.uselocal, useglobal=self.useglobal,
                                    addresidual = self.layer_addresidual, conv_kernel=self.convkernel,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                    pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                    subtract_last=subtract_last, verbose=verbose, **kwargs)
                if self.global_addresidual:
                    self.residualff = nn.Sequential(nn.Linear(self.seq_len, d_ff),
                                        nn.Dropout(dropout),
                                        nn.Linear(d_ff, self.target_window))
        # if self.usefreq:
        #     self.model_freq = Uni_backbone(c_in=c_in, context_window = self.dominance_freq, target_window=int(self.dominance_freq*self.length_ratio), patch_len=1, stride=1, usepe=usepe, useatt=useatt,
        #                         max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model, usepembed = self.usepembed, deconly = self.deconly,
        #                         n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm='none', attn_dropout=attn_dropout,
        #                         dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, uselocal=self.uselocal, useglobal=1,
        #                         addresidual = self.layer_addresidual, conv_kernel=self.convkernel,
        #                         attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
        #                         pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
        #                         pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
        #                         subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x, xmark,ymark):           # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1)
        if self.norm=='allrev':
            x = self.revin_layer(x, 'norm')
            tx = x
        tx = x
        # torch.Size([48, 7, 576]) bs, input_seq, n_val
        if self.global_addresidual:
            res_x = self.residualff(tx.permute(0,2,1))

        allmark = torch.cat((xmark, ymark), dim=2).permute(0,2,1)
        dynamic = ymark
        out = 0
        if self.uselocal == 1 or self.useglobal == 1:
            if self.decomposition:
                # TODO multi kernel pooling + 把residual 放到外面来
                res_init, trend_init = self.decomp_module(x)
                # trend = self.regression(trend_init.permute(0,2,1))
                res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
                if self.usepembed:
                    embed_proj = self.dynamic_proj(allmark).permute(0,2,1).repeat(1, x.shape[2], 1)
                    in_proj = embed_proj[:,:,:self.seq_len]
                    target_proj = embed_proj[:,:,-self.target_window:]
                # res_init, trend_init = torch.cat( (res_init, embed_proj), dim=2 ), torch.cat( (trend_init, embed_proj), dim=2 )
                    res = self.model_res(res_init, in_proj, target_proj)
                    trend = self.model_trend(trend_init, in_proj, target_proj)
                else:
                    res = self.model_res(res_init, embed_proj=None, target_proj=None)
                    trend = self.model_trend(trend_init, embed_proj=None, target_proj=None)
                out = res + trend
            else:
                x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
                out = self.model(x, embed_proj=None, target_proj=None)
                x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        if self.usefreq:
            # freq不能用revin
            # x = x.permute(0,2,1)
            # RIN
            # x = x.permute(0,2,1)
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean
            x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
            # print(x_var)
            x = x / torch.sqrt(x_var)
            low_specx = torch.fft.rfft(x, dim=1)
            low_specx[:,self.dominance_freq:]=0 # LPF
            # print(low_specx.shape)
            low_specx = low_specx[:,0:self.dominance_freq,:] # LPF
            # low_specxy_ = self.model_freq(low_specx.permute(0,2,1).to(torch.float32)).permute(0,2,1)
            low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)
            low_specxy = torch.zeros([low_specxy_.size(0),int((self.seq_len+self.target_window)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)
            low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding
            low_xy=torch.fft.irfft(low_specxy, dim=1)
            low_xy=low_xy * self.length_ratio # compemsate the length change
            xy=(low_xy) * torch.sqrt(x_var) +x_mean
            out = out + xy.permute(0,2,1)[:,:,-self.target_window:]
        if self.global_addresidual: out += res_x
        # torch.Size([256, 7, 96])
        if self.norm=='allrev':
            x = out.permute(0,2,1)
            x = self.revin_layer(x, 'denorm')
            x = x.permute(0,2,1)
        return out