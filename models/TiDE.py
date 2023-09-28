# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.TiDE_backbone import PatchTST_backbone
from layers.Autoformer_EncDec import series_decomp
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
        context_window = configs.seq_len
        self.target_window = configs.pred_len
        
        n_layers = configs.nhl
        self.norm = configs.norm
        usepe = configs.usepe
        useatt = configs.useatt
        n_heads = configs.n_heads
        d_model = configs.ims
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
        kernel_size = 25
        
        
        # model
        if self.norm=='allrev':
            self.revin_layer = RevIN(7, affine=True, subtract_last= False)
            revin=False
        else:
            if self.norm=='revin':
                revin=True
            else:
                revin=False
        self.decomposition = decomposition
        self.regression = nn.Linear(configs.seq_len, configs.pred_len)
        # dynamic_input_size = (configs.seq_len + configs.pred_len + configs.label_len)*5
        # print(dynamic_input_size)
        self.dynamic_proj = nn.Linear( 5 , 1)
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=self.target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch, usepe=usepe, useatt=useatt,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=self.target_window, patch_len=patch_len, stride=stride, usepe=usepe, useatt=useatt,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=self.target_window, patch_len=patch_len, stride=stride, usepe=usepe, useatt=useatt,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x, xmark,ymark):           # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1)
        if self.norm=='allrev':
            x = self.revin_layer(x, 'norm')
            # x = x.permute(0,2,1)
        # torch.Size([48, 7, 576]) bs, input_seq, n_val
        allmark = torch.cat((xmark, ymark), dim=2).permute(0,2,1)
        dynamic = ymark
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            # trend = self.regression(trend_init.permute(0,2,1))
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            # embed_proj = self.dynamic_proj(allmark.reshape(allmark.shape[0], -1)).unsqueeze(1).repeat(1,x.shape[2],1)
            embed_proj = self.dynamic_proj(allmark).permute(0,2,1).repeat(1, x.shape[2], 1)
            # print(embed_proj.shape)
            target_proj = embed_proj[:,:,-self.target_window:]
            # res_init, trend_init = torch.cat( (res_init, embed_proj), dim=2 ), torch.cat( (trend_init, embed_proj), dim=2 )
            res = self.model_res(res_init, embed_proj, target_proj)
            trend = self.model_trend(trend_init, embed_proj, target_proj)
            x = res + trend
            # x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            # x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        if self.norm=='allrev':
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'denorm')
            x = x.permute(0,2,1)
        return x