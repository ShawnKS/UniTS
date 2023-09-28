import os
import os
import sys
from typing import List
import numpy as np
import random
import fire
import torch
import transformers
from datasets import load_dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from utils.tools import EarlyStopping, adjust_learning_rate, visual , RevIN
from models.UniTS_t1 import Model as UniModel
from utils.tools import multi_RevIN as RevIN
from torch import optim
import argparse
import pickle
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
# TODO 魔改成custom model
# TODO learning rate schedule
def MSE(pred, true):
    return np.mean((pred[pred!=0] - true[pred!=0]) ** 2)

# from peft import (
#     LoraConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     prepare_model_for_kbit_training,
#     set_peft_model_state_dict,
# )
# from transformers.models.llama.tokenization_llama import LlamaTokenizer
# from ts_llama_pos_enc import LlamaForCausalLM, LlamaModel
# from transformers.models.llama.configuration_llama import LlamaConfig

# from utils.prompter import Prompter

def str2bool(v):
    # 自定义函数用于将字符串转换为布尔值
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='seriesLLaMA')
# basic config
parser.add_argument('--hs', type=int, default=16,
                    help='hidden_size')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu')
parser.add_argument('--nhl', type=int, default=4,
                    help='num_hidden_layers')
parser.add_argument('--ims', type=int, default=32,
                    help='hidden_size of attention layer')
parser.add_argument('--nah', type=int, default=2,
                    help='hidden_size of attention layer')
parser.add_argument('--bs', type=int, default=32,
                    help='batch_size')
parser.add_argument('--sl', type=int, default=384,
                    help='seq_len')
parser.add_argument('--prel', type=int, default=96,
                    help='pred_len')
parser.add_argument('--dataset', default = 'ETTh1',
                    help='dataset')
parser.add_argument('--epoch', type=int, default=100,
                    help='epoch')
parser.add_argument('--encin', type=int, default=7,
                    help='epoch')
parser.add_argument('--ha', type=str, default='relu',
                    help='hidden activation')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--rs', type=lambda x : None if x == 'None' else str(x), nargs='?', default=None,
                    help='rope_scaling')
parser.add_argument('--dr', type=float, default=0.978,
                    help='decay rate')
parser.add_argument('--ast', type=int, default=1,
                    help='accumulate steps')

parser.add_argument('--pem', type=int, default=1, help='use pos embed or not')
parser.add_argument('--dc', type=int, default=0, help='decoder only or encoder-decoder model')
parser.add_argument('--uf', type=int, default=1, help='use freq or not')
parser.add_argument('--ul', type=int, default=0, help='use local featurizer or not')
parser.add_argument('--ug', type=int, default=1, help='use global featurizer or not')
parser.add_argument('--gre', type=int, default=0, help='global residual')
parser.add_argument('--lre', type=int, default=0, help='local residual')
parser.add_argument('--convk', nargs='+', type=int, help='conv kernel (if uselocal)')
parser.add_argument('--dek', type=int, default=25, help='pool kernel, odd needed')
parser.add_argument('--dfre', type=int, default=50, help='dominance frequency')

parser.add_argument('--d', type=int, default=0,
                    help='decompose or not')
parser.add_argument('--pp', type=str, default='end',
                    help='padding_patch')
# add 9.10
parser.add_argument('--upe', type=int, default=1)
parser.add_argument('--uat', type=int, default=1)
# parser.add_argument('--pe', default=8, help='rotate')
parser.add_argument('--norm', default='rms', help='use norm or not')
# parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--pl', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args = parser.parse_args()
def args_to_string(args):
    # 遍历args对象的属性并构建参数字符串
    param_string = ""
    for attr, value in vars(args).items():
        if attr == 'gpu':
            continue
        else:
            param_string += f"{attr}_{value}_"
    # 删除末尾的下划线
    param_string = param_string.rstrip('_')
    return param_string
import copy
parsearg = copy.deepcopy(args)
del parsearg.gpu
del parsearg.dataset
del parsearg.epoch
del parsearg.encin
del parsearg.dc

args_string = args_to_string(parsearg)
path = './UniTS_output/' + str(args.dataset) + '/' + str(args.prel) + '/'
if not os.path.exists(path):
    os.makedirs(path)
print(args_string)
file_name = path + args_string
# file_name+'_train_mse_loss.npy'

if os.path.exists(file_name + '_train_mse_loss.npy'):
    print(f"{file_name} already ran. This run will pass.")
    sys.exit(0)
device_map = "auto"

# tslm_config = LlamaConfig(
#         hidden_size= args.hs ,
#         max_position_embeddings = args.hs,
#         intermediate_size= args.ims,
#         num_hidden_layers= args.nhl,
#         num_attention_heads=args.nah,
#         num_key_value_heads=None,
#         hidden_act="silu",
#         torch_dtype=torch.float16,
#         device_map=device_map,
#     )
# if args.rs is not None:
#     tslm_config.rope_scaling = {}
#     tslm_config.rope_scaling['type'] = args.rs
#     tslm_config.rope_scaling['factor'] = 1

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda:"+str(args.gpu))

# test_model = 
# print('')
# test_model.print_trainable_parameters()
# /data/zqq/llm/llmforts/Time-Series-Library/dataset/electricity
class Args:
    def __init__(self):
        # Default values for the properties
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.root_path = '/data/zqq/llm/llmforts/Time-Series-Library/dataset/ETT-small/'
        self.data_path = args.dataset + '.csv'
        self.model_id = args.dataset + '_' + str(args.prel)
        self.model = '$model_name'
        self.embed = 'timeFN'
        self.frac = '$3'
        self.frac_num = 32
        self.data = args.dataset
        self.dataset = args.dataset
        self.nhl = args.nhl
        self.n_heads = args.nah
        self.ims = args.ims
        self.individual = 0
        self.patch_len = args.pl
        self.stride = args.stride
        self.usepe = args.upe
        self.useatt = args.uat
        
        self.uselocal = args.ul
        self.useglobal = args.ug
        self.gaddres = args.gre
        self.laddres = args.lre
        self.convkernel = args.convk
        # self.convkernel = [24]
        # 必须是奇数？
        self.decomp_kernel = args.dek
        self.dominance_freq = args.dfre

        self.features = 'M'
        self.seq_len = args.sl
        self.norm = args.norm
        self.freq = 'h'
        self.label_len = 48
        self.pred_len = args.prel
        self.target = 'OT'
        self.seasonal_patterns = 'Monthly'
        self.batch_size = args.bs
        self.seed = 2021
        self.d_layers = 1
        self.d = args.d
        self.pembed = args.pem
        self.deconly = args.dc
        self.usefreq = args.uf
        self.num_workers = 8
        self.d_model = '$7'
        self.d_ff = 128
        self.dropout = 0.3
        self.fc_dropout = 0.3
        self.head_dropout = 0
        self.factor = 3
        self.enc_in = 7
        self.padding_patch = args.pp
        self.dec_in = 7
        self.ratio = 1
        self.c_out = 7
        self.des = 'Exp'
        self.itr = 1
# 创建args对象
data_args = Args()
if args.dataset == 'electricity':
    data_args.root_path = '/data/zqq/llm/llmforts/Time-Series-Library/dataset/electricity/'
    data_args.enc_in = 321
if args.dataset == 'traffic':
    data_args.root_path = '/data/zqq/llm/llmforts/Time-Series-Library/dataset/traffic/'
    data_args.enc_in = 862
if args.dataset == 'exchange':
    data_args.root_path = '/data/zqq/llm/llmforts/Time-Series-Library/dataset/exchange_rate/'
    data_args.data_path = 'exchange_rate.csv'
if args.dataset == 'weather':
    data_args.root_path = '/data/zqq/llm/llmforts/Time-Series-Library/dataset/weather/'
    data_args.enc_in = 21
if args.dataset == 'illness':
    data_args.root_path = '/data/zqq/llm/llmforts/Time-Series-Library/dataset/national_illness/'
# parser.add_argument('--pe', default=8, help='rotate')
# parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
from data_provider.data_factory import data_provider
data_set, data_loader = data_provider(data_args, flag='train')
val_data_set, val_data_loader = data_provider(data_args, flag='val')
test_data_set, test_data_loader = data_provider(data_args, flag='test')

series_num = 7
pred_length = args.prel
# hidden_size = tslm_config.hidden_size
use_lora = False

pl = args.pl
stride = args.stride
norm = args.norm
dr = args.dr

pn = int((args.sl - pl)/stride + 1)
# pe=args.pe

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            print(name,num_params)
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )

custom_model = UniModel(configs=data_args).to(device)
print_trainable_parameters(custom_model)
# sys.exit(0)
learning_rate = args.lr  # You can adjust this learning rate according to your task
# 3e-4
# add lr type

model_optim = optim.Adam(custom_model.parameters(), lr=learning_rate)
# model_optim = optim.SGD(custom_model.parameters(), lr=learning_rate, momentum=0.9)

# model.print_trainable_parameters()
# param_frozen_list = [] # should be changed into torch.nn.ParameterList()
# param_active_list = ['mapping.weight', 'post_mapping.weight']

train_loss_total = []
test_loss_total = []
valid_loss_total = []
criterion = MSELoss(reduction='none')
backwindow = int(-1*(pred_length))
total_gradients = []
accumulation_steps = args.ast
model_optim.zero_grad()
norm = 'revin'
for epoch in range(args.epoch):
    train_loss = []
    vali_loss = []
    test_loss =[]
    epoch_gradients = []
    valid_pred, valid_y = [], []
    test_pred, test_y = [], []
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        # model_optim.zero_grad()
        bs, series_num, seq_len = batch_x.size()
        batch_x, batch_x_mark, batch_y, batch_y_mark = batch_x.to(device), batch_x_mark.to(device), batch_y.to(device), batch_y_mark.to(device)
        outputs = custom_model(batch_x,batch_x_mark,batch_y_mark)
        loss = criterion(outputs[:,:,backwindow:], batch_y[:,:,backwindow:])
        loss.mean().backward()
        train_loss = train_loss + list(loss.mean(dim=1).mean(dim=1).cpu().detach().numpy())
        gradients = []
        # for name, param in custom_model.named_parameters():
            # print(name,'---',param.requires_grad)
            # if param.requires_grad and param.grad is not None:
                # gradients.append(param.grad.flatten().cpu().numpy())
        # for name, param in custom_model.llama_model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         gradients.append(param.grad.flatten().cpu().numpy())
        # TODO save gradient
        # epoch_gradients.append(gradients)
        # if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(data_loader)):
            # print('iter',i)
        model_optim.step()
        model_optim.zero_grad()
        # TODO save gradient
    # total_gradients.append(epoch_gradients)
    adjust_learning_rate(model_optim, epoch + 1, learning_rate, dr)
    # print(epoch,learning_rate)
    with torch.no_grad():
        for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_data_loader):
            bs, series_num, seq_len = batch_x.size()
            # print(batch_x_mark)
            batch_x, batch_x_mark, batch_y, batch_y_mark = batch_x.to(device), batch_x_mark.to(device), batch_y.to(device), batch_y_mark.to(device)
            outputs = custom_model(batch_x,batch_x_mark,batch_y_mark)
            loss = criterion(outputs[:,:,backwindow:], batch_y[:,:,backwindow:])
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            # mae, mse, rmse, mape, mspe, per_mae, per_mse = metric(outputs, batch_y)
            # vali_mse.append(mse)
            valid_pred.append(outputs)
            valid_y.append(batch_y)
            # print('mse:{}, mae:{}'.format(mse, mae))
            vali_loss = vali_loss + list(loss.mean(dim=1).mean(dim=1).cpu().detach().numpy())
        for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_data_loader):
            bs, series_num, seq_len = batch_x.size()
            batch_x, batch_x_mark, batch_y, batch_y_mark = batch_x.to(device), batch_x_mark.to(device), batch_y.to(device), batch_y_mark.to(device)
            outputs = custom_model(batch_x,batch_x_mark,batch_y_mark)
            loss = criterion(outputs[:,:,backwindow:], batch_y[:,:,backwindow:])
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            # mae, mse, rmse, mape, mspe, per_mae, per_mse = metric(outputs, batch_y)
            test_pred.append(outputs)
            test_y.append(batch_y)
            # print('mse:{}, mae:{}'.format(mse, mae))s
            test_loss = test_loss + list(loss.mean(dim=1).mean(dim=1).cpu().detach().numpy())
    train_loss_total.append(np.average(train_loss))
    valid_loss_total.append(np.average(vali_loss))
    test_loss_total.append(np.average(test_loss))
    print('epoch',epoch, 'train loss', np.average(train_loss),'vali loss', np.average(vali_loss),'test_loss', np.average(test_loss))
    print('--------------------------------')
# with open(file_name + '_gradient_traj.pkl', 'wb') as f:
#     pickle.dump(total_gradients, f)
np.save(file_name+'_train_mse_loss.npy',train_loss_total)
np.save(file_name+'_test_mse_loss.npy',test_loss_total)
np.save(file_name+'_valid_mse_loss.npy',valid_loss_total)

