      
      
      
      
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from layers.RevIN import RevIN
import logging
from scipy.fftpack import fft, ifft
from torch.nn.utils import weight_norm



# 配置日志记录
logging.basicConfig(filename='model_debug.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')





class series_decomp(nn.Module):
    """
    Series decomposition block using AvgPool1d for moving average.
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # 通过在两端进行填充来模拟 'same' padding 的效果
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x_padded = torch.cat([front, x, end], dim=2)
        
        # 计算滑动平均
        moving_mean = self.moving_avg(x_padded)
        # 计算残差
        res = x - moving_mean
        return res, moving_mean




class block_model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, input_channels, input_len, out_len, individual):
        super(block_model, self).__init__()
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len
        self.individual = individual

        if self.individual:
            self.Linear_channel = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_channel.append(nn.Linear(self.input_len, self.out_len))
        else:
            self.Linear_channel = nn.Linear(self.input_len, self.out_len)
        self.ln = nn.LayerNorm(out_len)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [Batch, Channel, Input length]
        if self.individual:
            output = torch.zeros([x.size(0),x.size(1),self.out_len],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,i,:] = self.Linear_channel[i](x[:,i,:])
        else:
            output = self.Linear_channel(x)
        #output = self.ln(output)
        #output = self.relu(output)
        return output # [Batch, Channel, Output length]

### NEW CODE START: MLPBlock class ###
class MLPBlock(nn.Module):
    """
    A simple MLP block: Linear -> GELU -> Dropout
    Input shape: [..., dim]
    Output shape: [..., dim] (the same as input)
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x shape is (..., dim)
        out = self.linear(x)
        out = self.act(out)
        out = self.drop(out)
        return out
### NEW CODE END ###





class UpDownSamplingLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UpDownSamplingLayer, self).__init__()
        self.layer = nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            nn.GELU(),
            torch.nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.layer(x)

class PatchEncoder(nn.Module):
    def __init__(self, input_channels, input_len, out_len, individual,
        patch_len, stride, padding_patch,
        shared_embedding=True, head_dropout = 0):
        
        super(PatchEncoder, self).__init__()  # 注意这里修改了父类引用
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len
        self.individual = individual
        self.d_model = 128
        
        context_window = input_len
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        patch_num = int((context_window - patch_len)/stride + 1)
        
        if padding_patch == 'end': 
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
    
    def forward(self, x):
        # x: [Batch, Channel, Input length]
        
        # do patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # X: [Batch, Channel, Num_patches, Patch_len]
        x = x.permute(0,1,3,2)  # X: [Batch, Channel, Patch_len, Num_patches]
        
        return x  # 返回 [Batch, Channel, Patch_len, Num_patches]

class PatchDecoder(nn.Module):
    def __init__(self, input_channels, input_len, out_len, individual,
        patch_len, stride, padding_patch,
        shared_embedding=True, head_dropout = 0):
        
        super(PatchDecoder, self).__init__()  # 注意这里修改了父类引用
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len
        self.individual = individual
        self.d_model = 128
        
        context_window = input_len
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        patch_num = int((context_window - patch_len)/stride + 1)
        
        if padding_patch == 'end': 
            patch_num += 1
            
        # Backbone 
        self.backbone = FC2Encoder(c_in = self.channels, patch_len = patch_len, d_model = self.d_model,
                                  shared_embedding = shared_embedding)
        # Head
        self.head_nf = self.d_model * patch_num
        self.n_vars = self.channels
        self.individual = individual
        
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, out_len, head_dropout=head_dropout)
    
    def forward(self, x):
        # x: [Batch, Channel, Patch_len, Num_patches]
        
        # model
        x = self.backbone(x) 
        x = self.head(x) + self.head(x)  # 注意这里与原始模型保持一致
        
        return x  # [Batch, Channel, Output length]

class PatchBlockModel(nn.Module):
    def __init__(self, input_channels, input_len, out_len, individual,
        patch_len, stride, padding_patch,
        shared_embedding=True, head_dropout = 0):
        
        super(PatchBlockModel, self).__init__()
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len
        self.individual = individual
        self.d_model=128
        
        
        context_window = input_len

        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        
        
        patch_num = int((context_window - patch_len)/stride + 1)
        
        if padding_patch == 'end': 
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
            
        # Backbone 
        self.backbone = FC2Encoder(c_in = self.channels, patch_len = patch_len, d_model = self.d_model,
                                  shared_embedding = shared_embedding)
        # Head
        self.head_nf = self.d_model * patch_num
        self.n_vars = self.channels
        self.individual = individual
        

        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, out_len, head_dropout=head_dropout)
        


    def forward(self, x):
        # x: [Batch, Channel, Input length]

        
        # do patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)        #X:  [Batch, Channel, Num_patches, Patch_len]，         
        x = x.permute(0,1,3,2)    #  X : [Batch, Channel, Patch_len, Num_patches]，
        

        # model
        x = self.backbone(x) 
        x = self.head(x) + self.head(x)   

    
        return x  # [Batch, Channel, Output length]

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):

        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        

        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
class FC2Encoder(nn.Module):
    def __init__(self, c_in, patch_len,  d_model=128, shared_embedding=True, **kwargs):
        super().__init__()
        self.n_vars = c_in
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        
        self.act = nn.ReLU(inplace=True)

        
        
        if not shared_embedding: 
            self.W_P1 = nn.ModuleList()
            self.W_P2 = nn.ModuleList()
            for _ in range(self.n_vars): 
                self.W_P1.append(nn.Linear(patch_len, d_model))
                self.W_P2.append(nn.Linear(d_model, d_model))
        else:
            self.W_P1 = nn.Linear(patch_len, d_model)      
            self.W_P2 = nn.Linear(d_model, d_model)      

    def forward(self, x):          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        # [128, 7, 12, 56]
        """
        x = x.permute(0,3,1,2)
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P1[i](x[:,:,i,:])
                x_out.append(z)
                z = self.act(z)
                z = self.W_P2[i](z) # ??
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P1(x)                                                     
            x = self.act(x)
            x = self.W_P2(x)  
            
                                                        
        x = x.transpose(1,2)                                                     
        x = x.permute(0,1,3,2)
        return x

class MSTF(nn.Module):
    def __init__(self, input_channels, out_len, num_scales):
        super(MSTF, self).__init__()
        self.input_channels = input_channels
        self.out_len = out_len
        self.num_scales = num_scales    
        # 定义可学习的权重矩阵，用于加权不同层的预测结果
        self.weights = nn.Parameter(torch.ones(num_scales))  # 初始化为全1的权重，可以根据需要调整

    def forward(self, preds):
        """
        preds: List of tensors with shape [Batch, Channels, Seq_len] for each scale
        """
        # 将 preds 转换为张量并加权求和
        preds = torch.stack(preds, dim=-1)  # [Batch, Channels, Seq_len, M]
        # 打印权重矩阵的值
        #logging.info(f"Weight matrix values: {self.weights.detach().cpu().numpy()}")
        # 对每个预测结果进行加权求和
        weighted_preds = preds * self.weights  # [Batch, Channels, Seq_len, M] * [M]
        final_prediction = torch.sum(weighted_preds, dim=-1)  # [Batch, Channels, Seq_len]
        
        return final_prediction
   

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()

        self.input_channels = configs.enc_in
        self.input_len = configs.seq_len
        self.out_len = configs.pred_len
        self.individual = configs.individual
        self.stage_num = configs.stage_num
        self.stage_pool_kernel = configs.stage_pool_kernel
        self.stage_pool_stride = configs.stage_pool_stride
        self.stage_pool_padding = configs.stage_pool_padding   
        
        self.freq_downsampling_percentage = configs.freq_downsampling_percentage
        self.series_decomp_layer = series_decomp(kernel_size=25)  # 这里的 kernel_size 根据你的需要设置

        
        #patch
        initial_patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        context_window = configs.seq_len
        shared_embedding = configs.shared_embedding
        head_dropout = configs.head_dropout
        

        self.revin_layer = RevIN(self.input_channels, affine=True, subtract_last=False)

        len_in = self.input_len
        len_out = self.out_len
        down_in = [len_in]
        down_out = [len_out]
        i = 0
        while i < self.stage_num - 1:
            linear_in = int((len_in + 2 * self.stage_pool_padding - self.stage_pool_kernel) / self.stage_pool_stride + 1)
            linear_out = int((len_out + 2 * self.stage_pool_padding - self.stage_pool_kernel) / self.stage_pool_stride + 1)
            down_in.append(linear_in)
            down_out.append(linear_out)
            len_in = linear_in
            len_out = linear_out
            i = i + 1

        # 最大池化层和特征提取层
        self.Maxpools = nn.ModuleList() 
        self.down_blocks = nn.ModuleList()
        for in_len, out_len in zip(down_in, down_out):
            logging.info(f"Creating down_block with in_len: {in_len}, out_len: {out_len}")
            self.down_blocks.append(block_model(self.input_channels, in_len, out_len, self.individual))
            self.Maxpools.append(nn.AvgPool1d(kernel_size=self.stage_pool_kernel, stride=self.stage_pool_stride, padding=self.stage_pool_padding))
            #self.Maxpools.append(block_model(self.input_channels, in_len, out_len, self.individual))
            



        # 获取最后一层 down_block 的 out_len
        last_down_block_out_len = down_out[-1]



            
            
        # 特征融合层和预测层
        self.down_r_blocks = nn.ModuleList()
        self.up_r_blocks = nn.ModuleList()
        self.mix_r_blocks = nn.ModuleList()
        self.mix_r_blocks1 = nn.ModuleList()
        self.mix_r_blocks2 = nn.ModuleList()
        
        
        self.pred_layers = nn.ModuleList()
        self.seasonal_pred_layers = nn.ModuleList()
        self.trend_pred_layers = nn.ModuleList()
        
        
        self.pred_layers1 = nn.ModuleList()
        self.seasonal_pred_layers1 = nn.ModuleList()
        self.trend_pred_layers1 = nn.ModuleList()
        
        self.pred_layers2 = nn.ModuleList()
        self.seasonal_pred_layers2 = nn.ModuleList()
        self.trend_pred_layers2 = nn.ModuleList()
        

        len_down_out = len(down_out)
        
        
        
        
        # 将 last_down_block_out_len 放在 in_lens 的第一位
        in_lens = [last_down_block_out_len]
        # 计算其余层的 in_len 并存储到列表中
        for i in range(len_down_out - 1):
            in_len = down_out[len_down_out - i - 1] + down_out[len_down_out - i - 2]
            in_lens.append(in_len)
        # 计算比例，最后一项比例设为1（作为基准）
        ratios = [in_len / in_lens[-1] for in_len in in_lens]  # 计算所有层的比例
        ratios[-1] = 1.0  # 将最后一项的比例明确设为1
        # 输出比例日志，便于调试
        logging.info(f"Computed in_lens: {in_lens}")
        logging.info(f"Computed ratios: {ratios}")
        
        
        
        # 计算每一层的 patch_len 并存储在列表中
        patch_lens = [initial_patch_len for _ in range(len(ratios))]
        logging.info(f"Initialized patch_lens with initial_patch_len: {patch_lens}")
        
        
        for i, in_len in enumerate(down_out):
            used_patch_len = patch_lens[0]  # 选择当前的 patch_len
            # 创建并添加 PatchBlockModel 到 pred_layers, seasonal_pred_layers 和 trend_pred_layers
            logging.info(f"Creating pred_layers[{i}] with in_len: {in_len}, out_len: {self.out_len}")
            # 这里 mix_r_blocks 需要反向遍历 down_out，所以使用 reversed(down_out)
            reverse_in_len = list(reversed(down_out))[i]
            self.mix_r_blocks.append(block_model(self.input_channels, reverse_in_len, reverse_in_len, self.individual))
            self.mix_r_blocks1.append(block_model(self.input_channels, reverse_in_len, reverse_in_len, self.individual))
            self.mix_r_blocks2.append(block_model(self.input_channels, reverse_in_len, reverse_in_len, self.individual))
           
           
            # ===== 注意：此处第三个参数改为 self.out_len =====
            self.pred_layers.append(PatchBlockModel(self.input_channels, in_len, self.out_len, self.individual,
                                                    used_patch_len, stride, padding_patch,
                                                    shared_embedding, head_dropout))
            self.seasonal_pred_layers.append(PatchBlockModel(self.input_channels, in_len, self.out_len, self.individual,
                                                            used_patch_len, stride, padding_patch,
                                                            shared_embedding, head_dropout))
            self.trend_pred_layers.append(PatchBlockModel(self.input_channels, in_len, self.out_len, self.individual,
                                                        used_patch_len, stride, padding_patch,
                                                        shared_embedding, head_dropout))
            
            
            self.pred_layers1.append(PatchBlockModel(self.input_channels, in_len, self.out_len, self.individual,
                                                    used_patch_len, stride, padding_patch,
                                                    shared_embedding, head_dropout))
            self.seasonal_pred_layers1.append(PatchBlockModel(self.input_channels, in_len, self.out_len, self.individual,
                                                            used_patch_len, stride, padding_patch,
                                                            shared_embedding, head_dropout))
            self.trend_pred_layers1.append(PatchBlockModel(self.input_channels, in_len, self.out_len, self.individual,
                                                        used_patch_len, stride, padding_patch,
                                                        shared_embedding, head_dropout))
            
            
            self.pred_layers2.append(PatchBlockModel(self.input_channels, in_len, self.out_len, self.individual,
                                                    used_patch_len, stride, padding_patch,
                                                    shared_embedding, head_dropout))
            self.seasonal_pred_layers2.append(PatchBlockModel(self.input_channels, in_len, self.out_len, self.individual,
                                                            used_patch_len, stride, padding_patch,
                                                            shared_embedding, head_dropout))
            self.trend_pred_layers2.append(PatchBlockModel(self.input_channels, in_len, self.out_len, self.individual,
                                                        used_patch_len, stride, padding_patch,
                                                        shared_embedding, head_dropout))
          
        # 反向采样信息恢复层，跳过最后一个元素,up->low
        for i in range(len(down_out) - 1):
            # 当前层的 in_len
            in_len = down_out[i]
            # 下一层的 out_len
            out_len = down_out[i + 1]
            self.down_r_blocks.append(UpDownSamplingLayer(in_len, out_len))
            logging.info(f"Creating up_r_block with in_len: {in_len}, out_len: {out_len}")
            
            
        # 正向采样信息恢复层，low->up，遍历方向修正
        for i in range(len(down_out) - 1, 0, -1):
            # 当前层的 in_len
            in_len = down_out[i]
            # 上一层的 out_len
            out_len = down_out[i - 1]  # out_len 是更浅层的
            self.up_r_blocks.append(UpDownSamplingLayer(in_len, out_len))

        # 定义 MSTF 模块
        self.mstf = MSTF(self.input_channels, self.out_len, self.stage_num) #----------------------------------
        
        
        #-------------------############--------新增部分--------------############-------------#
    
        
        self.configs = configs
        # 1) 预先计算多尺度下采样后的序列长度 (类似原来down_in的逻辑)
        #    用于对 current_sequence (即 [B, C, L]) 做 stage_num 层下采样
        self.ms_down_in = []
        cur_len = self.out_len           # ===== 修改：以 out_len 为初始长度 =====
        self.ms_down_in.append(cur_len)
        for i in range(self.stage_num - 1):
            next_len = int((cur_len + 2*self.stage_pool_padding - self.stage_pool_kernel)
                        / self.stage_pool_stride + 1)
            self.ms_down_in.append(next_len)
            cur_len = next_len

        # 2) 下采样模块(AvgPool1d) ，与之前同参
        self.downsamplers_extra = nn.ModuleList()
        for i in range(self.stage_num - 1):
            self.downsamplers_extra.append(
                nn.AvgPool1d(
                    kernel_size=self.stage_pool_kernel,
                    stride=self.stage_pool_stride,
                    padding=self.stage_pool_padding
                )
            )

        # 3) 为各层(除最深层外) 定义新的PatchEncoder
        #    注意 input_len, out_len 都“事先写死”在此
        self.patch_encoders_extra = nn.ModuleList()
        for i in range(self.stage_num - 1):
            encoder = PatchEncoder(
                input_channels=self.input_channels,
                input_len=self.ms_down_in[i],
                out_len=self.ms_down_in[i],       # 这里写死跟 input_len 相同
                individual=self.individual,
                patch_len=self.configs.patch_len,
                stride=self.configs.stride,
                padding_patch=self.configs.padding_patch,
                shared_embedding=self.configs.shared_embedding,
                head_dropout=self.configs.head_dropout
            )
            self.patch_encoders_extra.append(encoder)

        # 4) 为每个stage定义一个新的PatchDecoder，用来输出最终预测
        self.patch_decoders_extra = nn.ModuleList()
        for i in range(self.stage_num):
            decoder = PatchDecoder(
                input_channels=self.input_channels,
                input_len=self.ms_down_in[i],  # 对应当前stage序列长度
                out_len=self.out_len,          # 最终要预测 self.out_len
                individual=self.individual,
                patch_len=self.configs.patch_len,
                stride=self.configs.stride,
                padding_patch=self.configs.padding_patch,
                shared_embedding=self.configs.shared_embedding,
                head_dropout=self.configs.head_dropout
            )
            self.patch_decoders_extra.append(decoder)

        # 5) 定义一个新的 MSTF，用于本多尺度流程的最终融合
        self.mstf_extra = MSTF(self.input_channels, self.out_len, self.stage_num)

        # 6) 定义一个辅助函数做1D线性插值
        def upsample_1d_linear(x, target_len):
            """
            x: [B, C, L] -> 线性插值到 [B, C, target_len]
            """
            return F.interpolate(x, size=target_len, mode='linear', align_corners=False)

        # 让它可以在 forward 中被调用
        self.upsample_1d_linear = upsample_1d_linear
        
        
        # -------------------############--------新增部分--------------############-------------#

        ### NEW CODE START: define mlp_block1s & mlp_block2s for each stage ###
        self.mlp_block1s = nn.ModuleList()
        self.mlp_block2s = nn.ModuleList()

        for i_stage in range(self.stage_num):
            # block1 负责在 shape=[B, C, out_len] 中，对“最后一维 out_len”做线性映射
            self.mlp_block1s.append(
                MLPBlock(dim=self.out_len, dropout=self.configs.head_dropout)  # 你可改成别的超参
            )

            # block2 负责在 shape=[B, out_len, C] 中，对“最后一维 C”做线性映射
            self.mlp_block2s.append(
                MLPBlock(dim=self.input_channels, dropout=self.configs.head_dropout)
            )
        ### NEW CODE END ###

        
                
        #-------------------############------------------------------############-------------#
  
    def forward(self, x):
        # 对输入数据进行归一化
        x = self.revin_layer(x, 'norm')  # [Batch, Seq_len, Channels]
        #logging.info(f"Input shape after RevIN and permute: {x.shape}")
        
        x1 = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        logging.info(f"x1 shape after permute: {x1.shape}")
        
        e_out = []
        i = 0
        
        # 下采样特征提取
        for down_block in self.down_blocks:
            e_out_block = down_block(x1)  # x: [Batch, Channel, Input length]  → → →
            e_out.append(e_out_block)
            x1 = self.Maxpools[i](x1)    #                                     ↓ ↓ ↓
            i = i + 1
        

        preds = []
        e_last_bwd_list = []  # 反向采样的 e_last 列表
        e_last_fwd_list = []  # 正向采样的 e_last 列表
        
    
        # 正向采样循环
        e_last = e_out[self.stage_num - 1]  # 初始化 e_last 为最后一层输出
        e_last_fwd_list.append(e_last) 
        for i in range(self.stage_num - 1):
            feature1 = e_out[self.stage_num - i - 2]  # 来自较浅层的时序特征
            feature2 = e_last  # 当前的 e_last
            e_last = feature1 + self.up_r_blocks[i](feature2)
            e_last_fwd_list.append(e_last)


        
        
        # 遍历每一层，将正向和反向的 e_last 相加并预测
        for i in range(self.stage_num):
            fused_e_last2 = e_last_fwd_list[i]   #反向特征
            fused_e_last2 = self.mix_r_blocks2[i](fused_e_last2)
            
            season2, trend2 = self.series_decomp_layer(fused_e_last2)
            
            reverse_index = len(self.pred_layers) - 1 - i
            
            # ===== 注：下列 PatchBlockModel 均已输出 self.out_len =====
            season_pred2 = self.seasonal_pred_layers2[reverse_index](season2)
            trend_pred2 = self.trend_pred_layers2[reverse_index](trend2)
            
            pred = season_pred2 + trend_pred2
            preds.append(pred)

        
        # 最终预测结果
        current_sequence = self.mstf(preds)  # [Batch, Channels, Out_len]
        
        
        #-------------------############--------新增部分--------------############-------------#
        # 这里 current_sequence 的形状是 [Batch, Channels, Out_len]
        # 命名： B, C, L
        B, C, L = current_sequence.shape

        ########################################
        # 1) 多层下采样 => stages
        ########################################
        stages = [current_sequence]  # stage1 = 原序列
        for i in range(self.stage_num - 1):
            x_in = stages[-1]                          # [B, C, length_i]
            bc = x_in.reshape(B*C, 1, -1)              # => [B*C,1,length_i]
            ds = self.downsamplers_extra[i](bc)        # => [B*C,1,length_(i+1)]
            ds = ds.reshape(B, C, -1)                  # => [B, C, length_(i+1)]
            stages.append(ds)

        ########################################
        # 2) unfold切片 + (PatchEncoder) => patched_stages, weighed_stages
        ########################################
        patched_stages = []
        weighed_stages = []
        for i in range(self.stage_num):
            stage_x = stages[i]  # [B, C, length_i]

            # (A) unfold => [B, C, num_patches, patch_len]
            patch_len = self.configs.patch_len
            stride_   = self.configs.stride
            if self.configs.padding_patch == 'end':
                pad_stage_x = F.pad(stage_x, (0, stride_), mode='replicate')
            x_unfold = pad_stage_x.unfold(dimension=-1, size=patch_len, step=stride_)

            patched_stages.append(x_unfold)  # [B, C, num_patches, patch_len]

            # (B) 除最深层外，都再过一次PatchEncoder
            if i < self.stage_num - 1:
                # PatchEncoder forward输入是 [B, C, input_len]
                weighed_x = self.patch_encoders_extra[i](stage_x)  
                # weighed_x => [B, C, patch_len, num_patches], 需转到 [B, C, num_patches, patch_len]
                weighed_x = weighed_x.permute(0,1,3,2)
                weighed_stages.append(weighed_x)
            else:
                weighed_stages.append(None)  # 最深层不需要

        ########################################
        # 3) 自底向上，上采样线性插值 + 融合
        ########################################
        for i in reversed(range(self.stage_num - 1)):
            deeper_patched = patched_stages[i+1]  # [B, C, np_d, p_d]
            deeper_weighed = weighed_stages[i+1]  # 可能是 None
            cur_patched    = patched_stages[i]    # [B, C, np_c, p_c]
            cur_weighed    = weighed_stages[i]    # 可能是 None(若 i=最深层)

            # (A) 先把 deeper_patched 还原回 [B, C, length_d]
            B_, C_, npd, pld = deeper_patched.shape
            length_d = npd * pld  # 简易：假设无 overlap
            deeper_seq = deeper_patched.reshape(B_, C_, length_d)

            # (B) 上采样到 stage i 的长度 length_c = np_c * p_c
            B_, C_, npc, plc = cur_patched.shape
            length_c = npc * plc
            upsampled = self.upsample_1d_linear(deeper_seq, length_c)  # => [B, C, length_c]

            # (C) reshape回 [B, C, npc, plc]
            upsampled = upsampled.reshape(B_, C_, npc, plc)


            # (D) 与 cur_weighed 相加(如果有)，再与 cur_patched 相加
            if cur_weighed is not None:
                upsampled = upsampled + cur_weighed
            upsampled = upsampled + cur_patched

            # 更新
            patched_stages[i] = upsampled


        ########################################
        # 4) 每个stage都走一个PatchDecoder => predictions
        ########################################
        predictions = []
        for i in range(self.stage_num):
            # patched_stages[i]: [B, C, np_i, p_i]
            x_in = patched_stages[i].permute(0,1,3,2)  # => [B, C, patch_len, num_patches]
            pred_i = self.patch_decoders_extra[i](x_in)  # => [B, C, self.out_len]

            ### NEW CODE START: 通道间信息增强模块 ###
            # 1) pred_i_1 = pred_i + mlp_block1(pred_i)  (针对 out_len 维度)
            B_, C_, L_ = pred_i.shape  # L_ should be self.out_len
            # reshape [B, C, L] -> [B*C, L],  apply mlp_block1s[i], then reshape back
            pred_i_reshaped = pred_i.reshape(B_*C_, L_)
            out1 = self.mlp_block1s[i](pred_i_reshaped)   # => [B*C, L_]
            out1 = out1.reshape(B_, C_, L_)
            pred_i_1 = pred_i + out1

            # 2) pred_i_1_R => mlp_block2 => pred_i_2
            pred_i_1_R = pred_i_1.permute(0, 2, 1)  # [B, L_, C_]
            # reshape => [B*L_, C_], apply block2 => [B*L_, C_]
            pred_i_1_R_reshaped = pred_i_1_R.reshape(B_*L_, C_)
            out2 = self.mlp_block2s[i](pred_i_1_R_reshaped)
            out2 = out2.reshape(B_, L_, C_)
            pred_i_2 = out2.permute(0, 2, 1)       # => [B, C_, L_]

            # 3) pred_i_x = pred_i_2 * pred_i_1 (逐元素点乘)
            pred_i_x = pred_i_2 * pred_i_1         # [B, C_, L_]

            # 4) final_output = pred_i_x + pred_i
            pred_i_enhanced = pred_i_x + pred_i
            ### NEW CODE END ###

            # 把增强后的pred_i丢进predictions
            predictions.append(pred_i_enhanced)

        ########################################
        # 5) 多尺度融合 => final_prediction
        ########################################
        final_prediction = self.mstf_extra(predictions)  # => [B, C, self.out_len]
                
        
        
        
        #-------------------############------------------------------############-------------#

        final_prediction = final_prediction.permute(0, 2, 1)  # [Batch, Out_len, Channels]
        final_prediction = self.revin_layer(final_prediction, 'denorm')

        
        return final_prediction




    

    

    

    