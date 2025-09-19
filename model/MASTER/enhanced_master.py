"""
Enhanced MASTER Model with Multi-Task Learning Support
Integrates Polygon.io data, news sentiment, and trading schemes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from base_model import SequenceModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for different prediction tasks
    """
    def __init__(self, d_model, num_tasks, task_weights=None):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_weights = task_weights if task_weights is not None else [1.0] * num_tasks
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Linear(d_model // 4, 1) for _ in range(num_tasks)
        ])
    
    def forward(self, x):
        # Shared representation
        shared_repr = self.shared_layers(x)
        
        # Task-specific predictions
        task_predictions = []
        for i, head in enumerate(self.task_heads):
            pred = head(shared_repr)
            task_predictions.append(pred)
        
        return torch.cat(task_predictions, dim=-1)


class EnhancedMASTER(nn.Module):
    """
    Enhanced MASTER model with multi-task learning support
    """
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, 
                 gate_input_start_index, gate_input_end_index, beta, num_tasks=8, task_weights=None):
        super(EnhancedMASTER, self).__init__()
        
        # Market gate
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        
        # Feature processing layers
        self.feature_layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
        )
        
        # Attention layers
        self.temporal_attention = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        self.spatial_attention = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.temporal_aggregation = TemporalAttention(d_model=d_model)
        
        # Multi-task prediction head
        self.multi_task_head = MultiTaskHead(d_model, num_tasks, task_weights)
        
        # Task weights for loss calculation
        self.register_buffer('task_weights', torch.tensor(task_weights if task_weights else [1.0] * num_tasks))
    
    def forward(self, x):
        # Split input into features and gate input
        src = x[:, :, :self.gate_input_start_index]  # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        
        # Apply feature gate
        gate_weights = self.feature_gate(gate_input)
        src = src * torch.unsqueeze(gate_weights, dim=1)
        
        # Process features
        src = self.feature_layers(src)
        
        # Apply attention layers
        src = self.temporal_attention(src)
        src = self.spatial_attention(src)
        src = self.temporal_aggregation(src)
        
        # Multi-task predictions
        predictions = self.multi_task_head(src)
        
        return predictions


class EnhancedMASTERModel(SequenceModel):
    """
    Enhanced MASTER model with multi-task learning support
    """
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
                 T_dropout_rate, S_dropout_rate, beta, num_tasks=8, task_weights=None, **kwargs):
        super(EnhancedMASTERModel, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.d_feat = d_feat
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        self.num_tasks = num_tasks
        self.task_weights = task_weights if task_weights else [1.0] * num_tasks
        
        self.init_model()
    
    def init_model(self):
        self.model = EnhancedMASTER(
            d_feat=self.d_feat, 
            d_model=self.d_model, 
            t_nhead=self.t_nhead, 
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate, 
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index, 
            beta=self.beta,
            num_tasks=self.num_tasks,
            task_weights=self.task_weights
        )
        super(EnhancedMASTERModel, self).init_model()
    
    def multi_task_loss_fn(self, pred, label):
        """
        Multi-task loss function with weighted tasks
        
        Args:
            pred: Predictions of shape (N, num_tasks)
            label: Labels of shape (N, num_tasks)
            
        Returns:
            Weighted multi-task loss
        """
        # Mask for valid labels
        mask = ~torch.isnan(label)
        
        # Calculate loss for each task
        task_losses = []
        for i in range(self.num_tasks):
            if mask[:, i].any():
                task_pred = pred[:, i]
                task_label = label[:, i]
                task_mask = mask[:, i]
                
                # MSE loss for each task
                task_loss = F.mse_loss(task_pred[task_mask], task_label[task_mask])
                task_losses.append(task_loss)
            else:
                task_losses.append(torch.tensor(0.0, device=pred.device))
        
        # Weighted combination
        total_loss = 0.0
        for i, (loss, weight) in enumerate(zip(task_losses, self.task_weights)):
            total_loss += weight * loss
        
        return total_loss
    
    def train_epoch(self, data_loader):
        """Enhanced training epoch with multi-task loss"""
        self.model.train()
        losses = []
        
        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            
            # Extract features and labels
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1:].to(self.device)  # Primary return label
            
            # For multi-task learning, we need all task labels
            # This assumes the last columns contain all task labels
            if data.shape[-1] > 1:
                # Multi-task labels
                multi_task_labels = data[:, -1, -self.num_tasks:].to(self.device)
            else:
                # Single task - replicate the label
                multi_task_labels = label.repeat(1, self.num_tasks)
            
            # Get predictions
            pred = self.model(feature.float())
            
            # Calculate multi-task loss
            loss = self.multi_task_loss_fn(pred, multi_task_labels)
            losses.append(loss.item())
            
            # Backward pass
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
        
        return float(np.mean(losses))
    
    def predict(self, dl_test):
        """Enhanced prediction with multi-task outputs"""
        if self.fitted < 0:
            raise ValueError("model is not fitted yet!")
        else:
            print('Epoch:', self.fitted)
        
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)
        
        all_predictions = []
        ic_metrics = []
        
        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]  # Primary return label
            
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            
            # Extract primary prediction (return)
            primary_pred = pred[:, 0]  # First task is primary return prediction
            all_predictions.append(primary_pred.ravel())
            
            # Calculate IC for primary task
            daily_ic, daily_ric = self.calc_ic(primary_pred, label.detach().numpy())
            ic_metrics.append((daily_ic, daily_ric))
        
        predictions = pd.Series(np.concatenate(all_predictions), index=dl_test.get_index())
        
        # Calculate metrics
        ic_values = [ic for ic, _ in ic_metrics]
        ric_values = [ric for _, ric in ic_metrics]
        
        metrics = {
            'IC': np.mean(ic_values),
            'ICIR': np.mean(ic_values) / np.std(ic_values) if np.std(ic_values) > 0 else 0,
            'RIC': np.mean(ric_values),
            'RICIR': np.mean(ric_values) / np.std(ric_values) if np.std(ric_values) > 0 else 0
        }
        
        return predictions, metrics
    
    def calc_ic(self, pred, label):
        """Calculate Information Coefficient"""
        df = pd.DataFrame({'pred': pred, 'label': label})
        ic = df['pred'].corr(df['label'])
        ric = df['pred'].corr(df['label'], method='spearman')
        return ic, ric


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'd_feat': 200,  # Increased for more features
        'd_model': 256,
        't_nhead': 4,
        's_nhead': 2,
        'dropout': 0.5,
        'gate_input_start_index': 200,
        'gate_input_end_index': 263,
        'beta': 5,
        'num_tasks': 8,  # 1 return + 7 trading schemes
        'task_weights': [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],  # Weight primary task higher
        'n_epochs': 10,
        'lr': 1e-5,
        'GPU': 0,
        'train_stop_loss_thred': 0.95
    }
    
    # Initialize enhanced model
    model = EnhancedMASTERModel(**config)
    print("Enhanced MASTER model initialized successfully!")
    print(f"Number of tasks: {model.num_tasks}")
    print(f"Task weights: {model.task_weights}")
