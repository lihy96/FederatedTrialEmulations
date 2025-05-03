import os.path as osp
import pandas as pd
from sklearn.base import BaseEstimator
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import KFold
import math
import copy

device = torch.device("cpu")
logdir = './'

def compute_hessian(loss, params):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads = torch.cat([g.view(-1) for g in grads])
    hessian = torch.zeros(len(grads), len(grads))
    for i, grad in enumerate(grads):
        hess_grad = torch.autograd.grad(grad, params, retain_graph=True)
        hess_grad = torch.cat([g.contiguous().view(-1) for g in hess_grad])
        hessian[i] = hess_grad
    return hessian, grads

def split_df(df, train_test_ratio):
    len_df = len(df)
    shuffled_indices = np.random.permutation(np.arange(len_df))
    train_val_idx, test_idx = shuffled_indices[:int(len_df * train_test_ratio)], shuffled_indices[int(len_df * train_test_ratio):]
    return train_val_idx, test_idx

def calculate_mean_and_std(dict_list):
    mean_dict = {}
    std_dict = {}

    keys = dict_list[0].keys()

    for key in keys:
        values = [d[key] for d in dict_list]
        
        if isinstance(values[0], (int, float)):
            mean_dict[key] = np.mean(values)
            std_dict[key] = np.std(values)
        
        elif isinstance(values[0], list):
            values_array = np.array(values)
            
            mean_dict[key] = np.mean(values_array, axis=0).tolist()
            std_dict[key] = np.std(values_array, axis=0).tolist()

    return mean_dict, std_dict

def cal_weights(golds_treatment, probs, stabilized=True):
    ones_idx, zeros_idx = torch.nonzero(golds_treatment == 1).reshape(-1), torch.nonzero(golds_treatment == 0).reshape(-1)
    p_T = len(ones_idx) / (len(ones_idx) + len(zeros_idx))

    if stabilized:
        treated_w, controlled_w = p_T / (probs[ones_idx] + 1e-5), (1 - p_T) / (1. - probs[zeros_idx] + 1e-5)
    else:
        treated_w, controlled_w = 1. / (probs[ones_idx] + 1e-5), 1. / (1. - probs[zeros_idx] + 1e-5)
    treated_w = torch.clamp(treated_w, min=1e-06, max=1e2)
    controlled_w = torch.clamp(controlled_w, min=1e-06, max=1e2)

    weights = torch.zeros(len(golds_treatment))
    weights = weights.to(device)
    weights[ones_idx] = treated_w
    weights[zeros_idx] = controlled_w

    treated_w, controlled_w = torch.reshape(treated_w, (len(treated_w), 1)), torch.reshape(controlled_w, (len(controlled_w), 1))
    return treated_w, controlled_w, weights

def weighted_smd(X, T, weights=None, return_numerator=False):
    if weights is None:
        weights = torch.ones(T.shape)
    T = T.long()
    group1 = X[T == 1]
    group0 = X[T == 0]
    weight1 = weights[T == 1]
    weight0 = weights[T == 0]
    mean1 = torch.sum(group1 * weight1[:, None], dim=0) / torch.sum(weight1)
    mean0 = torch.sum(group0 * weight0[:, None], dim=0) / torch.sum(weight0)
    if return_numerator:
        return torch.abs(mean1 - mean0)
    var1 = torch.sum(weight1[:, None] * (group1 - mean1)**2, dim=0) / torch.sum(weight1)
    var0 = torch.sum(weight0[:, None] * (group0 - mean0)**2, dim=0) / torch.sum(weight0)
    pooled_std = torch.sqrt((var1 + var0) / 2) + 1e-6
    smd = (mean1 - mean0) / pooled_std 
    return torch.abs(smd)

class LogReg(nn.Module):
    def __init__(self, input_dim):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(input_dim, 2)  
    def forward(self, x):
        return self.linear(x)

class FeatureSelector(torch.nn.Module):
    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
    
        self.mu = torch.nn.Parameter(1.0 * torch.ones(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size()) * 0.001
        self.sigma = sigma
        self.input_dim = input_dim

    def forward(self, prev_x):
        if prev_x.shape[-1] == self.input_dim:
            z = self.mu + self.sigma * self.noise.normal_() * self.training
            stochastic_gate = self.hard_sigmoid(z)
            new_x = prev_x * stochastic_gate
            return new_x
        elif prev_x.shape[-1] == self.input_dim + 1:
            z = self.mu + self.sigma * self.noise.normal_() * self.training
            stochastic_gate = self.hard_sigmoid(z)
            long_stochastic_gate = torch.cat((torch.tensor([1.], requires_grad=False), stochastic_gate))
            new_x = prev_x * long_stochastic_gate
            return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def regularizer(self, x):  
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    
    def reg_loss(self):  
        reg = torch.norm(self.mu, 1)
        return reg
    
    def get_mask(self):
        return self.hard_sigmoid(self.mu.detach())

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif val_loss < self.best_loss + self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class CoxModel(BaseEstimator):
    def __init__(self, X_dim, random_state=None):
        self.fs_sigma = args.fs_sigma
        self.random_state = random_state
        self.X_dim = X_dim
        self.result_dict = {}
        self.init_paras(self.X_dim)

        self.result_df = []

    def _padToMatch2d(self, inputtens, targetshape, fill_value=-1e3):
        target = torch.full(targetshape, fill_value=fill_value)
        target[:inputtens.shape[0], :inputtens.shape[1]] = inputtens        
        return target

    def init_paras(self, d):
        self.feature_selector = FeatureSelector(self.X_dim, sigma=self.fs_sigma).to(device)  
        self.ps_model = LogReg(self.X_dim).to(device)  
        self.beta = torch.zeros(self.X_dim+1).to(device).float() 

    def my_get_params(self):
        return {
            'fs': self.feature_selector.mu.detach(),
            'ps_w': self.ps_model.linear.weight.detach(), 
            'ps_b': self.ps_model.linear.bias.detach(), 
            'beta': self.beta.detach()
        }

    def my_set_params(self, params_dict):
        
        with torch.no_grad():
            self.feature_selector.mu.copy_(params_dict['fs'])
            self.ps_model.linear.weight.copy_(params_dict['ps_w'])
            self.ps_model.linear.bias.copy_(params_dict['ps_b'])
            self.beta.copy_(params_dict['beta'])

    def _get_beta(self, fs=True):
        if fs:
            self.feature_selector.eval()
            return self.feature_selector(self.beta)
        else:
            return self.beta

    def get_results(self):
        ret = {}
        
        ret['num'] = torch.tensor(self.result_dict['num'])
        ret['best_ps_par_key'] = self.result_dict['best_ps_par_key']
        ret['weight'] = torch.tensor(self.result_dict['weight'])
        ret['smd_noweight'] = torch.tensor(self.result_dict['smd_noweight'])
        ret['smd'] = torch.tensor(self.result_dict['smd'])
        ret['smd_masked'] = torch.tensor(self.result_dict['smd_masked'])
        ret['ratio_balanced_noweight'] = self.result_dict['ratio_balanced_noweight']
        ret['ratio_balanced'] = self.result_dict['ratio_balanced']
        ret['ratio_balanced_masked'] = self.result_dict['ratio_balanced_masked']

        ret['beta'] = self.result_dict['beta']
        ret['HR'] = self.result_dict['HR']
        ret['fs'] = self.result_dict['fs']
        ret['fs_beta'] = tuple(torch.cat((torch.tensor([1.]), torch.tensor(ret['fs']))) * torch.tensor(self.result_dict['beta']))
        
        ret['se'] = self.result_dict['se']
        ret['CI_lo'] = self.result_dict['CI_lo']
        ret['CI_hi'] = self.result_dict['CI_hi']
        ret['val_loss'] = self.result_dict['val_loss']
        ret['best_outcome_par_key'] = self.result_dict['best_outcome_par_key']
        ret['C_index'] = self.result_dict['C_index']

        for k, v in ret.items():
            if type(v) is str:
                new_v = str(v)
            else:
                new_v = torch.tensor(v).numpy()
            ret[k] = new_v
        return ret
        
    @staticmethod
    def average_params(params_dict_list, N_list):
        N_list = torch.tensor(N_list).float()
        normalized_N_list = N_list / N_list.sum()
        ret = {}
        keys = params_dict_list[0].keys()
        for key in keys:
            if key == 'beta':
                values = [torch.cat((torch.tensor([1.]), torch.clamp(params_dict['fs'], 0.0, 1.0))) * params_dict['beta']  for params_dict in params_dict_list]
            elif key == 'fs':
                values = [torch.clamp(params_dict['fs'], 0.0, 1.0) for params_dict in params_dict_list]
            else:
                values = [params_dict[key] for params_dict in params_dict_list]
            stacked_values = torch.stack(values)
            if len(stacked_values.shape) == 2:
                mean_values = torch.einsum('ik,i->k', stacked_values, normalized_N_list)
            elif len(stacked_values.shape) == 3:
                mean_values = torch.einsum('ijk,i->jk', stacked_values, normalized_N_list)

            ret[key] = mean_values
        
        ret['avg_fs'] = ret['fs']
        ret['fs'] = torch.ones(ret['fs'].shape)

        return ret


    def get_loss(self, tensor, event_tens, num_tied, beta, mask=None):
        
        loss_event = torch.einsum('ik,k->i', event_tens, beta)  
        w_d = torch.tensor([tensor[i, :num_tied[i], -1].sum() for i in range(tensor.shape[0])])
        XB = torch.einsum('ijk,k->ij', tensor[:,:,:-1], beta) + torch.log(tensor[:,:,-1])  
        if mask is not None:
            XB = XB.masked_fill(~mask, -float('inf'))  
        loss_atrisk = -w_d*torch.logsumexp(XB, dim=1)
        loss = torch.sum(loss_event + loss_atrisk)
        return -loss

    def fit(self, data, duration_col=None, event_col=None, weights_col=None, scale=True, train_val_idx=None, test_idx=None, fit_options=None):
        df = data.copy()
        if duration_col is None and event_col is None:  
            df.columns = ["duration_col", "event_col", "treatment_col"] + ["Z" + str(i) for i in range(1, df.shape[1]-2)]
            duration_col = "duration_col"
            event_col = "event_col"
        if weights_col is None: 
            weights_col = 'newly_added_weights'
            df[weights_col] = np.ones(len(df))
    
        self.tname = duration_col
        self.dname = event_col
        self.Xnames = [col for col in df.columns if col not in [self.tname, self.dname, weights_col]]
        T = torch.tensor(df[self.Xnames[0]].to_numpy()).float()
        
        if scale:
            scaler = StandardScaler()  
            df[self.Xnames] = scaler.fit_transform(df[self.Xnames]) 
            df[weights_col] *= len(df[weights_col]) / np.sum(df[weights_col].to_numpy())

        self.Xnames.append(weights_col)
        self.result_dict['num'] = len(df)  

        iterative_num_epochs = fit_options['iterative_num_epochs']
        for epoch_iterative in range(iterative_num_epochs):

            X = torch.tensor(df[self.Xnames[1:-1]].to_numpy()).float()
            T_label = (T > 0.5).long()
            X, T = X.to(device), T_label.to(device) 
            X_train_val, X_test = X[train_val_idx, :], X[test_idx, :]
            T_label_train_val, T_label_test = T_label[train_val_idx], T_label[test_idx]

            smd_threshold = fit_options.get('smd_threshold')
            ps_hp_grid = {}
            for _ in ['ps_num_epochs', 'ps_lr', 'ps_es_patience', 'ps_es_threshold', 'ps_optim_method', 'ps_fedprox_reg', 'balance_reg']:
                ps_hp_grid[_] = fit_options[_] if type(fit_options[_]) is list else [fit_options[_]]
            ps_hp_list = [dict(zip(ps_hp_grid, v)) for v in product(*ps_hp_grid.values())]

            ps_results_dict = {}

            for _, par in tqdm(enumerate(ps_hp_list), total=len(ps_hp_list), desc=f"Iterative Epoch={epoch_iterative} PS hp"):
                ps_num_epochs = par['ps_num_epochs']
                ps_lr = par['ps_lr']
                ps_es_patience = par['ps_es_patience']
                ps_es_threshold = par['ps_es_threshold']
                ps_optim_method = par['ps_optim_method']
                ps_fedprox_reg = par['ps_fedprox_reg']
                balance_reg = par['balance_reg']

                par_key = f'PS-E{ps_num_epochs}-lr{ps_lr}-es{ps_es_patience}_{ps_es_threshold}-opt{ps_optim_method}-fedprox{ps_fedprox_reg}'
                ps_results_dict[par_key] = []
 
                train_idx_list, val_idx_list = [], []
                kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
                for train_idx, val_idx in kf.split(X_train_val):
                    train_idx_list.append(train_idx)
                    val_idx_list.append(val_idx)
                
                train_idx_list.append(np.concatenate((train_idx, val_idx), axis=0))
                val_idx_list.append(np.concatenate((train_idx, val_idx), axis=0))

                for fold_idx, (train_idx, val_idx) in enumerate(zip(train_idx_list, val_idx_list)):
                    T_label_train, T_label_val = T_label_train_val[train_idx], T_label_train_val[val_idx]


                for fold_idx, (train_idx, val_idx) in enumerate(zip(train_idx_list, val_idx_list)):
                    
                    if not (fold_idx == 0 or fold_idx == len(train_idx_list) - 1):  
                        continue

                    X_train, X_val = X_train_val[train_idx, :], X_train_val[val_idx, :]
                    T_label_train, T_label_val = T_label_train_val[train_idx], T_label_train_val[val_idx]

                    ps_early_stopping = EarlyStopping(patience=ps_es_patience, min_delta=ps_es_threshold) 
                    
                    ps_model = copy.deepcopy(self.ps_model).to(device)  
                    init_ps_params = copy.deepcopy(ps_model).parameters()

                    if ps_optim_method.lower() == 'adam': 
                        ps_optimizer = optim.Adam(ps_model.parameters(), lr=ps_lr)
                    else:
                        ps_optimizer = optim.SGD(ps_model.parameters(), lr=ps_lr)
                    criterion = nn.CrossEntropyLoss()

                    for epoch_ps in range(ps_num_epochs):

                        ps_model.train()
                        ps_optimizer.zero_grad()
                        X_train_masked = self.feature_selector(X_train)
                        logits = ps_model(X_train_masked)
                        loss = criterion(logits, T_label_train)
                        ce_loss_item = loss.item()
                        if balance_reg > 0.:
                            T_probs = F.softmax(ps_model(X_train_masked), dim=-1)[:, 1]
                            _, _, ps_weights_train_val = cal_weights(golds_treatment=T_label_train, probs=T_probs, stabilized=True)                    
                            ps_weights_train_val /= float(sum(ps_weights_train_val) / len(ps_weights_train_val))
                            balance_term = weighted_smd(X=X_train_masked, T=T_label_train, weights=ps_weights_train_val, return_numerator=True).norm(2)
                            loss += balance_reg * balance_term
                        else:
                            balance_term = torch.tensor(0.0)
                        
                        if ps_fedprox_reg > 0.:
                            proximal_term = 0.0
                            for param, global_param in zip(ps_model.parameters(), init_ps_params):
                                proximal_term += ps_fedprox_reg * torch.norm(param - global_param) ** 2
                            loss += proximal_term

                        loss.backward()
                        ps_optimizer.step()

                        with torch.no_grad():
                            ps_model.eval()
                            self.feature_selector.eval()

                            T_probs_train_val = F.softmax(ps_model(self.feature_selector(X_train_val)), dim=-1)[:, 1]
                            _, _, ps_weights_train_val = cal_weights(golds_treatment=T_label_train_val, probs=T_probs_train_val, stabilized=True)                       
                            ps_weights_train_val /= (sum(ps_weights_train_val) / len(ps_weights_train_val))

                            smd_train_val = weighted_smd(X=X_train_val, T=T_label_train_val, weights=ps_weights_train_val)
                            n_balanced_train_val =  int((smd_train_val <= smd_threshold).sum())
                            ratio_balanced_train_val =  float(n_balanced_train_val) / len(smd_train_val)
                            
                            fs_mask = self.feature_selector.get_mask()
                            smd_train_val_masked = smd_train_val * fs_mask
                            n_balanced_train_val_masked =  int((smd_train_val_masked <= smd_threshold).sum())
                            T_probs_val = F.softmax(ps_model(self.feature_selector(X_val)), dim=-1)[:, 1]
                            auc_score_val = roc_auc_score(y_true=T_label_val.cpu().detach().numpy(), y_score=T_probs_val.cpu().numpy())


                        val_loss = - n_balanced_train_val_masked - auc_score_val   
                        ps_early_stopping(val_loss, (epoch_ps, ps_model, ps_weights_train_val))
                        if ps_early_stopping.early_stop or epoch_ps == ps_num_epochs - 1:  
                            
                            best_epoch, ps_model, ps_weights_train_val = ps_early_stopping.best_model  
                            val_loss = ps_early_stopping.best_loss

                            T_probs_test = F.softmax(ps_model(self.feature_selector(X_test)), dim=-1)[:, 1]
                            _, _, ps_weights_test = cal_weights(golds_treatment=T_label_test, probs=T_probs_test, stabilized=True)              
                            ps_weights_test /= (sum(ps_weights_test) / len(ps_weights_test))

                            T_probs_all = F.softmax(ps_model(self.feature_selector(X)), dim=-1)[:, 1]
                            _, _, ps_weights_all = cal_weights(golds_treatment=T_label, probs=T_probs_all, stabilized=True)                 
                            ps_weights_all /= (sum(ps_weights_all) / len(ps_weights_all))
                            
                            def smd_related_values(X, T_label, weights, mask=None):
                                smd = weighted_smd(X=X, T=T_label, weights=weights)
                                if mask is not None:
                                    smd = mask * smd
                                n_balanced = int((smd <= smd_threshold).sum())
                                ratio_balanced =  float(n_balanced) / len(smd)
                                return smd, n_balanced, ratio_balanced
                            
                            smd_noweight_train_val, n_balanced_noweight_train_val, ratio_balanced_noweight_train_val = smd_related_values(X=X_train_val, T_label=T_label_train_val, weights=None)
                            smd_noweight_test, n_balanced_noweight_test, ratio_balanced_noweight_test = smd_related_values(X=X_test, T_label=T_label_test, weights=None)
                            smd_noweight_all, n_balanced_noweight_all, ratio_balanced_noweight_all = smd_related_values(X=X, T_label=T_label, weights=None)
                            smd_train_val, n_balanced_train_val, ratio_balanced_train_val = smd_related_values(X=X_train_val, T_label=T_label_train_val, weights=ps_weights_train_val)
                            smd_test, n_balanced_test, ratio_balanced_test = smd_related_values(X=X_test, T_label=T_label_test, weights=ps_weights_test)
                            smd_all, n_balanced_all, ratio_balanced_all = smd_related_values(X=X, T_label=T_label, weights=ps_weights_all)
                            smd_masked_train_val, n_balanced_masked_train_val, ratio_balanced_masked_train_val = smd_related_values(X=X_train_val, T_label=T_label_train_val, weights=ps_weights_train_val, mask=self.feature_selector.get_mask())
                            smd_masked_test, n_balanced_masked_test, ratio_balanced_masked_test = smd_related_values(X=X_test, T_label=T_label_test, weights=ps_weights_test, mask=self.feature_selector.get_mask())
                            smd_masked_all, n_balanced_masked_all, ratio_balanced_masked_all = smd_related_values(X=X, T_label=T_label, weights=ps_weights_all, mask=self.feature_selector.get_mask())

                            with torch.no_grad():   
                                T_probs_val = F.softmax(ps_model(self.feature_selector(X_val)), dim=-1)[:, 1]
                                auc_score_val = roc_auc_score(y_true=T_label_val.cpu().detach().numpy(), y_score=T_probs_val.cpu().numpy())                          
                                T_probs_test = F.softmax(ps_model(self.feature_selector(X_test)), dim=-1)[:, 1]
                                auc_score_test = roc_auc_score(y_true=T_label_test.cpu().detach().numpy(), y_score=T_probs_test.cpu().numpy())  
                                T_probs_all = F.softmax(ps_model(self.feature_selector(X)), dim=-1)[:, 1]
                                auc_score_all = roc_auc_score(y_true=T_label.cpu().detach().numpy(), y_score=T_probs_all.cpu().numpy())     

                            ps_results_k = { 
                                'epoch_ps': epoch_ps,
                                'auc_score_val': auc_score_val,
                                'auc_score_test': auc_score_test,
                                'auc_score_all': auc_score_all,
                                'ps_weights_train_val': list(ps_weights_train_val.detach().numpy()),
                                'ps_weights_test': list(ps_weights_test.detach().numpy()),
                                'ps_weights_all': list(ps_weights_all.detach().numpy()),
                                'ps_weights_mean_min_max_train_val': [ps_weights_train_val.mean().item(), ps_weights_train_val.min().item(), ps_weights_train_val.max().item()],
                                'ps_weights_mean_min_max_test': [ps_weights_test.mean().item(), ps_weights_test.min().item(), ps_weights_test.max().item()],
                                'ps_weights_mean_min_max_all': [ps_weights_all.mean().item(), ps_weights_all.min().item(), ps_weights_all.max().item()],
                                'smd_noweight_train_val': [float("%.3f" % _)  for _ in smd_noweight_train_val], 'ratio_balanced_noweight_train_val': ratio_balanced_noweight_train_val, 'n_balanced_noweight_train_val': n_balanced_noweight_train_val, 
                                'smd_noweight_test': [float("%.3f" % _)  for _ in smd_noweight_test], 'ratio_balanced_test': ratio_balanced_noweight_test, 'n_balanced_test': n_balanced_noweight_test, 
                                'smd_noweight_all': [float("%.3f" % _)  for _ in smd_noweight_all], 'ratio_balanced_noweight_all': ratio_balanced_noweight_all, 'n_balanced_noweight_all': n_balanced_noweight_all, 
                                'smd_train_val': [float("%.3f" % _)  for _ in smd_train_val], 'ratio_balanced_train_val': ratio_balanced_train_val, 'n_balanced_train_val': n_balanced_train_val, 
                                'smd_test': [float("%.3f" % _)  for _ in smd_test], 'ratio_balanced_test': ratio_balanced_test, 'n_balanced_test': n_balanced_test, 
                                'smd_all': [float("%.3f" % _)  for _ in smd_all], 'ratio_balanced_all': ratio_balanced_all, 'n_balanced_all': n_balanced_all, 
                                'smd_masked_train_val': [float("%.3f" % _)  for _ in smd_masked_train_val], 'ratio_balanced_masked_train_val': ratio_balanced_masked_train_val, 'n_balanced_masked_train_val': n_balanced_masked_train_val, 
                                'smd_masked_test': [float("%.3f" % _)  for _ in smd_masked_test], 'ratio_balanced_test': ratio_balanced_masked_test, 'n_balanced_test': n_balanced_masked_test, 
                                'smd_masked_all': [float("%.3f" % _)  for _ in smd_masked_all], 'ratio_balanced_masked_all': ratio_balanced_masked_all, 'n_balanced_masked_all': n_balanced_masked_all, 
                                'best_model': ps_model,
                            }
                            ps_results_dict[par_key].append(ps_results_k)
                            break 

            tmp_ps_results_df = {}
            for key, _ in ps_results_dict.items():
                tmp_ps_results_df[key] = {}
                ps_results_hp_mean, ps_results_std = calculate_mean_and_std(ps_results_dict[key][:-1])
                ps_results_retrain = ps_results_dict[key][-1]
                tmp_ps_results_df[key].update({key + '_mean': value for key, value in ps_results_hp_mean.items()})
                tmp_ps_results_df[key].update({key + '_std': value for key, value in ps_results_std.items()})
                tmp_ps_results_df[key].update({key + '_zretrain': value for key, value in ps_results_retrain.items()})
            
            ps_results_df = pd.DataFrame.from_dict(tmp_ps_results_df, orient='index').reset_index().drop(columns=['best_model_zretrain'])
            ps_results_df = ps_results_df[sorted(ps_results_df.columns)]  
            ps_results_df = ps_results_df[['index'] + [_ for _ in ps_results_df.columns if _ != 'index']]
            ps_results_df = ps_results_df.rename(columns={'index': 'par_key'})
            ps_results_df = ps_results_df.sort_values(by=['n_balanced_masked_train_val_mean', 'auc_score_val_mean'], ascending=[False, False])  
            ps_results_path = osp.join(args.logdir, 'ps_results.csv')
            ps_results_df.to_csv(ps_results_path, mode='a', index=False)
            
            self.result_dict['best_ps_par_key'] = ps_results_df.iloc[0]['par_key']
            self.result_dict['weight'] = torch.tensor(ps_results_df.iloc[0]['ps_weights_all_zretrain'], dtype=float)
            self.result_dict['smd_noweight'] = ps_results_df.iloc[0]['smd_noweight_all_zretrain']
            self.result_dict['smd'] = ps_results_df.iloc[0]['smd_all_zretrain']
            self.result_dict['smd_masked'] = ps_results_df.iloc[0]['smd_masked_all_zretrain']
            self.result_dict['ratio_balanced_noweight'] = ps_results_df.iloc[0]['ratio_balanced_noweight_all_zretrain']
            self.result_dict['ratio_balanced'] = ps_results_df.iloc[0]['ratio_balanced_all_zretrain']
            self.result_dict['ratio_balanced_masked'] = ps_results_df.iloc[0]['ratio_balanced_masked_all_zretrain']

            df[self.Xnames[-1]] = self.result_dict['weight']  
            self.ps_model = copy.deepcopy(tmp_ps_results_df[self.result_dict['best_ps_par_key']]['best_model_zretrain'])  

            inputdf = df[[self.tname,self.dname,*self.Xnames]].sort_values([self.dname,self.tname], ascending=[False,True])
            tiecountdf = inputdf.loc[inputdf[self.dname]==1,:].groupby([self.tname]).size().reset_index(name='tiecount')
            num_tied = torch.from_numpy(tiecountdf.tiecount.values).int() 
            tensin = torch.from_numpy(inputdf[[self.tname,self.dname,*self.Xnames]].values)  
            tensin_events = torch.unique(tensin[tensin[:,1]==1, 0]) 
            
            MINIMUM_VALUE = -1e6
            tensor = torch.stack([self._padToMatch2d(tensin[tensin[:,0] >= eventtime, :], tensin.shape, fill_value=MINIMUM_VALUE) for eventtime in tensin_events]) 
            mask = ~(tensor < MINIMUM_VALUE + 0.1)
            mask_reduced = torch.empty(mask.shape[:2], dtype=torch.bool)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    mask_reduced[i, j] = torch.all(mask[i, j])

            event_tens = torch.stack([torch.einsum('ij,i->j', tensor[i, :num_tied[i], 2:-1], tensor[i, :num_tied[i], -1]) for i in range(tensor.shape[0])])
            tensor = tensor[:,:,2:]  
            hp_grid = {}
            for _ in ['num_epochs', 'lr', 'es_patience', 'es_threshold', 'optim_method', 'fedprox_reg', 'fs_reg', 'beta_l1_reg', 'beta_l2_reg']:
                hp_grid[_] = fit_options[_] if type(fit_options[_]) is list else [fit_options[_]]
            hp_list = [dict(zip(hp_grid, v)) for v in product(*hp_grid.values())]
            results_dict = {}
            for _, par in tqdm(enumerate(hp_list), total=len(hp_list) , desc=f"Iterative Epoch={epoch_iterative} OC hp"):
                num_epochs = par['num_epochs']
                lr = par['lr']
                es_patience = par['es_patience']
                es_threshold = par['es_threshold']
                optim_method = par['optim_method']
                fedprox_reg = par['fedprox_reg']
                fs_reg = par['fs_reg']
                beta_l1_reg = par['beta_l1_reg']
                beta_l2_reg = par['beta_l2_reg']
                par_key = f'Outcome-E{num_epochs}-lr{lr}-es{ps_es_patience}_{es_threshold}-opt{optim_method}-fedprox{fedprox_reg}-l{beta_l1_reg}-{beta_l2_reg}'
                results_dict[par_key] = {}

                early_stopping = EarlyStopping(patience=es_patience, min_delta=es_threshold)
                beta = nn.Parameter(copy.deepcopy(self.beta))
                feature_selector = copy.deepcopy(self.feature_selector)
                
                init_beta = copy.deepcopy(beta)
                init_feature_selector = copy.deepcopy(feature_selector)

                if optim_method.lower() == 'adam':
                    optimizer = optim.Adam(list([beta]) + list(feature_selector.parameters()), lr=lr)
                else:
                    optimizer = optim.SGD(list([beta]) + list(feature_selector.parameters()), lr=lr)
                
                for epoch_outcome in range(num_epochs): 
                    optimizer.zero_grad()
                    loss = self.get_loss(tensor, event_tens, num_tied, feature_selector(beta), mask_reduced) + fs_reg * feature_selector.reg_loss()

                    if fedprox_reg > 0.:
                        proximal_term = fedprox_reg * torch.norm(feature_selector(beta) - init_feature_selector(init_beta)) ** 2
                        loss += proximal_term

                    n_samples = sum(df[self.dname] == 1) 
                    if beta_l1_reg > 0.:
                        loss += beta_l1_reg * torch.norm(beta, 1) * n_samples
                    if beta_l2_reg > 0.:
                        loss += beta_l2_reg * torch.norm(beta, 2) * n_samples

                    loss.backward()
                    optimizer.step()

                    if scale:
                        scaled_beta = torch.tensor(beta.detach().numpy() / (np.sqrt(scaler.var_) + 1e-6)).float()
                    else:
                        scaled_beta = torch.tensor(beta.detach().numpy()).float()

                    early_stopping(val_loss, (beta, feature_selector))
                    if early_stopping.early_stop or epoch_outcome == num_epochs - 1:
                        beta, feature_selector = early_stopping.best_model
                        val_loss = early_stopping.best_loss
                        loss = self.get_loss(tensor, event_tens, num_tied, feature_selector(beta), mask_reduced) + fs_reg * feature_selector.reg_loss()
                        if fedprox_reg > 0.: 
                            proximal_term = fedprox_reg * torch.norm(feature_selector(beta) - init_feature_selector(init_beta)) ** 2
                            loss += proximal_term

                        hessian_matrix, grads_vector = compute_hessian(loss, list([beta]) + list(feature_selector.parameters()))
                        hessian_matrix = hessian_matrix[:len(beta), :len(beta)]
                        n_samples = sum(df[self.dname] == 1)  
                        if scale:
                            hessian_matrix = hessian_matrix / np.outer(np.sqrt(scaler.var_) + 1e-6,  np.sqrt(scaler.var_) + 1e-6)
                        cov_matrix = torch.inverse(hessian_matrix)
    
                        se = torch.sqrt(torch.diag(cov_matrix) / n_samples)
                        if torch.isnan(se).any(): 
                            eigvals, eigvecs = torch.linalg.eigh(hessian_matrix) 
                            eigvals = torch.clamp(eigvals, min=1e-5) 
                            hessian_matrix = eigvecs @ torch.diag(eigvals) @ eigvecs.T 
                            cov_matrix = torch.inverse(hessian_matrix)
                            se = torch.sqrt(torch.diag(cov_matrix) / n_samples)
                        se_beta = se[:len(beta)]

                        results_dict[par_key]['val_loss'] = early_stopping.best_loss
                        results_dict[par_key]['beta'] = tuple(scaled_beta.detach().numpy())
                        results_dict[par_key]['HR'] = tuple(np.exp(results_dict[par_key]['beta']))
                        results_dict[par_key]['se'] = tuple(se_beta.detach().numpy())
                        results_dict[par_key]['CI_lo'] = tuple(np.exp(results_dict[par_key]['beta']) * np.exp(se_beta.detach().numpy() * -1.96))
                        results_dict[par_key]['CI_hi'] = tuple(np.exp(results_dict[par_key]['beta']) * np.exp(se_beta.detach().numpy() * 1.96))
                        results_dict[par_key]['fs'] = tuple(feature_selector.get_mask().detach().numpy())

                        results_dict[par_key]['outcome_model'] = (beta, feature_selector)
                        break  
                                    
            results_df = pd.DataFrame.from_dict(results_dict, orient='index').reset_index().drop(columns=['outcome_model'])
            results_df = results_df.rename(columns={'index': 'par_key'})
            results_df = results_df.sort_values(by=['val_loss', 'se'], ascending=[True, True])  
            results_path = osp.join(args.logdir, 'outcome_results.csv')
            results_df.to_csv(results_path, mode='a', index=False)

            self.result_dict['best_outcome_par_key'] = results_df.iloc[0]['par_key']
            self.result_dict['beta'] = results_df.iloc[0]['beta']
            self.result_dict['HR'] = results_df.iloc[0]['HR']
            self.result_dict['se'] = results_df.iloc[0]['se']
            self.result_dict['CI_lo'] = results_df.iloc[0]['CI_lo']
            self.result_dict['CI_hi'] = results_df.iloc[0]['CI_hi']
            self.result_dict['fs'] = results_df.iloc[0]['fs']
            self.result_dict['val_loss'] = results_df.iloc[0]['val_loss']
            self.result_dict['C_index'] = self.calculate_c_index(data=data, duration_col=duration_col, event_col=event_col)

            beta, feature_selector = results_dict[self.result_dict['best_outcome_par_key']]['outcome_model']
            self.feature_selector = copy.deepcopy(feature_selector)
            self.beta = copy.deepcopy(beta)
            
            new_row = {'epoch_iterative': epoch_iterative}
            new_row.update(self.get_results())
            for k, v in new_row.items():
                try:
                    new_v = '|'.join([f'{_:.4f}' for _ in list(v)])
                except:
                    new_v = v
                new_row[k] = new_v
            self.result_df.append(new_row)

        if fit_options['basehaz']:
            t, _ = torch.sort(torch.from_numpy(inputdf[self.tname].values))
            t_uniq = torch.unique(t)
            h0 = []
            for time in t_uniq:  
                X_j = torch.from_numpy(inputdf.loc[inputdf[self.tname] >= time.numpy(), self.Xnames[:-1]].values).float()  
                w_j = torch.from_numpy(inputdf.loc[inputdf[self.tname] >= time.numpy(), self.Xnames[-1]].values).float()
                sum_wt = torch.from_numpy(inputdf.loc[(inputdf[self.tname] == time.numpy()) & (inputdf[self.dname]==1), self.Xnames[-1]].values).sum().float() 
                XB_j = torch.einsum('ij,j->i', X_j, beta.clone().detach())
                value = sum_wt / torch.sum(w_j*torch.exp(XB_j))
                h0.append({'time':time.numpy(), 'h0':value.detach().numpy()})

            h0df = pd.DataFrame(h0)
            h0df['H0'] = h0df.h0.cumsum()

            self.basehaz = h0df  
        
        if scale: 
            self.beta = beta.clone().detach() / torch.tensor(np.sqrt(scaler.var_) + 1e-6).float()
        else:
            self.beta = beta.clone().detach()

    
    def calculate_c_index(self, data, duration_col, event_col, fs=True):
        def _calculate_c_index(event_times, predicted_scores, event_observed):
            n = 0
            n_concordant = 0
            n_tied = 0

            for i in range(len(event_times)):
                for j in range(i + 1, len(event_times)):
                    if event_times[i] != event_times[j]:
                        if event_times[i] < event_times[j] and event_observed[i] == 1:
                            n += 1
                            if predicted_scores[i] > predicted_scores[j]:
                                n_concordant += 1
                            elif predicted_scores[i] == predicted_scores[j]:
                                n_tied += 1
                        elif event_times[j] < event_times[i] and event_observed[j] == 1:
                            n += 1
                            if predicted_scores[j] > predicted_scores[i]:
                                n_concordant += 1
                            elif predicted_scores[j] == predicted_scores[i]:
                                n_tied += 1

            c_index = (n_concordant + 0.5 * n_tied) / n if n > 0 else 0
            return c_index

        def calculate_risk_score(row, coefficients):
            risk_score = np.sum(row * coefficients)
            return np.exp(risk_score)
        
        coefficients = self._get_beta(fs=fs).detach().numpy()
        risk_scores = data.drop(columns=[duration_col, event_col]).apply(calculate_risk_score, axis=1, coefficients=coefficients)
        c_index_manual = _calculate_c_index(data[duration_col].to_numpy(), risk_scores.to_numpy(), data[event_col].to_numpy())
        return c_index_manual
                        
    def predict_proba(self, testdf, Xnames=None, tname=None):
        betas = self.beta.clone().detach().numpy()
        H0 = np.asarray([self.basehaz.loc[self.basehaz.time<=t, 'H0'].iloc[-1] for t in testdf[tname].values])
        S = np.exp(np.multiply(-np.exp(np.dot(testdf[Xnames].values, betas)), H0))
        return S
    


df = None 
site_list = None
args = None 
site_col = 'site_id'  
duration_col = 'duration' 
event_col = 'event'  
fit_options = {}
final_results_df = []

siteid_N_model_sitedata_list = []
for k, site_id in enumerate(site_list):
    site_data = df[df[site_col] == site_id].drop(columns=[site_col]).reset_index().drop(columns=['index'])
    siteid_N_model_sitedata_list.append((site_id, len(site_data), CoxModel(X_dim=len(site_data.columns)-3), site_data))
model_list = [_[2] for _ in siteid_N_model_sitedata_list]
N_list = [_[1] for _ in siteid_N_model_sitedata_list]

for epoch_fl in range(fit_options['fed_epoch']):
    for k, (site_id,  _, coxmodel_k, site_data) in enumerate(siteid_N_model_sitedata_list):
        train_val_idx, test_idx = split_df(site_data, args.train_test_ratio) 
        coxmodel_k.fit(site_data, duration_col=duration_col, event_col=event_col, train_val_idx=train_val_idx, test_idx=test_idx, fit_options=fit_options)
        site_result_df = pd.DataFrame(coxmodel_k.result_df)
        final_results_df.append(site_result_df)
    params_dict_list = [model.my_get_params() for model in model_list]
    new_params_dict = CoxModel.average_params(params_dict_list, N_list=N_list)
    for k in range(len(model_list)):
        model_list[k].my_set_params(new_params_dict)
