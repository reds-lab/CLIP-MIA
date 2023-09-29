import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from data import get_data, get_data_val
import argparse
import random
import requests
import sys
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop, \
RandomRotation, RandomAffine, AugMix, GaussianBlur, RandomHorizontalFlip, RandomVerticalFlip, RandomAutocontrast, \
RandomAdjustSharpness, RandomPosterize, RandomResizedCrop, ColorJitter
from torch.utils.data import Dataset, TensorDataset, DataLoader
import open_clip
from open_clip import tokenizer, tokenize
from typing import Optional, Sequence, Tuple
from PIL import Image
import util

from open_clip import create_model_and_transforms, trace_model
from params import *
from text_preprocessing import text_preprocessing
from nontrain_selection import *
from pseudotrain_selection import *
from train_attackmodel import train_attackmodel
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import copy
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
from open_clip import tokenizer, tokenize
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import RandomResizedCrop, RandomRotation, RandomAffine, ColorJitter 
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.autograd import Function

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc

def main(args, device):
    nontrain_length = args.nt_length # sample size for each dataset
    length_train = args.t_length # 
    evaluate_length = args.eval_length # 
    lambda_selection = [-2.5, -1.5, -0.5, 0., 0.5, 1.5, 2.5] # 7 choices
    
    target_model, target_preprocess_train, target_preprocess_val = create_model_and_transforms(
        args.model, # 'ViT-B-32' / 'RN50' / 'ViT-L-14' / 'ViT-B-16'
        pretrained = 'laion400m_e32', # args.pretrained, 'laion400m_e32' 
    )
    
    ## initialize datasets
    original_batch_size = args.batch_size
    args.batch_size = args.nt_batch_size
    ## initialize datasets
    start_epoch = 0
    data = get_data(args, (target_preprocess_train, target_preprocess_val), epoch=start_epoch) # data : data[train]: dataloader, data[val]: dataloader 
    assert len(data), 'At least one train or eval dataset must be specified.'
    valloader = data['val'].dataloader
    dataloader = data['train'].dataloader

    print(util.red('current step: target model selection : [{}]').format(args.model))
    print(util.red('current step: hyper-lambda : [{}]').format(args.hyper_lambda))    
    print(util.red('current step: train data is : [{}]').format(data['train']))
    print(util.red('current step: val data is : [{}]').format(data['val']))

    target_model = target_model.eval()
    target_model = target_model.to(device)
    
    #--------------------------------------- Approach : Hierarchical Sampling ---------------------------------------#
    print(util.yellow('Approach - Step 1 : collecting the non-member set'))
    ## first, need to collect the non-train set 
    selected_nt_txt, selected_nt_url, selected_nt_cs_lst_tar, selected_nt_feat_lst_tar = select_nontrain(args, target_model, target_preprocess_train, target_preprocess_val, device, nontrain_length)
    print(util.yellow('Done, sampling nontrain set : [{}]').format(len(selected_nt_url))) 

    #--------------------------------------- Approach : Hierarchical Sampling ---------------------------------------#  
    print(util.blue('Approach - Step 2 : collecting the pseudo-member set'))    
    train_threshold = np.mean(selected_nt_cs_lst_tar) + args.hyper_lambda * np.std(selected_nt_cs_lst_tar)
    print(util.blue('Train_threshold: [{}]').format(train_threshold))

    selected_t_txt, selected_t_url, selected_t_cs_lst_tar, selected_t_feat_lst_tar, true_train, pseudo_train = select_pseudotrain(args, target_model, selected_nt_txt, selected_nt_url, dataloader, train_threshold, target_preprocess_train, target_preprocess_val, device, length_train)
        
    #--------------------------------------- Approach : Hierarchical Sampling ---------------------------------------#  
    print(util.magenta('Approach - Step 3 : trainining an attacker model'))    
    attack_model, mis_rate = train_attackmodel(args, selected_t_feat_lst_tar, selected_nt_feat_lst_tar, true_train, pseudo_train, train_threshold, device)       
    attack_model = attack_model.to(device)
    
    #--------------------------------------- Approach : Hierarchical Sampling ---------------------------------------#      
    print(util.green('Approach - Step 4 : evaluation'))    

    evaluate_selected_t_img_lst = []
    evaluate_selected_t_url_lst = []    
    evaluate_selected_t_txt_lst = []
    evaluate_selected_t_cs_lst_tar = []
    evaluate_selected_t_feat_lst_tar = []

    evaluate_selected_nt_img_lst = []
    evaluate_selected_nt_url_lst = []    
    evaluate_selected_nt_txt_lst = []
    evaluate_selected_nt_cs_lst_tar = []
    evaluate_selected_nt_feat_lst_tar = []
    
    args.train_data = args.train_data_eval # "/home/myeongseob/clip-privacy/LAION/dataset/laion/laion400m-data/{13000..26000}.tar"
    args.train_num_samples = args.train_num_samples_eval # 130000000
    args.val_data = args.val_data_eval_1 # "/home/myeongseob/clip-privacy/Open_clip_training/src/data/cc3m/{00200..00299}.tar"
    args.val_num_samples = args.val_num_samples_eval_1 # 1000000

    start_epoch = 0
    evaluate_data = get_data(args, (target_preprocess_train, target_preprocess_val), epoch=start_epoch) 
    assert len(evaluate_data), 'At least one train or eval dataset must be specified.'

    evaluate_dataloader = evaluate_data['train'].dataloader
    evaluate_valloader = evaluate_data['val'].dataloader

    #################################################
    ## cal feature info 

    CC12M_LAION_commonset = np.load('./CC12M_LAION_commonset.npy')
    CC12M_LAION_unqiue_commonset = np.load('./CC12M_LAION_unqiue_commonset.npy')
    CC12M_LAION_url_commonset = np.load('./CC12M_LAION_url_commonset.npy')

    CC3M_LAION_commonset = np.load('./CC3M_LAION_commonset.npy')
    CC3M_LAION_unqiue_commonset = np.load('./CC3M_LAION_unqiue_commonset.npy')
    CC3M_LAION_url_commonset = np.load('./CC3M_LAION_url_commonset.npy')

    MSCOCO_LAION_commonset = np.load('./MSCOCO_LAION_commonset.npy')
    MSCOCO_LAION_unqiue_commonset = np.load('./MSCOCO_LAION_unqiue_commonset.npy')

    SBU_LAION_commonset = np.load('./SBU_LAION_commonset.npy')
    SBU_LAION_unqiue_commonset = np.load('./SBU_LAION_unqiue_commonset.npy')

    import time
    start_time = time.time()
        
    ## train data : LAION
    evaluate_cnt_train = 0
    for i, batch in enumerate( evaluate_dataloader ):

        ## Train       
        evlauate_train_text_lst = [text_preprocessing(q) for q in batch[1]]   
        evlauate_train_url = [d['url'] for d in batch[2]]             

        common = np.intersect1d(np.array(evlauate_train_text_lst), np.array(selected_t_txt)) ## duplication check with [attack model]train data
        x_ind = np.where(np.isin(np.array(evlauate_train_text_lst), common))[0]
        common2 = np.intersect1d(np.array(evlauate_train_text_lst), np.array(selected_nt_txt)) ## duplication check with [attack model]nontrain data
        x_ind_2nd = np.where(np.isin(np.array(evlauate_train_text_lst), common2))[0]
        common3 = np.intersect1d(np.array(evlauate_train_url), np.array(selected_t_url))
        x_ind_3rd = np.where(np.isin(np.array(evlauate_train_url), common3))[0]
        common4 = np.intersect1d(np.array(evlauate_train_url), np.array(selected_nt_url))
        x_ind_4th = np.where(np.isin(np.array(evlauate_train_url), common4))[0]
        combined_x_ind = np.union1d(np.union1d(np.union1d(x_ind, x_ind_2nd), x_ind_3rd), x_ind_4th)

        if len(combined_x_ind) > 0:
            selected_ind = np.setdiff1d(np.arange(len(batch[0])), combined_x_ind)
        else:
            selected_ind = np.arange(len(batch[0]))

        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        evaluate_cnt_train += len(images)

        evaluate_selected_t_img_lst.extend( images ) 
        evaluate_selected_t_txt_lst.extend( texts )

        images = images.to(device)
        texts = texts.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)    
        cs_2 = torch.diagonal(image_features2@text_features2.T)

        evaluate_selected_t_cs_lst_tar.extend( cs_2.detach().cpu().numpy() )  
        evaluate_selected_t_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1).detach().cpu() )

        if evaluate_cnt_train >= evaluate_length:
            break

    print(util.green('Evaluating the trainloader is finished [{}]').format(len(evaluate_selected_t_cs_lst_tar)))    
    print(util.green('--- %s seconds --- [{}]').format((time.time() - start_time)))    

    ### Nontrain data from valloader
    start_time = time.time()

    ## first val loader : CC3M
    ## nontrain data 
    evaluate_cnt_val = 0
    for i, batch in enumerate( evaluate_valloader ): 

        evlauate_non_train_text_lst = [text_preprocessing(q) for q in batch[1]]   
        evlauate_non_train_url = [d['url'] for d in batch[2]]             

        common, _, _ = np.intersect1d(np.array(evlauate_non_train_text_lst), CC3M_LAION_commonset, return_indices=True) ## duplication check with train data
        x_ind = np.where(np.isin(np.array(evlauate_non_train_text_lst), common))[0]    
        common2, _, _ = np.intersect1d(np.array(evlauate_non_train_text_lst), np.array(selected_t_txt), return_indices=True) ## duplication check with [attack model] train data
        x_ind_2nd = np.where(np.isin(np.array(evlauate_non_train_text_lst), common2))[0]    
        common3, _, _ = np.intersect1d(np.array(evlauate_non_train_text_lst), np.array(selected_nt_txt), return_indices=True) ## duplication check with [attack model] nontrain data
        x_ind_3rd = np.where(np.isin(np.array(evlauate_non_train_text_lst), common3))[0] 
        common4, _, _ = np.intersect1d(np.array(evlauate_non_train_url), CC3M_LAION_url_commonset, return_indices=True)
        x_ind_4th = np.where(np.isin(np.array(evlauate_non_train_url), common4))[0]    
        common5, _, _ = np.intersect1d(np.array(evlauate_non_train_url), np.array(selected_t_url), return_indices=True) 
        x_ind_5th = np.where(np.isin(np.array(evlauate_non_train_url), common5))[0]    
        common6, _, _ = np.intersect1d(np.array(evlauate_non_train_url), np.array(selected_nt_url), return_indices=True)
        x_ind_6th = np.where(np.isin(np.array(evlauate_non_train_url), common6))[0]        
        combined_x_ind = np.union1d(np.union1d(np.union1d(np.union1d(np.union1d(x_ind, x_ind_2nd), x_ind_3rd), x_ind_4th), x_ind_5th), x_ind_6th)

        if len(combined_x_ind) > 0:
            selected_ind = np.setdiff1d(np.arange(len(batch[0])), combined_x_ind)
        else:
            selected_ind = np.arange(len(batch[0]))

        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        evaluate_cnt_val += len(images)

        evaluate_selected_nt_img_lst.extend( images ) 
        evaluate_selected_nt_txt_lst.extend( texts )

        images = images.to(device)
        texts = texts.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)  
        cs_2 = torch.diagonal(image_features2@text_features2.T)     

        evaluate_selected_nt_cs_lst_tar.extend( cs_2.detach().cpu().numpy() ) 
        evaluate_selected_nt_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1).detach().cpu() )
        
        if evaluate_cnt_val >= int(evaluate_length/2):
            break

    print(util.green('Evaluating the cc3m valloader is finished [{}]').format(len(evaluate_selected_nt_cs_lst_tar)))    
    print(util.green('--- %s seconds --- [{}]').format((time.time() - start_time)))    
    
    ## second val loader : CC12M
    start_time = time.time()

    args.val_data = args.val_data_eval_2 # "/home/myeongseob/clip-privacy/Open_clip_training/src/data/cc12m/{00800..01199}.tar"
    args.val_num_samples = args.val_num_samples_eval_2 # 4000000
    cc12m_valoader = get_data_val(args, (target_preprocess_train, target_preprocess_val))
    cc12m_valoader = cc12m_valoader.dataloader

    ## nontrain data 
    evaluate_cnt_val = 0
    for i, batch in enumerate( cc12m_valoader ): 

        evlauate_non_train_text_lst = [text_preprocessing(q) for q in batch[1]]   
        evlauate_non_train_url = [d['url'] for d in batch[2]]             
        
        common, _, _ = np.intersect1d(np.array(evlauate_non_train_text_lst), CC12M_LAION_commonset, return_indices=True)
        x_ind = np.where(np.isin(np.array(evlauate_non_train_text_lst), common))[0]    
        common2, _, _ = np.intersect1d(np.array(evlauate_non_train_text_lst), np.array(selected_t_txt), return_indices=True) 
        x_ind_2nd = np.where(np.isin(np.array(evlauate_non_train_text_lst), common2))[0]    
        common3, _, _ = np.intersect1d(np.array(evlauate_non_train_text_lst), np.array(selected_nt_txt), return_indices=True)
        x_ind_3rd = np.where(np.isin(np.array(evlauate_non_train_text_lst), common3))[0] 
        common4, _, _ = np.intersect1d(np.array(evlauate_non_train_url), CC12M_LAION_url_commonset, return_indices=True)
        x_ind_4th = np.where(np.isin(np.array(evlauate_non_train_url), common4))[0]    
        common5, _, _ = np.intersect1d(np.array(evlauate_non_train_url), np.array(selected_t_url), return_indices=True) 
        x_ind_5th = np.where(np.isin(np.array(evlauate_non_train_url), common5))[0]    
        common6, _, _ = np.intersect1d(np.array(evlauate_non_train_url), np.array(selected_nt_url), return_indices=True)
        x_ind_6th = np.where(np.isin(np.array(evlauate_non_train_url), common6))[0]        
        combined_x_ind = np.union1d(np.union1d(np.union1d(np.union1d(np.union1d(x_ind, x_ind_2nd), x_ind_3rd), x_ind_4th), x_ind_5th), x_ind_6th)

        if len(combined_x_ind) > 0:
            selected_ind = np.setdiff1d(np.arange(len(batch[0])), combined_x_ind)
        else:
            selected_ind = np.arange(len(batch[0]))

        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        evaluate_cnt_val += len(images)

        evaluate_selected_nt_img_lst.extend( images ) 
        evaluate_selected_nt_txt_lst.extend( texts )

        images = images.to(device)
        texts = texts.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)  
        cs_2 = torch.diagonal(image_features2@text_features2.T)

        evaluate_selected_nt_cs_lst_tar.extend( cs_2.detach().cpu().numpy() ) 
        evaluate_selected_nt_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1).detach().cpu() )
        
        if evaluate_cnt_val >= int(evaluate_length/2):
            break

    print(util.green('Evaluating the cc12m valloader is finished [{}]').format(len(evaluate_selected_nt_cs_lst_tar)))    
    print(util.green('--- %s seconds --- [{}]').format((time.time() - start_time)))    
    
    ## third val loader : MSCOCO
    start_time = time.time()

    args.val_data = args.val_data_eval_3 # "/home/myeongseob/clip-privacy/Open_clip_training/src/data/mscoco/mscoco/{00040..00059}.tar"
    args.val_num_samples = args.val_num_samples_eval_3 # 200000
    mscoco_valoader =  get_data_val(args, (target_preprocess_train, target_preprocess_val))
    mscoco_valoader = mscoco_valoader.dataloader

    ## nontrain data 
    evaluate_cnt_val = 0
    for i, batch in enumerate( mscoco_valoader ): 

        evlauate_non_train_text_lst = [text_preprocessing(q) for q in batch[1]]   
        evlauate_non_train_url = [d['url'] for d in batch[2]]             

        common, _, _ = np.intersect1d(np.array(evlauate_non_train_text_lst), MSCOCO_LAION_commonset, return_indices=True)
        x_ind = np.where(np.isin(np.array(evlauate_non_train_text_lst), common))[0]    
        common2, _, _ = np.intersect1d(np.array(evlauate_non_train_text_lst), np.array(selected_t_txt), return_indices=True) 
        x_ind_2nd = np.where(np.isin(np.array(evlauate_non_train_text_lst), common2))[0]    
        common3, _, _ = np.intersect1d(np.array(evlauate_non_train_text_lst), np.array(selected_nt_txt), return_indices=True)
        x_ind_3rd = np.where(np.isin(np.array(evlauate_non_train_text_lst), common3))[0]   
        common4, _, _ = np.intersect1d(np.array(evlauate_non_train_url), np.array(selected_t_url), return_indices=True) 
        x_ind_4th = np.where(np.isin(np.array(evlauate_non_train_url), common4))[0]    
        common5, _, _ = np.intersect1d(np.array(evlauate_non_train_url), np.array(selected_nt_url), return_indices=True)
        x_ind_5th = np.where(np.isin(np.array(evlauate_non_train_url), common5))[0]        
        combined_x_ind = np.union1d(np.union1d(np.union1d(np.union1d(x_ind, x_ind_2nd), x_ind_3rd), x_ind_4th), x_ind_5th)

        if len(combined_x_ind) > 0:
            selected_ind = np.setdiff1d(np.arange(len(batch[0])), combined_x_ind)
        else:
            selected_ind = np.arange(len(batch[0]))

        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        evaluate_cnt_val += len(images)

        evaluate_selected_nt_img_lst.extend( images ) 
        evaluate_selected_nt_txt_lst.extend( texts )

        images = images.to(device)
        texts = texts.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)  
        cs_2 = torch.diagonal(image_features2@text_features2.T)

        evaluate_selected_nt_cs_lst_tar.extend( cs_2.detach().cpu().numpy() ) 
        evaluate_selected_nt_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1).detach().cpu() )
        
        if evaluate_cnt_val >= int(evaluate_length/2):
            break

    print(util.green('Evaluating the mscoco valloader is finished [{}]'.format(len(evaluate_selected_nt_cs_lst_tar))))
    print(util.green('--- {:.2f} seconds ---'.format(time.time() - start_time)))

    print(util.red('The number of non-train data for evaluating the attack model [{}]'.format(len(evaluate_selected_nt_cs_lst_tar))))
    print(util.red('The number of non-train data for evaluating the attack model [{}]'.format(len(evaluate_selected_t_cs_lst_tar))))
    print(util.red('Since the numbers between two datasets are not balanced, we do random sampling here to train an attack model'))

    ################################################## random sampling

    length_tuned = min([len(evaluate_selected_t_cs_lst_tar), len(evaluate_selected_nt_cs_lst_tar)])

    evaluate_X1_choice = np.random.choice(np.arange(len(evaluate_selected_t_cs_lst_tar)), size=length_tuned, replace=False)
    evaluate_X2_choice = np.random.choice(np.arange(len(evaluate_selected_nt_cs_lst_tar)), size=length_tuned, replace=False)
    print(util.cyan('the length of evaluate_selected_t_cs_lst_tar [{}]'.format(len(evaluate_selected_t_cs_lst_tar))))
    print(util.cyan('the length of evaluate_selected_nt_feat_lst_tar [{}]'.format(len(evaluate_selected_nt_cs_lst_tar))))
          
    prediction = np.concatenate( [np.stack(evaluate_selected_t_cs_lst_tar)[evaluate_X1_choice].squeeze(), np.stack(evaluate_selected_nt_cs_lst_tar)[evaluate_X2_choice].squeeze() ])
    ground_truth = np.concatenate( [np.ones( length_tuned ).astype('int'), np.zeros( length_tuned ).astype('int') ] ) 

    fpr, tpr, thresholds = metrics.roc_curve(ground_truth, prediction)
    roc_auc = metrics.auc(fpr, tpr)

    fpr_original, tpr_original, auc_original, acc_original = sweep(prediction, ground_truth)
    low_original = tpr_original[np.where(fpr_original<.01)[0][-1]]
    print(util.cyan(f'[CSA] AUC for Evaluation Dataset is : {metrics.auc(fpr, tpr)}'))    
    print(util.cyan(f'[CSA] AUC for Evaluation Dataset is : {metrics.auc(fpr, tpr)}'))
    print(util.cyan(f'[CSA] AUC for Evaluation Dataset is : {auc_original:.4f}, Accuracy is {acc_original:.4f}, TPR@1%%FPR is {low_original:.4f}'))
    
    evaluate_X1 = torch.stack(evaluate_selected_t_feat_lst_tar)[evaluate_X1_choice]
    evaluate_X2 = torch.stack(evaluate_selected_nt_feat_lst_tar)[evaluate_X2_choice]

    evaluate_Y1 = torch.ones(len(evaluate_X1)).to(dtype = torch.long)
    evaluate_Y2 = torch.zeros(len(evaluate_X2)).to(dtype = torch.long)
    
    evaluate_data = torch.cat( [evaluate_X1 , evaluate_X2] ) 
    evaluate_labels = torch.cat( [evaluate_Y1 , evaluate_Y2] )     
        
    evaluate_dataset = TensorDataset(evaluate_data, evaluate_labels)
    evaluate_loader = DataLoader(evaluate_dataset, batch_size=1, shuffle=False)

    # Test the model
    criterion = nn.CrossEntropyLoss()
    final_loss = []
    final_prediction = []
    final_prediction_for_score = []
    final_groundtruth = []
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in evaluate_loader:
            inputs, labels = inputs.to(device), labels.to(device)        
            outputs = attack_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            final_loss.append(loss.detach().cpu().numpy())
            final_prediction_for_score.extend(F.sigmoid(outputs.data)[:,1].detach().cpu().numpy())
            final_prediction.extend(outputs.data.detach().cpu().numpy())
            final_groundtruth.extend(labels.detach().cpu().numpy())        

    fpr_improved, tpr_improved, auc_improved, acc_improved = sweep(np.array(final_prediction_for_score), np.array(final_groundtruth))
    low_improved = tpr_improved[np.where(fpr_improved<.01)[0][-1]]
    print(util.cyan('[WSA] Attack Ours (online, fixed variance)   AUC {:.4f}, Accuracy {:.4f}, TPR@1%FPR of {:.4f}'.format(auc_improved, acc_improved, low_improved)))
    print(util.cyan(f'Mislabel rate is ===================== {mis_rate}'))
    print(util.cyan(f'Non-train pool size is ===================== {nontrain_length}'))
    print(util.cyan(f'Train pool size is ===================== {length_train}'))
    print(util.cyan(f'lambda_selection is ===================== {train_threshold}'))

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=args.seed)

    main(args, device)