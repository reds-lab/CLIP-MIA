import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import time

import open_clip
from open_clip import tokenizer, tokenize
from data import get_data, get_data_val
from text_preprocessing import text_preprocessing

def select_pseudotrain(args, target_model, selected_nt_txt, selected_nt_url, dataloader, train_threshold, preprocess_train, preprocess_val, device, length):

    selected_t_txt = [] ## to save the train text information to exclude those data in evaluation section
    selected_t_url = [] ## to save the train text information to exclude those data in evaluation section
    selected_t_cs_lst_tar = []
    selected_t_feat_lst_tar = []
     
    ### Train dataloader
    start_time = time.time()
        
    cnt_train = 0
    for i, batch in enumerate( dataloader ): ## LAION
        
        train_text = [text_preprocessing(q) for q in batch[1]]   
        train_url = [d['url'] for d in batch[2]]     
          
        common = np.intersect1d(np.array(train_text), np.array(selected_nt_txt)) 
        x_ind = np.where(np.isin(np.array(train_text), common))[0]
        common_2 = np.intersect1d(np.array(train_url), np.array(selected_nt_url)) 
        x_ind_2nd = np.where(np.isin(np.array(train_url), common_2))[0]
        combined_x_ind = np.union1d(x_ind, x_ind_2nd)
        
        if len(combined_x_ind) > 0:
            selected_ind = np.setdiff1d(np.arange(len(batch[0])), combined_x_ind)
        else:
            selected_ind = np.arange(len(batch[0]))
            
        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        
        cnt_train += len(images)

        images = images.to(device)
        texts = texts.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)  
        cs_2 = torch.diagonal(image_features2@text_features2.T)    

        ## Batch-wise approach     
        train_ind = torch.where(cs_2 >= train_threshold)[0] ## we do assume the knowledge for non-train data distribution, so we only sample train data samples here
        train_ind = train_ind.detach().cpu()
        
        selected_t_txt.extend( np.array(train_text)[selected_ind][train_ind.numpy()] ) 
        selected_t_url.extend( np.array(train_url)[selected_ind][train_ind.numpy()] )
        selected_t_cs_lst_tar.extend( cs_2[train_ind].detach().cpu().numpy() ) 
        selected_t_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1)[train_ind].detach().cpu() )    

        if cnt_train >= length:
            break
    
    true_train = len(selected_t_cs_lst_tar)
    
    ################################################## 
    ## nontrain data | collect the data from the second val loader : CC3M
    args.val_data = args.val_data_train_1
    args.val_num_samples = args.val_num_samples_train_1
    valloader = get_data_val(args, (preprocess_val, preprocess_val))
    valloader = valloader.dataloader    

    ### Nontrain data from valloader
    start_time = time.time()

    CC3M_LAION_commonset = np.load('./CC3M_LAION_commonset.npy')
    CC3M_LAION_unqiue_commonset = np.load('./CC3M_LAION_unqiue_commonset.npy')
    CC3M_LAION_url_commonset = np.load('./CC3M_LAION_url_commonset.npy')

    cnt_nontrain = 0
    for i, batch in enumerate( valloader ): 
        
        train_text = [text_preprocessing(q) for q in batch[1]]   
        train_url = [d['url'] for d in batch[2]]     

        common= np.intersect1d(np.array(train_text), CC3M_LAION_commonset) ## to exclude the overlapped non-train pairs with train pairs
        x_ind = np.where(np.isin(np.array(train_text), common))[0]   
        common2 = np.intersect1d(np.array(train_text), np.array(selected_nt_txt))
        x_ind_2nd = np.where(np.isin(np.array(train_text), common2))[0]
        common_3 = np.intersect1d(np.array(train_url), CC3M_LAION_url_commonset) ## to exclude the overlapped non-train pairs with train pairs [url-wise]
        x_ind_3rd = np.where(np.isin(np.array(train_url), common_3))[0]     
        common_4 = np.intersect1d(np.array(train_url), np.array(selected_nt_url))
        x_ind_4th = np.where(np.isin(np.array(train_url), common_4))[0]
        combined_x_ind = np.union1d(np.union1d(np.union1d(x_ind, x_ind_2nd), x_ind_3rd), x_ind_4th)
        
        if len(combined_x_ind) > 0:
            selected_ind = np.setdiff1d(np.arange(len(batch[0])), combined_x_ind)
        else:
            selected_ind = np.arange(len(batch[0]))

        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        
        cnt_nontrain += len(images)

        images = images.to(device)
        texts = texts.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)
        cs_2 = torch.diagonal(image_features2@text_features2.T)

        train_ind = torch.where(cs_2 >= train_threshold)[0] 
        train_ind = train_ind.detach().cpu()
        
        selected_t_txt.extend( np.array(train_text)[selected_ind][train_ind.numpy()] ) 
        selected_t_url.extend( np.array(train_url)[selected_ind][train_ind.numpy()] )
        selected_t_cs_lst_tar.extend( cs_2[train_ind].detach().cpu().numpy() ) 
        selected_t_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1)[train_ind].detach().cpu() )    

        if cnt_nontrain >= length:
            break
    
    ################################################## 
    ## nontrain data | collect the data from the second val loader : CC12M    
    args.val_data = args.val_data_train_2
    args.val_num_samples = args.val_num_samples_train_2
    cc12m_valoader = get_data_val(args, (preprocess_val, preprocess_val))
    cc12m_valoader = cc12m_valoader.dataloader
    
    start_time = time.time()

    CC12M_LAION_commonset = np.load('./CC12M_LAION_commonset.npy')
    CC12M_LAION_unqiue_commonset = np.load('./CC12M_LAION_unqiue_commonset.npy')
    CC12M_LAION_url_commonset = np.load('./CC12M_LAION_url_commonset.npy')
    
    cnt_nontrain = 0
    for i, batch in enumerate( cc12m_valoader ): 
        
        train_text = [text_preprocessing(q) for q in batch[1]]   
        train_url = [d['url'] for d in batch[2]]     

        common= np.intersect1d(np.array(train_text), CC12M_LAION_commonset) ## to exclude the overlapped non-train pairs with train pairs
        x_ind = np.where(np.isin(np.array(train_text), common))[0]   
        common2 = np.intersect1d(np.array(train_text), np.array(selected_nt_txt))
        x_ind_2nd = np.where(np.isin(np.array(train_text), common2))[0]
        common_3 = np.intersect1d(np.array(train_url), CC12M_LAION_url_commonset) ## to exclude the overlapped non-train pairs with train pairs [url-wise]
        x_ind_3rd = np.where(np.isin(np.array(train_url), common_3))[0]     
        common_4 = np.intersect1d(np.array(train_url), np.array(selected_nt_url))
        x_ind_4th = np.where(np.isin(np.array(train_url), common_4))[0]
        combined_x_ind = np.union1d(np.union1d(np.union1d(x_ind, x_ind_2nd), x_ind_3rd), x_ind_4th)
        
        if len(combined_x_ind) > 0:
            selected_ind = np.setdiff1d(np.arange(len(batch[0])), combined_x_ind)
        else:
            selected_ind = np.arange(len(batch[0]))

        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        
        cnt_nontrain += len(images)
        images = images.to(device)
        texts = texts.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)
        cs_2 = torch.diagonal(image_features2@text_features2.T)
        train_ind = torch.where(cs_2 >= train_threshold)[0] 
        train_ind = train_ind.detach().cpu()

        selected_t_txt.extend( np.array(train_text)[selected_ind][train_ind.numpy()] ) 
        selected_t_url.extend( np.array(train_url)[selected_ind][train_ind.numpy()] )
        selected_t_cs_lst_tar.extend( cs_2[train_ind].detach().cpu().numpy() ) 
        selected_t_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1)[train_ind].detach().cpu() )    
        
        if cnt_nontrain >= length:
            break

    ################################################## 
    ## nontrain data | collect the data from the second val loader : MSCOCO
    args.val_data = args.val_data_train_3
    args.val_num_samples = args.val_num_samples_train_3
    mscoco_valoader =  get_data_val(args, (preprocess_val, preprocess_val))
    mscoco_valoader = mscoco_valoader.dataloader

    start_time = time.time()

    MSCOCO_LAION_commonset = np.load('./MSCOCO_LAION_commonset.npy')
    MSCOCO_LAION_unqiue_commonset = np.load('./MSCOCO_LAION_unqiue_commonset.npy')

    ## nontrain data 
    cnt_nontrain = 0
    for i, batch in enumerate( mscoco_valoader ): 
        
        train_text = [text_preprocessing(q) for q in batch[1]]   
        train_url = [d['url'] for d in batch[2]]             
        
        common= np.intersect1d(np.array(train_text), MSCOCO_LAION_commonset) ## to exclude the overlapped non-train pairs with train pairs
        x_ind = np.where(np.isin(np.array(train_text), common))[0]   
        common2 = np.intersect1d(np.array(train_text), np.array(selected_nt_txt))
        x_ind_2nd = np.where(np.isin(np.array(train_text), common2))[0]  
        common_3 = np.intersect1d(np.array(train_url), np.array(selected_nt_url))
        x_ind_3rd = np.where(np.isin(np.array(train_url), common_3))[0]
        combined_x_ind = np.union1d(np.union1d(x_ind, x_ind_2nd), x_ind_3rd)

        if len(combined_x_ind) > 0:
            selected_ind = np.setdiff1d(np.arange(len(batch[0])), combined_x_ind)
        else:
            selected_ind = np.arange(len(batch[0]))
            
        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        cnt_nontrain += len(images)

        images = images.to(device)
        texts = texts.to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)
        cs_2 = torch.diagonal(image_features2@text_features2.T)
        train_ind = torch.where(cs_2 >= train_threshold)[0] 
        train_ind = train_ind.detach().cpu()
        
        selected_t_txt.extend( np.array(train_text)[selected_ind][train_ind.numpy()] )
        selected_t_url.extend( np.array(train_url)[selected_ind][train_ind.numpy()] )
        selected_t_cs_lst_tar.extend( cs_2[train_ind].detach().cpu().numpy() ) 
        selected_t_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1)[train_ind].detach().cpu() )    
       
        if cnt_nontrain >= length:
            break
    pseudo_train = len(selected_t_cs_lst_tar) - true_train
    
    return selected_t_txt, selected_t_url, selected_t_cs_lst_tar, selected_t_feat_lst_tar,  true_train, pseudo_train
