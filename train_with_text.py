import argparse
import logging
import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import ConPairLoss
from utils import set_random_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

'''
修改版本的 PO2Vec 训练脚本
支持将 GO 图结构嵌入与文本嵌入拼接
'''

def main():
    parser = argparse.ArgumentParser(description='PO2Vec with text embeddings')
    parser.add_argument('--data_path',
                        '-dp',
                        default='data',
                        help='Path to store data')
    parser.add_argument('--model_path',
                        '-mp',
                        default='models_with_text',
                        help='Path to save model')
    parser.add_argument('--summary_path',
                        '-sp',
                        default='logs',
                        help='Path to save summary')
    parser.add_argument('--model_load',
                        '-ml',
                        type=int,
                        default=0,
                        help='Load model epoch and the model must exist')
    parser.add_argument('--concat_ratio',
                        '-cr',
                        type=float,
                        default=0.5,
                        help='Ratio for graph:text embeddings (0.5 means 5:5)')
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()

    # check data_path
    if not os.path.exists(args.data_path):
        print('Unable to find data path %s.' % args.data_path)
        return

    # check model_path
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        print('Create %s to save model.' % args.model_path)
    
    model_prefix = r'part_order_text_'
    model_suffix = r'.pth'
    model_file = os.path.join(args.model_path,
                              model_prefix + r'%d' + model_suffix)

    # check summary_path
    if not os.path.exists(args.summary_path):
        os.makedirs(args.summary_path)
        print('Create %s to save summary.' % args.summary_path)

    # log setting
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler(
        os.path.join(args.summary_path, 'exper_triple_text.txt'))
    logger.addHandler(f_handler)

    # File paths
    go_file = os.path.join(args.data_path, u'go.obo')
    out_terms_file = os.path.join(args.data_path, u'terms_all.pkl')
    pair_file = os.path.join(args.data_path, u'contra_part_pairs_all.pkl')
    contrast_file = os.path.join(args.data_path, u'contrast_pairs.pkl')
    text_emb_file = os.path.join(args.data_path, u'terms_text_embeddings.pkl')

    # Check if text embeddings exist
    if not os.path.exists(text_emb_file):
        print(f"Text embeddings file not found: {text_emb_file}")
        print("Please run extract_GO_text_embeddings.py first!")
        return

    # Hyper parameters
    params = {
        'learning_rate': 5e-2,
        'epochs': 400,
        'train_batch_size': 16,
    }

    args.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    set_random_seed(42)

    # Load terms
    print("Loading GO terms...")
    terms_df = pd.read_pickle(out_terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)
    print(f"Total GO terms: {nb_classes}")

    # Load text embeddings
    print("Loading text embeddings...")
    text_emb_df = pd.read_pickle(text_emb_file)
    text_emb_dict = dict(zip(text_emb_df['terms'], text_emb_df['text_embeddings']))
    
    # Create text embedding matrix aligned with terms
    # 获取文本嵌入维度
    sample_text_emb = list(text_emb_dict.values())[0]
    text_dim = len(sample_text_emb)
    print(f"Text embedding dimension: {text_dim}")
    
    text_emb_matrix = np.zeros((nb_classes, text_dim), dtype=np.float32)
    for term, idx in terms_dict.items():
        if term in text_emb_dict:
            text_emb_matrix[idx] = text_emb_dict[term]
        else:
            # 如果某个术语没有文本嵌入，使用零向量
            print(f"Warning: No text embedding for {term}, using zero vector")
    
    # Convert to torch tensor
    text_emb_tensor = torch.from_numpy(text_emb_matrix).to(args.device)
    print(f"Text embedding matrix shape: {text_emb_tensor.shape}")

    # Load pairs
    print("Loading training pairs...")
    with open(pair_file, 'rb') as fd:
        pair_list_file = pickle.load(fd)

    pair_list = []
    for i in range(len(pair_list_file)):
        pair_list += pair_list_file[i]

    with open(contrast_file, 'rb') as fd:
        contrast_dict = pickle.load(fd)

    # Create dataset
    neg_num = 80
    pair_data = con_pair_dataset_with_text(
        pair_list,
        contrast_dict,
        terms,
        terms_dict,
        text_emb_tensor,
        neg_num=neg_num,
        neg=0.5,
        neg1_len=0.25
    )
    print('Size of train dataset:', len(pair_data))

    train_dataloader = DataLoader(pair_data,
                                  batch_size=params['train_batch_size'],
                                  shuffle=True)

    # Create model with text embeddings
    model = PairModelWithText(
        nb_classes, 
        neg_num + 2,
        text_emb_tensor,
        concat_ratio=args.concat_ratio
    )
    model.to(args.device)
    
    print(f"Model created with concat_ratio={args.concat_ratio}")
    print(f"Graph embedding dim: {model.graph_emb_dim}")
    print(f"Text embedding dim: {model.text_emb_dim}")
    print(f"Final embedding dim after concat: {model.final_emb_dim}")

    # Load checkpoint if exists
    checkpoint = None
    files = os.listdir(args.model_path)
    epoch_list = [
        int(f[len(model_prefix):-len(model_suffix)]) for f in files
        if f[:len(model_prefix)] == model_prefix and f.endswith(model_suffix)
    ]
    
    if len(epoch_list) > 0 and args.model_load > 0:
        max_epoch_file = model_file % args.model_load if args.model_load in epoch_list else ''
    else:
        max_epoch_file = r''

    if os.path.exists(max_epoch_file):
        checkpoint = torch.load(max_epoch_file, map_location=args.device)
        model.load_state_dict(checkpoint['net'], strict=True)
        print('Load model from file:', max_epoch_file)
        start_epoch = args.model_load
    else:
        print('No model to load, training from scratch.')
        start_epoch = 0

    crition = ConPairLoss(neg_num=neg_num)

    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()),
                           lr=params['learning_rate'])
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('Training starts:')
    for epoch in range(start_epoch + 1, params['epochs'] + 1):
        print('--------Epoch %02d--------' % epoch)
        logger.info('--------Epoch %02d--------\n' % epoch)
        train_loss = train(model, args.device, optimizer, crition,
                           train_dataloader, args)

        logger.info('train_loss:{}, epoch:{}\n'.format(train_loss, epoch))

        # Save model every 40 epochs
        if epoch % 40 == 0:
            checkpoint = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'graph_embedding': model.graph_embedding.weight.data,
                'concat_ratio': args.concat_ratio,
                'text_emb_dim': model.text_emb_dim,
                'graph_emb_dim': model.graph_emb_dim
            }
            torch.save(checkpoint, model_file % epoch)
            print('Model parameters are saved!')

    print("Training completed!")


class PairModelWithText(nn.Module):
    """
    改进的 PairModel，支持图结构嵌入和文本嵌入的拼接
    """
    def __init__(self, input_emb, sample_size, text_emb_tensor, concat_ratio=0.5):
        super().__init__()
        
        # 文本嵌入（固定不训练）
        self.text_emb_tensor = text_emb_tensor
        self.text_emb_dim = text_emb_tensor.shape[1]
        
        # 计算拼接后的维度（按比例分配）
        # 如果 concat_ratio = 0.5，则图嵌入和文本嵌入各占 128 维
        self.concat_ratio = concat_ratio
        self.graph_emb_dim = 128  # 图结构嵌入部分的维度（降低以便拼接）
        
        # 图结构嵌入（可训练）
        self.graph_embedding = nn.Embedding(input_emb, self.graph_emb_dim)
        
        # 文本嵌入投影层（将文本嵌入投影到合适维度）
        self.text_projection = nn.Linear(self.text_emb_dim, self.graph_emb_dim)
        
        # 拼接后的维度
        self.final_emb_dim = self.graph_emb_dim * 2  # graph + text
        
        # 后续处理层
        self.temp_dim = 1024
        self.output_dim = 512
        self.fc = nn.Linear(self.final_emb_dim, self.temp_dim)
        self.fc2 = nn.Linear(self.temp_dim, self.output_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 获取图结构嵌入
        graph_emb = self.graph_embedding(x)  # [batch_size, seq_len, graph_emb_dim]
        
        # 获取文本嵌入
        text_emb = self.text_emb_tensor[x]  # [batch_size, seq_len, text_emb_dim]
        text_emb = self.text_projection(text_emb)  # [batch_size, seq_len, graph_emb_dim]
        
        # 按比例加权拼接
        graph_emb_weighted = graph_emb * self.concat_ratio
        text_emb_weighted = text_emb * (1 - self.concat_ratio)
        
        # 拼接
        combined_emb = torch.cat([graph_emb_weighted, text_emb_weighted], dim=-1)
        
        # 通过全连接层
        x = self.fc(combined_emb)
        x = F.relu(x)
        x = self.fc2(x)
        
        # L2 归一化
        x = x / torch.norm(x, dim=-1, keepdim=True)
        
        return x


class con_pair_dataset_with_text(torch.utils.data.Dataset):
    """
    修改后的数据集类，保持与原版相同的采样逻辑
    """
    def __init__(self,
                 con_pair,
                 contrast_dict,
                 terms,
                 terms_dict,
                 text_emb_tensor,
                 neg_num=80,
                 neg=0.5,
                 neg1_len=0.25):
        super().__init__()
        self.len_df = len(con_pair)
        self.n_cc = list(contrast_dict['n_cc'])
        self.n_bp = list(contrast_dict['n_bp'])
        self.n_mf = list(contrast_dict['n_mf'])
        self.terms = terms
        self.contrast_dict = contrast_dict
        self.terms_dict = terms_dict
        self.text_emb_tensor = text_emb_tensor
        self.neg_num = neg_num
        self.con_pair = con_pair
        self.neg = neg
        self.neg1_len = neg1_len

    def __len__(self):
        return self.len_df

    def __getitem__(self, idx):
        import random
        from numpy.random import randint
        
        terms_list = [self.con_pair[idx][0], self.con_pair[idx][1]]
        
        # negs1: 直接可达负样本
        negs1 = set()
        neg1_len = min(len(self.con_pair[idx][2][0]),
                       int(self.neg_num * self.neg1_len))
        if neg1_len > 0:
            negs1 = set(random.sample(self.con_pair[idx][2][0], k=neg1_len))
        negs1 = list(negs1)
        random.shuffle(negs1)

        # negs2: 间接可达负样本
        negs2 = set()
        neg2_len = int((self.neg_num - neg1_len) * self.neg)
        if len(self.contrast_dict[self.con_pair[idx][0]]) <= neg2_len:
            negs2 = negs2 | set(
                random.sample(self.contrast_dict[self.con_pair[idx][0]],
                              k=len(
                                  self.contrast_dict[self.con_pair[idx][0]])))
            negs2 = negs2 | set(
                random.sample(self.contrast_dict[self.con_pair[idx][0]],
                              k=neg2_len -
                              len(self.contrast_dict[self.con_pair[idx][0]])))
        else:
            negs2 = negs2 | set(
                random.sample(self.contrast_dict[self.con_pair[idx][0]],
                              k=neg2_len))
        negs2 = list(negs2)
        random.shuffle(negs2)

        # negs3: 不可达负样本
        neg_len = neg1_len + neg2_len
        neg_num = self.neg_num - neg_len
        negs3 = set()
        
        if self.contrast_dict[self.terms[terms_list[0]]] == 'GO:0005575':
            while len(negs3) < neg_num // 3:
                m = randint(0, len(self.n_mf) - 1)
                if self.terms_dict[self.n_mf[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_mf[m]])
            while len(negs3) < neg_num:
                m = randint(0, len(self.n_bp) - 1)
                if self.terms_dict[self.n_bp[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_bp[m]])
        elif self.contrast_dict[self.terms[terms_list[0]]] == 'GO:0003674':
            while len(negs3) < neg_num // 5:
                m = randint(0, len(self.n_cc) - 1)
                if self.terms_dict[self.n_cc[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_cc[m]])
            while len(negs3) < neg_num:
                m = randint(0, len(self.n_bp) - 1)
                if self.terms_dict[self.n_bp[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_bp[m]])
        elif self.contrast_dict[self.terms[terms_list[0]]] == 'GO:0008150':
            while len(negs3) < neg_num // 3:
                m = randint(0, len(self.n_cc) - 1)
                if self.terms_dict[self.n_cc[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_cc[m]])
            while len(negs3) < neg_num:
                m = randint(0, len(self.n_mf) - 1)
                if self.terms_dict[self.n_mf[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_mf[m]])
        negs3 = list(negs3)
        random.shuffle(negs3)

        # 生成权重
        neg1_num = [neg1_len for i in range(neg1_len)]
        neg2_num = [neg2_len for i in range(neg2_len)]
        neg3_num = [neg_num for i in range(neg_num)]
        neg_num_list = neg1_num + neg2_num + neg3_num
        neg_num_array = 1 / np.array(neg_num_list)
        
        terms_list = terms_list + negs1 + negs2 + negs3
        
        return torch.LongTensor(terms_list).view(
            len(terms_list)), torch.from_numpy(neg_num_array)


def train(model, device, optimizer, crition, train_dataloader, args):
    """训练函数"""
    model.train()
    train_loss = 0

    for index, (x, rate) in enumerate(tqdm(train_dataloader)):
        x = Variable(x.squeeze().to(device))
        rate = rate.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = crition(output, rate)
        train_loss += loss.item() * x.shape[0]
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader.dataset)
    print('Train set: Average loss: {:.4f}'.format(train_loss))

    return train_loss


if __name__ == '__main__':
    main()

