import torch
import pickle
import pandas as pd
import numpy as np
import argparse

"""
从训练好的包含文本嵌入的模型中提取最终的 GO 嵌入
拼接图结构嵌入和文本嵌入
"""

parser = argparse.ArgumentParser(description='Extract GO embeddings with text from trained model',
                                 add_help=False)

parser.add_argument('--terms-file',
                    '-tf',
                    default='data/terms_all.pkl',
                    type=str,
                    help='A DataFrame stored all terms')
parser.add_argument('--text-emb-file',
                    '-tef',
                    default='data/terms_text_embeddings.pkl',
                    type=str,
                    help='Text embeddings file')
parser.add_argument('--model-file',
                    '-mf',
                    default='models_with_text/part_order_text_400.pth',
                    type=str,
                    help='Model weights file with text embeddings')
parser.add_argument('--out-file',
                    '-of',
                    default='data/terms_all_embeddings_with_text.pkl',
                    type=str,
                    help='Output file for combined embeddings')


def main(terms_file, text_emb_file, model_file, out_file):
    print(f"Loading model from {model_file}...")
    model = torch.load(model_file, map_location="cpu")
    
    # 提取图结构嵌入
    graph_emb = model["graph_embedding"].numpy()
    concat_ratio = model.get("concat_ratio", 0.5)
    
    print(f"Graph embedding shape: {graph_emb.shape}")
    print(f"Concat ratio: {concat_ratio}")
    
    # 加载术语列表
    print(f"Loading terms from {terms_file}...")
    with open(terms_file, "rb") as fd:
        terms_all = pickle.load(fd)
        terms_all = list(terms_all["terms"])
    
    print(f"Total terms: {len(terms_all)}")
    
    # 加载文本嵌入
    print(f"Loading text embeddings from {text_emb_file}...")
    text_emb_df = pd.read_pickle(text_emb_file)
    text_emb_dict = dict(zip(text_emb_df['terms'], text_emb_df['text_embeddings']))
    
    # 获取文本嵌入维度
    sample_text_emb = list(text_emb_dict.values())[0]
    text_dim = len(sample_text_emb)
    graph_dim = graph_emb.shape[1]
    
    print(f"Text embedding dimension: {text_dim}")
    print(f"Graph embedding dimension: {graph_dim}")
    
    # 构建文本嵌入矩阵
    text_emb_matrix = np.zeros((len(terms_all), text_dim), dtype=np.float32)
    for i, term in enumerate(terms_all):
        if term in text_emb_dict:
            text_emb_matrix[i] = text_emb_dict[term]
        else:
            print(f"Warning: No text embedding for {term}")
    
    # 投影文本嵌入到与图嵌入相同的维度
    # 使用模型中的投影层权重（如果存在）
    if 'net' in model and 'text_projection.weight' in model['net']:
        print("Using text projection from model...")
        text_proj_weight = model['net']['text_projection.weight'].numpy()
        text_proj_bias = model['net']['text_projection.bias'].numpy()
        text_emb_projected = text_emb_matrix @ text_proj_weight.T + text_proj_bias
    else:
        print("No projection layer found, using linear projection...")
        # 如果没有投影层，使用简单的线性变换
        # 创建一个随机投影矩阵（或者使用 PCA）
        from sklearn.decomposition import PCA
        pca = PCA(n_components=graph_dim)
        text_emb_projected = pca.fit_transform(text_emb_matrix)
    
    print(f"Projected text embedding shape: {text_emb_projected.shape}")
    
    # 按权重拼接
    graph_emb_weighted = graph_emb * concat_ratio
    text_emb_weighted = text_emb_projected * (1 - concat_ratio)
    
    # 拼接
    combined_emb = np.concatenate([graph_emb_weighted, text_emb_weighted], axis=1)
    
    print(f"Combined embedding shape: {combined_emb.shape}")
    
    # 通过模型的 fc 层进行最终投影（如果需要）
    if 'net' in model:
        print("Applying final projection layers...")
        fc_weight = model['net']['fc.weight'].numpy()
        fc_bias = model['net']['fc.bias'].numpy()
        fc2_weight = model['net']['fc2.weight'].numpy()
        fc2_bias = model['net']['fc2.bias'].numpy()
        
        # 第一层
        x = combined_emb @ fc_weight.T + fc_bias
        x = np.maximum(0, x)  # ReLU
        
        # 第二层
        final_emb = x @ fc2_weight.T + fc2_bias
        
        # L2 归一化
        final_emb = final_emb / (np.linalg.norm(final_emb, axis=1, keepdims=True) + 1e-9)
        
        print(f"Final embedding shape: {final_emb.shape}")
    else:
        final_emb = combined_emb
    
    # 创建术语到嵌入的映射
    terms_emb_dict = {}
    for i in range(len(terms_all)):
        terms_emb_dict[terms_all[i]] = final_emb[i]
    
    # 保存
    print(f"Saving combined embeddings to {out_file}...")
    df_terms_all_embeddings = pd.DataFrame({
        'terms': list(terms_emb_dict.keys()), 
        'embeddings': list(terms_emb_dict.values())
    })
    df_terms_all_embeddings.to_pickle(out_file)
    
    print("Done!")
    print(f"Final embedding dimension: {final_emb.shape[1]}")
    print(f"Total terms processed: {len(terms_emb_dict)}")
    
    # 保存一些统计信息
    print("\n=== Embedding Statistics ===")
    print(f"Mean norm: {np.mean(np.linalg.norm(final_emb, axis=1)):.4f}")
    print(f"Std norm: {np.std(np.linalg.norm(final_emb, axis=1)):.4f}")
    print(f"Min value: {np.min(final_emb):.4f}")
    print(f"Max value: {np.max(final_emb):.4f}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.terms_file, args.text_emb_file, args.model_file, args.out_file)

