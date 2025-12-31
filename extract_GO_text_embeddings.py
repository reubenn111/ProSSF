"""提取GO文本嵌入 - 新增脚本（必需，因为原项目没有此功能）"""
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils import Ontology

parser = argparse.ArgumentParser(description='Extract GO text embeddings')
parser.add_argument('--go-file', '-gf', default='../../data/go.obo', type=str)
parser.add_argument('--terms-file', '-tf', default='../../data/terms_all.pkl', type=str)
parser.add_argument('--out-file', '-of', default='../../data/terms_text_embeddings.pkl', type=str)
parser.add_argument('--model-name', '-m', default='bert-base-uncased', type=str)
parser.add_argument('--batch-size', '-b', type=int, default=32)
parser.add_argument('--device', '-d', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_go_text(go_term, ontology):
    term = ontology.get_term(go_term)
    if term is None:
        return f"Gene ontology term {go_term}"
    text_parts = []
    if 'name' in term:
        text_parts.append(term['name'])
    if 'def' in term:
        definition = term['def']
        if '"' in definition:
            definition = definition.split('"')[1]
        text_parts.append(definition)
    return '. '.join(text_parts) if text_parts else f"Gene ontology term {go_term}"

def main(go_file, terms_file, out_file, model_name, batch_size, device):
    print(f"Loading GO ontology...")
    go = Ontology(go_file, with_rels=True, include_alt_ids=False)
    
    print(f"Loading terms...")
    with open(terms_file, 'rb') as fd:
        terms_df = pickle.load(fd)
        terms = list(terms_df['terms'])
    
    print(f"Extracting text descriptions...")
    go_texts = [get_go_text(term, go) for term in tqdm(terms)]
    
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    
    print("Encoding texts...")
    all_embeddings = []
    for i in tqdm(range(0, len(go_texts), batch_size)):
        batch_texts = go_texts[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        embeddings = mean_pooling(outputs, encoded['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    print(f"Text embeddings shape: {all_embeddings.shape}")
    
    df = pd.DataFrame({'terms': terms, 'text_embeddings': list(all_embeddings)})
    df.to_pickle(out_file)
    print(f"Saved to {out_file}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.go_file, args.terms_file, args.out_file, args.model_name, args.batch_size, args.device)
