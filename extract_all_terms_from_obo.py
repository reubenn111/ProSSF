import argparse
import pandas as pd
from po2go.po2vec.utils import Ontology

'''
3:
实例化 utils.py 中定义的 Ontology 类，加载并解析 go.obo 文件。

通过 go.ont.keys() 获取所有非过时 (non-obsolete) 的 GO 词条 ID。

将这些 ID 存储在一个 Pandas DataFrame 中，并序列化为 .pkl 文件。这个文件构成了后续所有步骤中 GO 词条的全集。

'''
parser = argparse.ArgumentParser(description='extract all terms in a go.obo file.',
                                 add_help=False)

parser.add_argument('--go-file',
                    '-g',
                    default=r'E:\08_python_daima\protein-annotation-master\data\go.obo',
                    #改成绝对路径，default='data/go.obo',

                    type=str,
                    help='go file downloaded from Gene Ontology website')
parser.add_argument('--out-file',
                    '-o',
                    #改成绝对路径，default='data/terms_all.pkl',
                    default=r'E:\08_python_daima\protein-annotation-master\data\terms_all.pkl',
                    type=str,
                    help='terms stored as a DataFrame in pkl format')


def main(go_file, out_file):
    go = Ontology(go_file, with_rels=True, include_alt_ids=False)
    df_terms = pd.DataFrame({'terms': go.ont.keys()})
    df_terms.to_pickle(out_file)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.go_file, args.out_file)
