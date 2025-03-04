#!/usr/bin/env python

import click as ck
import pandas as pd
import numpy as np
import gzip
import logging
from collections import deque
import os

logging.basicConfig(level=logging.INFO)

DATA_ROOT = 'data/'

INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X', '*'])
MAXLEN = 1002
EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

SPLIT = 0.8
annot_num_bp = 5
annot_num_mf = 22
annot_num_cc = 5

@ck.command()
@ck.option(
    '--swissprot-file', '-sf', default='data/uniprot-reviewed_yes+AND+organism__maize_.txt.gz',
    help='UniProt/SwissProt knowledgebase file in text format (archived)')
@ck.option(
    '--out-file', '-o', default='data/swissprot.pkl',
    help='Result file with a list of proteins, sequences and annotations')


def main(swissprot_file, out_file):
    subontologies = ['bp','mf','cc']


    go = get_gene_ontology('go.obo')

    if not os.path.exists(out_file):
        # 加载生成数据
        accessions, sequences, annotations = load_data(swissprot_file)
        # 生成表格
        df = pd.DataFrame({
            'accessions': accessions,
            'sequences': sequences,
            'annotations': annotations,
        })

        df.to_pickle(out_file)
        logging.info('Successfully saved %d proteins' % (len(df),))

    for subontology in subontologies:

        print('***subontology***:',subontology)
        GO_ID = FUNC_DICT[subontology]

        #选出作为标签的GO注释
        global func
        func = deque()
        dfs(go,GO_ID) # 以mf为例，寻找出所有属于mf的GO注释
        func.remove(GO_ID)  # 移除functions队列里边mf的id号'GO:0003674'
        func = list(func)
        #print(len(func)) # 所有属于mf的GO注释的个数
        global funcset
        funcset = set(func)
        global go_indexes
        go_indexes = dict()  # 创建一个空字典dict()
        for ind, go_id in enumerate(func):  # 加索引列
            go_indexes[go_id] = ind
        get_functions(go,subontology)

        #生成bp、mf、cc对应的训练集、测试集
        func_df = pd.read_pickle(DATA_ROOT + subontology + '.pkl')
        global functions
        functions = func_df['functions'].values
        global func_set
        func_set = get_go_set(go, GO_ID)  # 以mf为例，func_set存储了go.obo文件中所有属于mf的function注释，相当于get_functions.py里的dfs(go_id)
        #print(len(functions))
        global go_dataindexes
        go_dataindexes = dict()
        for ind, go_id in enumerate(functions):  # 加索引列
            go_dataindexes[go_id] = ind
        run(go,subontology,GO_ID)


def load_data(swissprot_file):
    accessions = list()
    sequences = list()
    annotations = list()
    with gzip.open(swissprot_file, 'rt') as f:
        prot_id = ''
        prot_ac = ''
        seq = ''
        annots = list()
        ipros = list()
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '':
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                prot_id = items[1]
                annots = list()
                ipros = list()
                seq = ''
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
                prot_ac = prot_ac.split(';')[0]
            elif items[0] == 'OX' and len(items) > 1:
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)
                if items[0] == 'InterPro':
                    ipro_id = items[1]
                    ipros.append(ipro_id)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq

        accessions.append(prot_ac)  # UniProt中的Entry name列，是蛋白质ID简要名字
        sequences.append(seq)
        annotations.append(annots)
    return accessions, sequences, annotations

def run(go,subontology,GO_ID):
    df = loaddata(go,GO_ID)
    index = df.index.values
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_n = int(len(df) * SPLIT)
    train_df = df.loc[index[:train_n]]
    test_df = df.loc[index[train_n:]]

    print('训练集个数：',len(train_df), ';','测试集个数：',len(test_df))
    print()
    train_df.to_pickle(DATA_ROOT + 'train-' + subontology + '.pkl')
    test_df.to_pickle(DATA_ROOT + 'test-' + subontology + '.pkl')

def is_ok(seq):
    if len(seq) > MAXLEN:
        return False
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True

def loaddata(go,GO_ID):
    gos = list()
    labels = list()
    sequences = list()
    accessions = list()

    df = pd.read_pickle(DATA_ROOT + 'swissprot.pkl')

    # Filtering data by sequences
    index = list()
    for i, row in df.iterrows():
        if is_ok(row['sequences']):
            index.append(i)
    df = df.loc[index]

    for i, row in df.iterrows():
        go_list = []
        for item in row['annotations']:
            items = item.split('|')
            #if items[1] in EXP_CODES:
            #    go_list.append(items[0])
            go_list.append(items[0])
        go_set = set()
        for go_id in go_list:
            if go_id in func_set: # 如果go_id在mf.pkl的集合里
                go_set |= get_anchestors(go, go_id) # '|='是把go_set和get_anchestors这两个集合合并，没有重复的元素，最终go_set存储了该GO对应的所有上层分类的GO(根据标签'is_a'查找)；
                                                    # 内部循环结束后，最终go_set存储了这一行数据里面所有GO的对应的所有上层分类的GO
        if not go_set or GO_ID not in go_set: # 以mf为例，如果在swissprot_exp.pkl中该行数据的id属于mf的，则往下执行，即执行以下一系列append()，否则continue跳过本次循环而执行下一次循环
            continue
        go_set.remove(GO_ID)
        gos.append(go_list)
        accessions.append(row['accessions'])
        seq = row['sequences']
        sequences.append(seq)

        label = np.zeros((len(functions),), dtype='int32') # mf.pkl中有多少个functions(即有多少个GO)，则赋值label多少个0

        for go_id in go_set: # go_set存储了该行数据内所有GOid的所有上层分类GO
            if go_id in go_dataindexes: # go_indexes是mf.pkl里的functions集合
                label[go_dataindexes[go_id]] = 1 # 如果该GOid在mf.pkl里也存在，则把1赋值到该GOid在mf.pkl里对应的位置上(对应的索引)
        labels.append(label)
        #print('标签：',labels)

    res_df = pd.DataFrame({
        'accessions': accessions,
        'labels': labels,
        'gos': gos,
        'sequences': sequences})
    print('蛋白质数量：',len(res_df))
    return res_df

def get_gene_ontology(filename='go.obo'):
    # Reading Gene Ontology from OBO Formatted file
    go = dict() # 创建一个空字典dict()
    obj = None
    with open('data/' + filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        go[obj['id']] = obj
    for go_id in list(go.keys()):
        if go[go_id]['is_obsolete']:
            del go[go_id]
    for go_id, val in go.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in go:
                if 'children' not in go[p_id]:
                    go[p_id]['children'] = set()
                go[p_id]['children'].add(go_id)
    #print(go)
    # 选取print结果中的其中一个如下： 'GO:0003779': {'is_a': ['GO:0008092'], 'part_of': [], 'regulates': [],
    # 'is_obsolete': False, 'id': 'GO:0003779', 'name': 'actin binding', 'children': {'GO:0003785', 'GO:0051015'}},
    # 查询go.obo文件可以发现，在'GO:0003779'下的'is_a'项显示着'GO:0008092'，这说明GO:0008092是GO:0003779的上层分类；
    # 再去到'GO:0003785'以及'GO:0051015'下，可以发现它们的'is_a'都是'GO:0003779'，这说明'GO:0003779'是'GO:0003785'和'GO:0051015'的上层分类
    # 所以在'GO:0003779'这里定义的'children'值为{'GO:0003785', 'GO:0051015'}
    # 注：'is_a'可以为空，说明该GO是根；'children'可以为空，说明该GO是叶子。
    return go

def get_anchestors(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        for parent_id in go[g_id]['is_a']: # 根据每个GO的is_a的信息，就可以得到所有GO之间的相互关系，利用程序处理，对于某个具体的GO，就可以得到其对应的所有上层分类；
            if parent_id in go:
                q.append(parent_id)
    return go_set

def get_go_set(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set

# Add functions to deque in topological order
# 以mf为例，从mf的id号'GO:0003674'着手(这个id号是mf类的根)，查找出在go.obo文件中'GO:0003674'的所有子类GO，即查找出所有属于mf的function注释(根据标签'children'递归查找)，并把结果存于functions队列
def dfs(go,go_id):
    if go_id not in func:
        for ch_id in go[go_id]['children']:
            dfs(go,ch_id) # 递归
        func.append(go_id)
        #print(functions) # 结果为形如deque(['GO:0034986', 'GO:0016532', 'GO:0016531', 'GO:0016530', 'GO:0036370',..., 'GO:0003674'])的很多个deque

def get_functions(go,subontology):
    df = pd.read_pickle(DATA_ROOT + 'swissprot.pkl')
    annots = dict()
    for i, row in df.iterrows():
        go_set = set()
        if not is_ok(row['sequences']):
            continue
        for go_id in row['annotations']:
            go_id = go_id.split('|')
            if go_id[1] not in EXP_CODES:
                continue
            go_id = go_id[0]
            if go_id in funcset: # 如果该go_id是属于mf的GO
                go_set |= get_anchestors(go, go_id) # '|='是把go_set和get_anchestors这两个集合合并，没有重复的元素，最终go_set存储了该GO对应的所有上层分类的GO(根据标签'is_a'查找)；
                                                    # 内部循环结束后，最终go_set存储了这一行数据里面所有GO的对应的所有上层分类的GO
                                                    # 每一行数据的go_set值是不一样的


        for go_id in go_set: # 遍历go_set里的每个go_id，go_set上会有不属于mf的GO
            if go_id not in annots:
                annots[go_id] = 0
            annots[go_id] += 1 # annots[go_id] = annots[go_id] + 1。这个循环相当于数数，该go_id在go_set出现了多少次，则其annots[go_id]为多少，说明其注释了多少个蛋白质
    filtered = list()
    if subontology=='bp':
        for go_id in func:  # 遍历functions即func_set上的go_id，func_set上存储了所有属于mf的GO
            if go_id in annots and annots[go_id] >= annot_num_bp:  # 筛选出属于mf的且注释了50个蛋白质以上的GO
                filtered.append(go_id)
    if subontology=='mf':
        for go_id in func:  # 遍历functions即func_set上的go_id，func_set上存储了所有属于mf的GO
            if go_id in annots and annots[go_id] >= annot_num_mf:  # 筛选出属于mf的且注释了50个蛋白质以上的GO
                filtered.append(go_id)
    if subontology=='cc':
        for go_id in func:  # 遍历functions即func_set上的go_id，func_set上存储了所有属于mf的GO
            if go_id in annots and annots[go_id] >= annot_num_cc:  # 筛选出属于mf的且注释了50个蛋白质以上的GO
                filtered.append(go_id)

    print('标签数量：',len(filtered))
    print('标签：', filtered)

    df = pd.DataFrame({'functions': filtered})  # 输出的文件为所筛选出来的function
    df.to_pickle(DATA_ROOT + subontology + '.pkl')
    #print('Saved ' + DATA_ROOT + subontology + '.pkl')


if __name__ == '__main__':
    main()
