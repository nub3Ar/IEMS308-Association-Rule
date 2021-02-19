#!/usr/bin/env python
# coding: utf-8


from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np



data_path = '/mnt/c/Course Data/Dillards_POS/'
file_name = ['deptinfo.csv', 'strinfo.csv', 'skuinfo.csv', 'trnsact.csv', 'skstinfo.csv']



data = pd.read_csv(data_path+'OKC.csv')


data['key'] = data.store.astype('str') + '_' +  data.trannum.astype('str') + '_' + data.saledate.astype('str')
data = data.set_index('key')
print("number of baskets", data.index.nunique())
print("total counts of transactions", len(data))

baskets = list(data.sku.groupby(data.index))
real_baskets = []
for item in baskets:
    real_baskets.append(list(item[1]))


te = TransactionEncoder()
te_ary = te.fit(real_baskets).transform(real_baskets)
baskets_df = pd.DataFrame(te_ary, columns = te.columns_)
shape = len(baskets_df), len(baskets_df.columns)
print(shape)

#frequent_items = apriori(baskets_df, min_support=0.00119, use_colnames=True)

#frequent_items.to_csv('freq_items.csv')

#rules = association_rules(frequent_items, metric="lift", min_threshold=0)

#rules.to_csv('rules.csv')

