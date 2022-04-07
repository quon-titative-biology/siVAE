import os

import numpy as np
import pandas as pd

## Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

datadir = "vcf/MTvcf"

vcffiles = os.listdir(datadir)
vcffiles = [f for f in vcffiles if '.vcf' in f]
tasks = [f.split(".vcf")[0] for f in vcffiles]

df_eff = pd.read_csv('diff_efficiency_neur.csv')
df_eff.index = df_eff.donor_id

columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'NA00001', 'NA00002', 'NA00003']

vcf_list = []
for task in tasks:
    df_vcf = pd.read_csv(os.path.join(datadir,f"{task}.vcf"),
                         delimiter = '\t',
                         header=None)
    df_vcf.columns = columns[:10]
    df_vcf['cellline'] = task
    eff = df_eff[df_eff.donor_id==task].diff_efficiency.to_numpy()[0]
    df_vcf['diff_efficiency'] = eff
    vcf_list.append(df_vcf)

df_ = pd.concat(vcf_list)

def to_adj(x,y,z=None):
    if isinstance(x,pd.Series):
        x = x.values
    if isinstance(y,pd.Series):
        y = y.values
    columns=np.unique(y)
    index=np.unique(x)
    df = pd.DataFrame(np.zeros([len(index),len(columns)]),
                      columns=columns,
                      index=index)
    zipped = zip(x,y) if z is None else zip(x,y,z)
    for x1,y1 in zipped:
        df.loc[x1,y1] = df.loc[x1,y1] + 1
    return df

df_heatmap = to_adj(df_.cellline,df_.POS)

df_heatmap = df_heatmap.iloc[:,np.where(df_heatmap.sum(axis=0) > 5)[0]]

effs = df_eff.reindex(df_heatmap.index).diff_efficiency
df_heatmap = df_heatmap.reindex(effs.sort_values().index)

series = df_eff.reindex(df_heatmap.index).diff_efficiency
series = pd.cut(series,5).sort_values()
series = series.astype('str')
lut = dict(zip(series.unique(),
               sns.color_palette("vlag")))
row_colors = series.map(lut)

g = sns.clustermap(df_heatmap,row_colors=row_colors, row_cluster=False,
                   cmap='Blues',linewidths=0.0,col_colors=col_colors)
plt.savefig('Heatmap-variant_vs_cellline_sort_by_eff.svg')
plt.close()


## wilcoxon test
from scipy.stats import ranksums
from scipy.stats import linregress

ranksum_test = []
linearm_test = []
for variant,idx in df_heatmap.iteritems():
    x = effs[np.where(idx == 1)[0]]
    y = effs[np.where(idx == 0)[0]]
    ranksum_test.append(ranksums(x,y))
    linearm_test.append(linregress(effs,idx))



df_ranksum = pd.DataFrame(stat_test, index=df_heatmap.columns)
df_ranksum = df_ranksum.fillna(1)
df_linearm = pd.DataFrame(linearm_test,index=df_heatmap.columns)

col_colors = []
for df_stat in [df_linearm,df_ranksum]:
    series = df_stat.pvalue
    series = pd.cut(series,5).sort_values()
    series = series.astype('str')
    lut = dict(zip(series.unique(),
                   sns.color_palette("vlag")))
    colors = series.map(lut)
    col_colors.append(colors)

# col_colors = pd.DataFrame(col_colors, columns=['LinearModel','RankSum'])
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr,pearsonr


def cross_validation(x,y,method='Lasso',n=5,**kwargs):
    skf = StratifiedKFold(n_splits=n)
    skf.get_n_splits(x)
    losses = []
    if method == 'Lasso':
        model = Lasso(**kwargs)
    elif method == 'Ridge':
        model = Ridge(**kwargs)
    elif method == 'Linear':
        model = LinearRegression(**kwargs)
    else:
        raise Exception('Input valid method [Linear,Ridge,Lasso]')
    for idx_train,idx_test in skf.split(x,[True]*len(x)):
        x_train, x_test = x[idx_train], x[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        linregress = model.fit(x_train,y_train)
        loss_train = np.square(y_train - linregress.predict(x_train)).mean()
        loss_test = np.square(y_test - linregress.predict(x_test)).mean()
        losses.append([loss_train,loss_test])
    return pd.DataFrame(losses,columns=['Train','Test'])


mt_dictionary = {'MT-ATP8': [8366,8572],
                 'MT-ATP6': [8527,9207],
                 'MT-CO1':  [5904,7445],
                 'MT-CO2':  [7586,8269],
                 'MT-CO3':  [9207,9990],
                 'MT-CYB':  [14747,15887],
                 'MT-ND1':  [3307,4262],
                 'MT-ND2':  [4470,5511],
                 'MT-ND3':  [10059,10404],
                 'MT-ND4L': [10470,10766],
                 'MT-ND4':  [10760,12137],
                 'MT-ND5':  [12337,14148],
                 'MT-ND6':  [14149,14673]}
mt_dictionary['all'] = [df_heatmap.columns.min(),
                        df_heatmap.columns.max()]

prs = []
burden = []
for gene,(start,end) in mt_dictionary.items():
    print(gene)
    idx_start = np.where(start <= df_heatmap.columns)[0][0]
    idx_end   = np.where(end >= df_heatmap.columns)[0][-1]+1
    df_gene = df_heatmap.iloc[:,idx_start:idx_end]
    if df_gene.shape[1] > 0:
        # PRS
        x = df_gene.values
        y = effs
        df_sum = df_gene.values
        for method in ('Lasso','Ridge'):
            df_cv = cross_validation(x,y,method=method)
            df_cv['gene'] = gene
            df_cv['Method'] = method
            prs.append(df_cv)
        # Burden test
        x = df_gene.sum(1).values
        burden.append([gene,pearsonr(x,y)])

baseline = np.square(y-y.mean()).mean()
df_prs = pd.concat(prs)

for metric in ('Test','Train'):
    g = sns.barplot(data=df_prs,x='gene',y=metric,hue='Method')
    g.plot(g.get_xlim(),[baseline,baseline],'red') # add a line
    g.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(f'PRS-barplot-{metric}.svg')
    plt.close()

pd.DataFrame(burden)
