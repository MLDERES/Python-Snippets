import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def lmse()
def RemoveConstantColumns(ds):
    colsToRemove = []
    for col in list(ds.select_dtypes(include=['float','int']).columns):
        if ds[col].std() == 0:
            colsToRemove.append(col)
    ds.drop(colsToRemove, axis=1, inplace=True)
    return colsToRemove

def check_sparsity(df):
    non_zeros = (df.ne(0).sum(axis=1)).sum()
    total = df.shape[1]*df.shape[0]
    zeros = total - non_zeros
    sparsity = round(zeros / total * 100,2)
    density = round(non_zeros / total * 100,2)

    print(" Total:",total,"\n Zeros:", zeros, "\n Sparsity [%]: ", sparsity, "\n Density [%]: ", density)
    return density

# Plot distribution of one feature
def plot_distribution(df,feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(df[feature].dropna(),color=color, kde=True,bins=100)
    plt.show()   

# Plot log distribution of one feature
def plot_log_distribution(df,feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(np.log1p(df[feature]).dropna(),color=color, kde=True,bins=100)
    plt.title("Distribution of log(target)")
    plt.show()  

# Calculate most highly correlated features to the target
def find_corr_features(ds, label='target', min_abs_correlation=0.25):
    labels = []
    values = []
    for col in ds.columns:
        if col != label :
            labels.append(col)
            c = np.corrcoef(ds[col].values, ds[label].values)
            values.append(c[0,1])
    corr_df = pd.DataFrame({'columns_labels':labels, 'corr_values':values})
    corr_df = corr_df.sort_values(by='corr_values')
    
    corr_df = corr_df[(corr_df['corr_values']>min_abs_correlation) | (corr_df['corr_values']<(min_abs_correlation * -1))]
    return corr_df


ds = pd.DataFrame({'one': [1, 2, 3, 4, 5], 'two':[0,0,0,0,0], 'three':[1,2,3,4,6]})
print(ds.columns)
print(ds)
cols = RemoveConstantColumns(ds)
print (cols)

print(check_sparsity(ds))

ds_c = find_corr_features(ds,label='one')
print(ds_c)
