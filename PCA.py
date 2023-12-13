main_folder = r"\\solon.prd\files\P\Global\Users\C63954\UserData\Desktop\Work_fromARG_Covid\RL"  
import os
os.chdir(main_folder) 
from PCA_Func import *
%pprint
# Fundamental Data
pcts={'ST':100,'LT':200,'SLT':300}
pct=pcts['ST']
selected_ret=['MXEF Index', 'BCOMPR Index', 'BCOMSO Index', 'BCOMIN Index', 'BCOMGR Index','RLG Index', 'RLV Index', 'LG30TRUU Index','SPX Index', 'DXY Curncy','LUATTRUU Index','BCOMEN Index'] # 'CCMP Index','BCOM Index','SML Index','EMUSTRUU Index'
selected_ret.extend(['USGG2YR Index','GSUSFCI Index', 'Vix Index', 'MOVE Index', 'SPX Index_PE_RATIO', 'SPX Index_RETURN_ON_ASSET', 'SPX Index_TOT_DEBT_TO_TOT_ASSET', 'f_BCOM Index']) # 'SPX Index_PS_RATIO','SFFRNEWS Index','SPX Index_RETURN_COM_EQY'
years_opt = ['1991-04-15','1984-01-13']
file_not_lstm, year ='\\PCs_diff_timeframes.xlsx' , years_opt[0]
funda = read_excel("Anom_Trend_CPD.xlsx","PCA_not_Prices",h=4).set_index('Dates').sort_index().drop_duplicates()#.drop(net_posit,axis=1)
funda=funda[funda.index > dateutil.parser.parse(year)][[c for c in colu(funda) if c in selected_ret]]
funda.drop(null_col_f(funda,0),axis=1,inplace=True)
ret=read_excel("Anom_Trend_CPD.xlsx","Prices",h=1).set_index('Dates').sort_index().drop_duplicates()
ret=ret[ret.index > dateutil.parser.parse(year)][[c for c in colu(ret) if c in selected_ret]]
ret.drop(null_col_f(ret,0),axis=1,inplace=True)
ret=ret.pct_change(periods=pct)
dfs_for_pca = {'r':ret,'f':funda}
df=dfs_for_pca['r'].copy()
for c in range(len(old_names)):
    df.rename(columns={old_names[c]:new_names[c]},inplace=True)
df=df.rolling(funda_av).mean().dropna()
pca_col,idxs = colu(df),df.index
df_pca = StandardScaler().fit_transform(df)
print('Corr among Var in Df:\n',round(df.corr(),1))
# PCA
comp_col = add_suffix('PC_',np.arange(comp_n)+1)
pca = PCA(comp_n).fit(df_pca)
plot_expl_var(pca,comp_n)
load = print_loadings(pca,comp_col,pca_col,0.15).sort_index()
letters_displayed=7
biplot('PC_1','PC_2',load,sz_bip,descr_combined['PC_1'][1]+" & "+descr_combined['PC_2'][1],letters_displayed)
biplot('PC_3','PC_4',load,sz_bip,descr_combined['PC_3'][1]+" & "+descr_combined['PC_4'][1],letters_displayed)
biplot('PC_5','PC_6',load,sz_bip,descr_combined['PC_5'][1]+" & "+descr_combined['PC_6'][1],letters_displayed)
biplot('PC_1','PC_3',load,sz_bip,descr_combined['PC_1'][1]+" & "+descr_combined['PC_3'][1],letters_displayed)
biplot('PC_1','PC_5',load,sz_bip,descr_combined['PC_1'][1]+" & "+descr_combined['PC_5'][1],letters_displayed)
loadings_hm(load,1,'Var')
pca_df=pd.DataFrame(data=pca.transform(df_pca),columns=comp_col,index=idxs)
print('Corr among PCs in PCA_Df:\n',round(pca_df.corr(),1))
df_corr = corr(pd.concat([df,pca_df], axis=1),load.index.tolist(),comp_col)
plot_hist(pca_df[comp_col[0:3]],10,3,30,hist_titles) # descr_ret
plot_hist(pca_df[comp_col[3:6]],10,3,30,hist_titles) # descr_ret
plot_hist(pca_df[comp_col[6:]],10,3,30,hist_titles) # descr_ret
plot_pca_entire(pca_df,['PC_1','PC_2','PC_3'],['PC_1','PC_2','PC_3'],av=funda_av)
plot_pca_entire(pca_df,['PC_1','PC_2'],['PC_1','PC_2'],av=funda_av)
plot_pca_entire(pca_df,['PC_3','PC_4'],['PC_3','PC_4'],av=funda_av)
plot_pca_entire(pca_df,['PC_5','PC_6'],['PC_5','PC_6'],av=funda_av)
#plot_pca_entire(pca_df,['PC_7'],['PC_7'],av=funda_av)
plot_pca_entire(pca_df.iloc[:int(len(pca_df)/4),:],colu(pca_df),colu(pca_df),av=funda_av,sz=8) # ['Risk-off','FI','Comm vs USD','Comm vs FI','SML vs all','SML vs Oil']
plot_pca_entire(pca_df.iloc[int(len(pca_df)/4):int(len(pca_df)/2),:],colu(pca_df),colu(pca_df),av=funda_av,sz=8) # ['Risk-off','FI','Comm vs USD','Comm vs FI','SML vs all','SML vs Oil']
plot_pca_entire(pca_df.iloc[int(len(pca_df)/2):int(len(pca_df)*3/4),:],colu(pca_df),colu(pca_df),av=funda_av,sz=8) # ['Risk-off','FI','Comm vs USD','Comm vs FI','SML vs all','SML vs Oil']
plot_pca_entire(pca_df.iloc[int(len(pca_df)*3/4):,:],colu(pca_df),colu(pca_df),av=funda_av,sz=8) # ['Risk-off','FI','Comm vs USD','Comm vs FI','SML vs all','SML vs Oil']
corr_w_lags(pca_df,[50,100,200,400,600])
# Joint PCA: Ret & Funda
pc_r_SLT=pca_df.copy() # pc_r_ST,pc_r_LT,pc_r_SLT,pc_fun
pc_r_SLT.columns=[c+'_R300' for c in colu(pc_r_SLT)]
pc_r_LT.columns=[c+'_R200' for c in colu(pc_r_LT)]
pc_r_ST.columns=[c+'_R100' for c in colu(pc_r_ST)]
pc_fun.columns=[c+'_F' for c in colu(pc_fun)]
ret_funda=pd.concat([pc_r_SLT,pc_r_LT,pc_r_ST,pc_fun],axis=1).dropna()
#r,f=1,1
#plot_pca_entire(ret_funda,['PC_'+str(r)+'_R300','PC_'+str(r)+'_R200','PC_'+str(r)+'_R100','PC_'+str(f)+'_F'],['300R_'+str(r),'200R_'+str(r),'100R_'+str(r),'F_'+str(f)],av=funda_av)
ret_funda = ret_funda[[c for c in colu(ret_funda) if 'PC_1_' in c]]
ret_funda.to_excel(main_folder+file_not_lstm,index_label='Dates')




