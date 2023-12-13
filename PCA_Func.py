# Libraries
from Libraries import *
# Variables
funda_av,comp_n,sz_bip,t,horizon = 100,4,7,'PC_3',{'month':20,'week':5}
ts,units,opt,loss,loss_pat,mon_loss,test_cut = int(20*12),600,'Adam',"mean_absolute_error",20,'loss',0.9
es=EarlyStopping(monitor=mon_loss, verbose=1,patience=loss_pat)
old_names = ['JNRETMOM Index','RSTAXAG% Index','RSSAEMUM Index','HSESEZAI Index',	 'HSESUSAI Index',	 'HSESJNAI Index',	 'HSESCHAI Index',	 'USGGBE05 Index',	 'USGG10YR Index',	 'USGG2YR Index',	 'GBTPGR10 Index',	 'GBTPGR2 Index',	 'GTDEM10Y Govt',	 'GTDEM2Y Govt',	 'JCBDAGTW Index',	 'JPBYAGSW Index',	 'LECPOAS Index',	 'BFCIUS Index',	 'GSUSFCI Index',	 'RFSIEMER Index',	 'RFSIOTHR Index',	 'RFSIUS Index',	 'SLFXFSI3 Index',	 'OPWVCBOR Index',	 'JPMVXYEM Index',	 'MXEF0CX0 Index',	 'SPX Index_PE_RATIO',	 'SPX Index_PX_TO_BOOK_RATIO',	 'SPX Index_PX_TO_TANG_BV_PER_SH',	 'SPX Index_PX_TO_CASH_FLOW',	 'SPX Index_NET_AGGTE_DVD_YLD',	 'SPX Index_RETURN_COM_EQY',	 'SPX Index_RETURN_ON_ASSET',	 'SPX Index_TOT_DEBT_TO_TOT_ASSET',	 'SPX Index_PROF_MARGIN',	 'CCMP Index_PE_RATIO',	 'CCMP Index_PX_TO_BOOK_RATIO',	 'CCMP Index_PX_TO_TANG_BV_PER_SH',	 'CCMP Index_PX_TO_CASH_FLOW',	 'CCMP Index_NET_AGGTE_DVD_YLD',	 'CCMP Index_RETURN_COM_EQY',	 'CCMP Index_RETURN_ON_ASSET',	 'CCMP Index_TOT_DEBT_TO_TOT_ASSET',	 'CCMP Index_PROF_MARGIN',	 'NKY Index_NET_AGGTE_DVD_YLD',	 'HSI Index_NET_AGGTE_DVD_YLD',	 'UKX Index_PE_RATIO',	 'UKX Index_PX_TO_BOOK_RATIO',	 'UKX Index_PX_TO_TANG_BV_PER_SH',	 'UKX Index_PX_TO_CASH_FLOW',	 'UKX Index_NET_AGGTE_DVD_YLD',	 'UKX Index_RETURN_COM_EQY',	 'UKX Index_RETURN_ON_ASSET',	 'UKX Index_TOT_DEBT_TO_TOT_ASSET',	 'UKX Index_PROF_MARGIN',	 'SXXP Index_PE_RATIO',	 'SXXP Index_PX_TO_BOOK_RATIO',	 'SXXP Index_PX_TO_TANG_BV_PER_SH',	 'SXXP Index_PX_TO_CASH_FLOW',	 'SXXP Index_NET_AGGTE_DVD_YLD',	 'SXXP Index_RETURN_COM_EQY',	 'SXXP Index_RETURN_ON_ASSET',	 'SXXP Index_TOT_DEBT_TO_TOT_ASSET',	 'SXXP Index_PROF_MARGIN',	 'MXEF Index_PE_RATIO',	 'MXEF Index_PX_TO_BOOK_RATIO',	 'MXEF Index_PX_TO_CASH_FLOW',	 'MXEF Index_NET_AGGTE_DVD_YLD',	 'MXEF Index_RETURN_COM_EQY',	 'MXEF Index_RETURN_ON_ASSET',	 'MXEF Index_TOT_DEBT_TO_TOT_ASSET',	 'MXEF Index_PROF_MARGIN','SPX Index_PS_RATIO','BFCIUS+ Index']
new_names=['JP_Ret_Sal','US_Ret_Sal','EU_Ret_Sal','Ec_Surp_EU','Ec_Surp_US','Ec_Surp_JP','Ec_Surp_CN','US_BE_5y','US_10y','US_2y','IT_10y','IT_2y','GE_10y','GE_2y','Cembi_Spr','Embi_Spr','EU_IG_Spr','Bbg_US_FCI','GS_US_FCI','Fin_Str_EM','Fin_Str_Oth','Fin_Str_US','Fin_Str_FED','Put_Opt_Volu','Vix_EM','EM_Ccy','SPX_P_E','SPX_P_B','SPX_P_T','SPX_P_CF','SPX_Div','SPX_ROE','SPX_ROA','SPX_D_A','SPX_Pro_Mg','QQQ_P_E','QQQ_P_B','QQQ_P_T','QQQ_P_CF','QQQ_Div','QQQ_ROE','QQQ_ROA','QQQ_D_A','QQQ_Pro_Mg','NKY_Div','HSI_Div','UK_P_E','UK_P_B','UK_P_T','UK_P_CF','UK_Div','UK_ROE','UK_ROA','UK_D_A','UK_Pro_Mg','SXXP_P_E','SXXP_P_B','SXXP_P_T','SXXP_P_CF','SXXP_Div','SXXP_ROE','SXXP_ROA','SXXP_D_A','SXXP_Pro_Mg','MXEF_P_E','MXEF_P_B','MXEF_P_CF','MXEF_Div','MXEF_ROE','MXEF_ROA','MXEF_D_A','MXEF_Pro_Mg','SPX_P_S','Bbg_US+_FCI']
ret_gr_bar,net_posit,eq_excl = ['BCOMEN_60_%' , 'SPX_60_%' , 'LUATTRUU_60_%'],['IMM2SNCN Index','CBT4TNCN Index','CBT42NCN Index','CBT55NCN Index','NYM1CNCN Index'],['RNIRISG Index','RNIRISV Index']
cyclical,defensive=['CON_D','MAT'],['ENE','CONS_S','HLT','UTI']
hist_titles={"PC_1":["","[PC_1]"],"PC_2":["","[PC_2]"],"PC_3":["","[PC_3]"],"PC_4":["","[PC_4]"],"PC_5":["","[PC_5]"],"PC_6":["","[PC_6]"],"PC_7":["","[PC_7]"]}

descr_combined={"PC_1":["","[High_Vol,Treas,DXY]"], 
       "PC_2":["","[GLD vs DXY]"], 
       "PC_3":["","[High_Yields_Leverage_MOVE_Oil_Valuat]"], 
       "PC_4":["","[High_Commo_Valuat vs FI]"],
       "PC_5":["","[High_INFT_Grow vs Ind, Neg_News]"],
       "PC_6":["","[High_Treas_GLD_Prof vs DXY&SML]"],
       "PC_7":["","[Gr&Soft vs Metals, High_MOVE, Low_Valuat]"]}

# Functions
def create_ts(m,sd,sz):
    return np.random.normal(loc=m, scale=sd, size=sz)

def quantiles(ts,q):
    return np.round(np.percentile(ts, np.arange(0, 100, q)),1)

def mult_diff(d,rang,av=30):
    col = d.columns.tolist()
    d = d.ewm(span=av,min_periods=av).mean().dropna()
    for r in rang:
        new_col = [c+'_r'+str(r) for c in col]
        d[new_col] = d[col].diff(r)
    return d

def plot_all_2axis(d,t,tit,bar,l):
    fig, ax = plt.subplots()
    ax.plot(d[t],label=l,linewidth=1.0)
    ax.plot(d[t+'_Pred'],label='Pred',linewidth=1.0)
    ax.axhline(y=0,color='black',linestyle='--',linewidth=2.0)
    ax.legend(loc='upper left')
    ax.tick_params(axis="x",rotation=45,labelsize='x-small')
    ax2=ax.twinx()
    bar = bar.loc[d.index]
    ax2.bar(bar.index, bar, edgecolor='maroon',alpha=0.1,label=bar.name)
    ax2.legend(loc='upper right')
    ax2.set_title(tit)
    
def plot_all(d,t,tit):
    plt.plot(d[t],label=t,linewidth=1.0)
    plt.plot(d[t+'_Pred'],label='Pred',linewidth=1.0)
    plt.axhline(y=0,color='black',linestyle='--',linewidth=2.0)
    plt.legend(loc='upper left')
    plt.tick_params(axis="x",rotation=45,labelsize='x-small')
    plt.title(tit)
    
def plot_pred(f,i_f,ind_tr,ind_ts,cut,y_tr,y_ts,pred,t,mape,fs,tit):
    plt.plot(ind_tr[cut:], y_tr[cut:], linestyle='solid',linewidth=1.0, c='b',label='end_training')
    plt.plot(ind_ts, y_ts, linestyle='dotted', linewidth=2.0, c='r',label='actual')
    plt.plot(ind_ts, pred, linestyle='dotted' , linewidth=2.0, c='green',label='pred')
    plt.plot(i_f,f, linestyle='solid' , linewidth=2.0, c='yellow',label='Current_fcst')
    plt.axhline(y=0,color='black',linestyle='--')
    plt.xlabel('Time',fontsize=fs)
    plt.ylabel(t,fontsize=fs)
    plt.legend()
    plt.title(tit+': ('+t+'_MAPE: {})'.format(mape),fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs)
    plt.tight_layout()

def plot_volat(d,tit,roll=60):
    d_vol=d.rolling(roll).std().dropna()
    d_vol.plot(title=tit)
    
def prepare_LSTM(d,t,day,ts,test_cut):
    d['Target']=d[t].shift(-day)
    d = d.dropna()
    y = d.pop('Target')
    y = y[ts-1:]
    indi,x=d.index[(ts-1):],d.values
    i,x_arr = 0,[]
    while i < len(x)-ts+1:
        x_arr.append(x[i:i+ts,:])
        i += 1
    x_arr=np.array(x_arr)
    cut=int(test_cut*len(x_arr))
    x_tr,x_ts,y_tr,y_ts = x_arr[:cut],x_arr[cut:],y[:cut],y[cut:]
    print('\nTarget: ',t,'\nTs: ',ts,'\nHorizon: ',day)
    return y_tr,y_ts,x_tr,x_ts,indi

    
def prepare_FCST(d,day,ts):
    f = d.iloc[-day-(ts-1):,:]
    indi,x=f.index[-day:],f.values
    i,x_arr = 0,[]
    while i < len(x)-ts+1:
        x_arr.append(x[i:i+ts,:])
        i += 1
    x_arr=np.array(x_arr)
    return x_arr,indi

def run(x_tr,y_tr,epo=1000,drop=0):
    model = Sequential()
    model.add(LSTM(units = int(units),dropout=drop, return_sequences = True, input_shape = (x_tr.shape[1], x_tr.shape[2])))
    model.add(LSTM(units = int(units/2),dropout=drop, return_sequences = True))
    model.add(LSTM(units = int(units/4),dropout=drop, return_sequences = True))
    model.add(LSTM(units = int(units/8),dropout=drop))
    model.add(Dense(units = 1,activation='linear'))
    model.compile(optimizer = opt, loss = loss)
    model.fit(x_tr,y_tr,epochs=epo,callbacks=[es])
    return model

def loadings_of_returns(pc,fu,com,load,n):
    new=load.loc[[c for c in pc if c not in fu],com]
    for p in com:
        new_sorted = new[p].sort_values(ascending=False)
        print('\n'+p+'\nHead')
        print(new_sorted.head(n))
        print(p+'\nTail')
        print(new_sorted.tail(n))
    
def print_loadings(pca,comp_col,col,n):
    return round(pd.DataFrame(pca.components_.T,index=col,columns=comp_col),1)
    

def index_vs_PC(ind,comp_col,df,df_corr,ma_comp=50,col_plot=2,sz=20,lw=2,fs=30,n=comp_n):
    df_spx = pd.concat([df[ind],df_corr[comp_col]], axis=1).dropna()#.reset_index()
    df_spx = df_spx.ewm(span=ma_comp,min_periods=ma_comp).mean().dropna().reset_index()
    fig, ax = plt.subplots(n//3+1, col_plot, figsize=(sz,sz))
    ax = ax.ravel()
    ii=0
    for c in range(1,n+1):
        ax[ii].plot(df_spx.index, df_spx[ind],linestyle='solid',linewidth=lw,label=ind[:-6], color='black')
        ax[ii].set_ylabel(ind.replace(' Index',''))
        ax[ii].legend(loc='upper left',fontsize=fs)
        ax2=ax[ii].twinx()
        ax2.plot(df_spx.index, df_spx['PC_'+str(c)],linestyle='solid',linewidth=lw,label=str(c), color='red')
        ax2.axhline(y = 0, color = 'g', linewidth=4,linestyle='dotted')
        ax2.set_ylabel('PC')
        ax2.set_title('vs_PC_'+str(c),fontsize=fs)
        ax2.legend(loc='upper right',fontsize=fs)
        plt.tight_layout()
        ii+=1
    

def plot_in_pc_dim(first_comp,pca_df):
    x_lab,y_lab='PC_'+str(first_comp),'PC_'+str(first_comp+1)
    #m, b = np.round(np.polyfit(pca_df[x_lab],pca_df[y_lab], 1),2)
    plt.scatter(pca_df[x_lab],pca_df[y_lab])
    #plt.plot(pca_df[x_lab], m*pca_df[x_lab]+b,color='y')
    plt.axhline(y = 0, color = 'r', linestyle='dotted')
    plt.axvline(x = 0, color = 'r', linestyle='dotted')
    plt.title('DF in PC_Comp')
    #plt.title(str(m)+'_'+str(b))
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.show()


def plot_hist(d,f,f_c,b,di):
    fig, ax = plt.subplots(1, f_c, figsize=(f,f))
    ax = ax.ravel()
    for i in range(d.shape[1]):
        col= d.iloc[:,i]
        ax[i].hist(col,bins=b)
        ax[i].axvline(x=0,color='r',linestyle='dotted',linewidth=2)
        ax[i].set_title(di[col.name][1])
    
def plot_pca_broken(d,f,f_c,cuts,leg):
    d=d.ewm(span=funda_av,min_periods=funda_av).mean()
    fig, ax = plt.subplots(cuts//3, f_c, figsize=(f,f))
    ax = ax.ravel()
    sz = int(d.shape[0]/cuts)
    for i in range(cuts):
        ax[i].plot(d.iloc[i*sz:(i+1)*sz,:])
        ax[i].axhline(y = 0, color = 'r', linestyle='dotted',linewidth=2)
        ax[i].legend(leg)
        ax[i].set_title('Chunk '+str(i+1))
        ax[i].tick_params(axis="x",rotation=45,labelsize='x-small')
        
def plot_pca_entire(d,pcs,leg,av=funda_av,sz=10):
    ax=d[pcs].ewm(span=av,min_periods=av).mean().dropna().plot()
    ax.axhline(y = 0, color = 'black', linestyle='dotted',linewidth=2)
    ax.legend(leg,fontsize=sz)
   

def loadings_hm(load,f,t):
    plt.figure(figsize = (12,8))
    sns.heatmap(load, cmap="YlGnBu", annot=True)
    plt.title(t+'-PC Loadings') 

def corr(d,l,comp,rou=1):
    corr = round(d.corr().loc[l,comp],rou) 
    plt.figure(figsize = (12,8))
    sns.heatmap(corr, cmap="YlGnBu", annot=True)
    plt.title('Var-PC Corr')
    return d

def corr_w_lags(d,lags):
    orig_cols,new=colu(d),d.copy()
    for l in lags:
        new[[c+'_L'+str(l) for c in orig_cols]]=new[orig_cols].shift(l)
    corr(new.dropna(),orig_cols,[c for c in colu(new) if 'L' in c],rou=1)
    
def biplot(uno,dos,load,fz,ti,char):
    plt.scatter(load[uno],load[dos],s=20,c='orange')
    plt.axhline(y = 0, color = 'r', linestyle='dotted')
    plt.axvline(x = 0, color = 'r', linestyle='dotted')
    for i in range(len(load)):
        plt.annotate(load.index[i][:char], (load[uno][i], load[dos][i]),fontsize=fz)
    plt.title(ti)
    plt.xlabel(uno)
    plt.ylabel(dos)
    plt.show()

def list_comp(col,word):
    return [c for c in col if word in c]

def colu(d):
    return d.columns.tolist()

def plot_expl_var(pca,many):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.plot(pca.explained_variance_ratio_, 'o-', linewidth=2, color='red')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    print('\n{} first PC Eigenvalues:\n{}'.format(many,np.round(pca.explained_variance_[:many],2)))
    expl_var = np.round(pca.explained_variance_ratio_[:many],2)
    print('\n{} first PC Explained Var: {}\n{}'.format(many,np.round(expl_var.sum(),2),expl_var))

def calc_pct_chang(pca_col,df,ranges,news_av=20,roll=20,ind=2):
    pct_col=[]
    for ran in ranges:#[ranges[ind]]
        colu=[col.replace(' Index','_'+str(ran)).replace(' Curncy','_'+str(ran)) for col in pca_col]
        col_p,col_v=[c+'_%' for c in colu],[c+'_Volat' for c in colu]
        df[col_p] = df[pca_col].pct_change(periods=ran)
        df[col_v] = df[col_p].rolling(roll).std()
        pct_col.extend(col_p+col_v)
    #df['SFFRNEWS Index'] = df['SFFRNEWS Index'].ewm(span=news_av,min_periods=news_av).mean()
    #pct_col.extend(['SFFRNEWS Index'])
    return df,pct_col


def toy_df(mn,mx,l,w,col):
    df = pd.DataFrame(np.random.randint(mn,mx,size=(l,w)), columns=list(col))
    return df

def toy_df_fl(mn,mx,l,w,col):
    df = pd.DataFrame(np.random.uniform(mn,mx,size=(l,w)), columns=list(col))
    return df

def read_pickle(nam):
    df = pd.read_pickle(nam)
    return df
 
def save_pickle(df,nam):
    pickle_out = open(nam,"wb")
    pickle.dump(df, pickle_out)
    pickle_out.close()
    print('Saved: {}'.format(nam))


def autocorrelation(d,ranges):
    cols=colu(d)
    for r in ranges:
        d[[c+'_'+str(r) for c in cols]]=d[cols].shift(-r)
    return d

def null_col_f(d,th):
    null_col = round(d.isnull().sum()/len(d),2)
    null_col2 = null_col[null_col>th].sort_values()
    null_col_nam = null_col2.index
    print('\nCol Below_Threshold:')
    print(null_col[(null_col<=th)*(null_col>0)].sort_values())
    print('\nCol Above_Threshold:')
    print(null_col2)
    return null_col_nam

def funda_pca_f(f,av):
    f = f.ewm(span=av,min_periods=av).mean().dropna()#.diff(av).dropna()
    idxs = f.index 
    f = StandardScaler().fit_transform(f)
    return f, idxs

def read_excel(f,s,h=0,path=None):
    if path==None:
        df = pd.read_excel(f,s,engine="openpyxl",header=h)
    else:
        df = pd.read_excel(path+f,s,engine="openpyxl",header=h)
    return df


def trend(df,lamb,tit,f,f_c,trends):
    solver = cvxpy.ECOS
    reg_norm = 1
    col_n=df.shape[1]
    fig, ax = plt.subplots(col_n//3+1, f_c, figsize=(f,f))
    ax = ax.ravel()
    ii=0
    for c in range(col_n):
        df_col=df.iloc[:,c]
        n = df.shape[0]
        ones_row = np.ones((1, n))
        D = scipy.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)
        colu = np.log(df_col).to_numpy()
        x = cvxpy.Variable(shape=n) 
        obj = cvxpy.Minimize(0.5 * cvxpy.sum_squares(colu-x) + lamb * cvxpy.norm(D@x, reg_norm))
        problem = cvxpy.Problem(obj)
        problem.solve(solver=solver, verbose=False)
        ax[ii].plot(np.arange(1, n+1), colu, linestyle='dotted',linewidth=1.0, c='b')
        ax[ii].plot(np.arange(1, n+1), np.array(x.value), 'b-', linewidth=2.0, c='r')
        ax[ii].set_xlabel('Time',fontsize=16)
        ax[ii].set_ylabel('Log (px)',fontsize=16)
        ax[ii].set_title(tit+': {}'.format(df_col.name),fontsize=20)
        ax[ii].tick_params(axis='both', labelsize=16)
        trends[df_col.name]=np.array(x.value)
        ii+=1
    return trends

def null_rows(d):
    return d[d.isnull().any(axis=1)]

def add_suffix(tx,lis):
    return [tx+str(c) for c in lis]

def anom_f(df,f,f_c,algor,lw,sz,anom,anom_algo):
    col_n=df.shape[1]
    colus=df.columns.tolist()
    fig, ax = plt.subplots(col_n//3+1, f_c, figsize=(f,f))
    ax = ax.ravel()
    ii=0
    for c in range(col_n):
        colu = colus[c]
        x=df.loc[:,colu].values.reshape(-1, 1)
        logui=np.log(x)
        algo=anom_algo[algor]
        #if_norm = algo.fit_predict(x)
        if_log = algo.fit_predict(logui)
        #df_norm = pd.DataFrame(data={colu:x.reshape(len(x)),'Score':if_norm})
        #df_an = df_norm[df_norm['Score']==-1]
        df_log = pd.DataFrame(data={colu:logui.reshape(len(logui)),'Score':if_log})
        df_al = df_log[df_log['Score']==-1]
        # plot
        """
        ax[ii].plot(df_norm.index, df_norm[colu],linestyle='dotted',linewidth=lw, color='black')
        ax[ii].scatter(df_an.index, df_an[colu], c ="yellow",s=sz,label="Original")
        ax[ii].set_ylabel('Original_Scale')
        ax[ii].legend(loc='upper left')
        ax2=ax[ii].twinx()
        """
        ax[ii].plot(df_log.index, df_log[colu],linestyle='dotted',linewidth=lw, color='black')
        ax[ii].scatter(df_al.index, df_al[colu], c ="g",s=sz,label="Log")
        ax[ii].set_ylabel('Log_Scale')
        ax[ii].set_title(algor+'_'+colu)
        ax[ii].legend(loc='upper center')
        plt.tight_layout()
        ii+=1
        anom[colu]=if_log
    return anom


def anom_group(df,col,algor,lw,sz,anom,f,fs,l_algo):
    colu = df.columns.tolist()
    logui=np.log(df)
    algo=l_algo[algor]
    df['Score'] = algo.fit_predict(logui)
    df_al = df[df['Score']==-1]
    # plot
    fig, ax = plt.subplots(figsize=(f,f))
    for c in [r for r in colu if col not in r]:
         ax.plot(df.index, df[c],linestyle='solid',linewidth=lw, color='black')
    ax2=ax.twinx()
    ax2.plot(df.index, df[col],linestyle='solid',linewidth=lw, color='red')
    ax2.scatter(df_al.index, df_al[col], c ="g",s=sz,label="Log")
    plt.ylabel('Log_Scale')
    plt.title(algor+'_'+col,fontsize=fs)
    plt.legend(loc='upper center',fontsize=fs)
    plt.tight_layout()
    anom[col]=df['Score']
    return anom


def cpd(df,tit,f,f_c,cpds):
    col_n=df.shape[1]
    fig, ax = plt.subplots(col_n//3+1, f_c, figsize=(f,f))
    ax = ax.ravel()
    ii=0
    for c in range(col_n):
        df_col=df.iloc[:,c]
        n = df.shape[0]
        points = np.log(df_col).to_numpy()
        algo = rpt.Pelt(model="rbf").fit(points)
        result = algo.predict(pen=40)
        ax[ii].plot(np.arange(1, n+1), points, linestyle='dotted',linewidth=1.0, c='b')
        for r in result:
            ax[ii].axvline(x=r,color='r',linestyle='--')
        ax[ii].set_xlabel('Time',fontsize=16)
        ax[ii].set_ylabel('Log (px)',fontsize=16)
        ax[ii].set_title('CPD: '+tit+': {}'.format(df_col.name),fontsize=20)
        ax[ii].tick_params(axis='both', labelsize=16)
        result = [r-1 for r in result]
        cpds[df_col.name]=result
        ii+=1
    return cpds


def cpd_group(df,dic,f,f_c,cpds):
    key, ii, n = dic.keys(),0,df.shape[0]
    fig, ax = plt.subplots(len(key)//3+1, f_c, figsize=(f,f))
    ax = ax.ravel()
    for col in key:
        df_col = df[dic[col]]
        points = np.log(df_col).to_numpy()
        points_df = pd.DataFrame(points)
        algo = rpt.Pelt(model="rbf").fit(points)
        result = algo.predict(pen=40)
        ax[ii].plot(np.arange(1, n+1), points_df, linestyle='dotted',linewidth=1.0, c='black')
        for r in result:
            ax[ii].axvline(x=r,color='r',linestyle='--')
        ax[ii].set_xlabel('Time',fontsize=16)
        ax[ii].set_ylabel('Log (px)',fontsize=16)
        ax[ii].set_title('CPD: '+col,fontsize=20)
        ax[ii].tick_params(axis='both', labelsize=16)
        result = [r-1 for r in result]
        cpds[col]=result
        ii+=1
    return cpds


def anom_index(df,dic,algor,lw,anom,f,fs):
    key, ii = dic.keys(),0
    fig, ax = plt.subplots(len(key)//3+1, 3, figsize=(f,f))
    ax = ax.ravel()
    for col in key:
        df_col = df[dic[col]].reset_index(drop=True)
        df_col=np.log(df_col)
        time.sleep(5)
        ind = algor.fit_predict(df_col.values)
        ind = np.where(ind == -1)[0].tolist()
        ax[ii].plot(df_col,linestyle='solid',linewidth=lw, color='black')
        for i in ind:
            ax[ii].axvline(x=i,color='r',linestyle='--')
        ax[ii].set_ylabel('Log_Scale')
        ax[ii].set_title('Anomalies_'+col,fontsize=fs)
        ax[ii].legend(loc='upper center',fontsize=fs)
        anom[col]=ind
        ii+=1
    return anom

def add_dict_to_df(dics,df):
    added=[]
    for dic in dics:
        this_dic=dics[dic]
        for k in this_dic:
            indices,name_c = this_dic[k],dic+k
            added.append(name_c)
            df[name_c]=0
            df.iloc[indices,-1]=1
    return df,added
    

def trend_online(df,lamb,trends,j,cut):
    solver = cvxpy.ECOS
    reg_norm = 1
    col_n=df.shape[1]
    for c in range(col_n):
        extension=[]
        start_time = time.time()
        for r in range(cut+j,len(df)+j,j):
            if r>=len(df):
                j = j-(r-len(df))
                r=len(df)
            else:
                pass
            #print(r,j)
            new_df = df.iloc[:r,:]
            df_col=new_df.iloc[:,c]
            n = new_df.shape[0]
            ones_row = np.ones((1, n))
            D = scipy.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)
            colu = np.log(df_col).to_numpy()
            x = cvxpy.Variable(shape=n) 
            obj = cvxpy.Minimize(0.5 * cvxpy.sum_squares(colu-x) + lamb * cvxpy.norm(D@x, reg_norm))
            problem = cvxpy.Problem(obj)
            problem.solve(solver=solver, verbose=False)
            extension.extend(x.value[-j:])
        trends[df_col.name]=np.array(extension)
        end_time = time.time()
        print("Elapsed Time: {} for Column: {}".format(round(end_time - start_time,2),df_col.name))
    return trends

def plot_trends(df_batch,df_online,f_c,f,cut):
    col_n,cols = df_batch.shape[1],df_batch.columns.tolist()
    fig, ax = plt.subplots(col_n//3+1, f_c, figsize=(f,f))
    ax = ax.ravel()
    ii=0
    for c in range(col_n):
        ax[ii].plot(df_batch.iloc[int(cut/2):cut,c], linestyle='solid',linewidth=1.0, c='b') 
        ax[ii].plot(df_batch.iloc[cut:,c], linestyle='dotted', linewidth=4.0, c='r',label='Batch')
        ax[ii].plot(df_online.iloc[:,c], linestyle='dashed', linewidth=2.0, c='g',label='Online')
        ax[ii].set_xlabel('Time',fontsize=16)
        ax[ii].set_ylabel('Log (px)',fontsize=16)
        ax[ii].set_title('Batch vs Online [Trends] '+cols[c],fontsize=20)
        ax[ii].legend()
        ax[ii].tick_params(axis='both', labelsize=16)
        ii+=1

    
def plot_1_trend(df_batch,df_online,c,cut):
    plt.plot(df_batch.iloc[int(cut/4):cut,:][c], linestyle='solid',linewidth=1.0, c='b') 
    plt.plot(df_batch.iloc[cut:,:][c], linestyle='dotted', linewidth=4.0, c='r',label='Batch')
    plt.plot(df_online[c], linestyle='dashed', linewidth=2.0, c='g',label='Online')
    plt.xlabel('Time',fontsize=16)
    plt.ylabel('Log (px)',fontsize=16)
    plt.title('Batch vs Online: '+c,fontsize=20)
    plt.legend()
    plt.tick_params(axis='both', labelsize=16)



"""
loadings_of_returns(all_col,funda_col,comp_col,load,5)
biplot('PC_1','PC_2',load.loc[ret_col,:],sz_bip,descr_ret['PC_1'][1]+" & "+descr_ret['PC_2'][1],6)
biplot('PC_1','PC_2',load.loc[funda_col,:],sz_bip,descr_funda['PC_1'][1]+" & "+descr_funda['PC_2'][1],7)
biplot('PC_3','PC_4',load.loc[ret_col,:],sz_bip,descr_ret['PC_3'][1]+" & "+descr_ret['PC_4'][1],6)
biplot('PC_3','PC_4',load.loc[funda_col,:],sz_bip,descr_funda['PC_3'][1]+" & "+descr_funda['PC_4'][1],7)
biplot('PC_5','PC_6',load.loc[ret_col,:],sz_bip,descr_ret['PC_5'][1]+" & "+descr_ret['PC_6'][1],6)
biplot('PC_5','PC_6',load.loc[funda_col,:],sz_bip,descr_funda['PC_5'][1]+" & "+descr_funda['PC_6'][1],7)
biplot('PC_5','PC_7',load.loc[ret_col,:],sz_bip,"PC_7_"+descr_ret['PC_7'][1],6)
biplot('PC_5','PC_7',load.loc[funda_col,:],sz_bip,"PC_7_"+descr_funda['PC_7'][1],7)

descr_ret={"PC_1":["","[Treas vs Rest]"], 
       "PC_2":["","[GLD vs DXY]"], 
       "PC_3":["","[Oil vs Rest]"], 
       "PC_4":["","[Commo-exIN vs EM&HY]"],
       "PC_5":["","[Commo vs Rest]"],
       "PC_6":["","[Metals&EM vs HY&DXY&Value&Soft]"],
       "PC_7":["","[GLD vs DXY&Soft&Oil&SML]"]}
descr_funda={"PC_1":["","[Tight_FC,High_Volat]"], 
       "PC_2":["","[High_MOVE_Lever,Low_Prof_Yield]"], 
       "PC_3":["","[Low_Prof_Vix, High_Lev_Yield]"], 
       "PC_4":["","[High_Valuat_Prof_Volat,Neg_News]"],
       "PC_5":["","[High_Prof,Low_Valuat]"], 
       "PC_6":["","[Low_Volat,Pos_News]"],
       "PC_7":["","[High_ROE_PCF,Low_PE]"]}
    
"""



