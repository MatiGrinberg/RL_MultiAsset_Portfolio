# Libraries
from Libraries import *
dig=6
ret_col=['LUACTRUU Index', 'LUATTRUU Index', 'LG50TRUU Index', 'EMUSTRUU Index', 'SPX Index', 'CCMP Index', 'NKY Index', 'HSI Index', 'UKX Index', 'SXXP Index', 'MXEF Index', 'BCOMEN Index', 'BCOMPR Index', 'BCOMSO Index', 'BCOMLI Index', 'BCOMIN Index', 'BCOMGR Index']
indu_col=['S5ENRS Index', 'S5INFT Index', 'S5COND Index', 'S5HLTH Index', 'S5FINL Index', 'S5UTIL Index', 'S5CONS Index', 'S5INDU Index', 'S5TELS Index', 'S5MATR Index', 'S5RLST Index']


def change_order_df(d,q,pcs):
    cant=len(d)/pcs
    fr,to = int((q-1)*cant),int(q*cant)
    ts = d.iloc[fr:to,:]
    idxs=ts.index
    new = pd.concat([d.drop(idxs),ts],axis=0)
    return new,idxs

def IRs(irs,b,d,t):
    irs=np.array(irs)
    cou,lent,avg = Counter(irs>0),len(irs),irs.mean()
    print(t)
    print('Len {} Rate {} Mean {}'.format(lent,round(cou[True]/lent,d),round(avg,d)))
    pd.Series(irs).hist(bins=b)


def plot_hist(d,b=100,f_r=2,f_c=2,f=10):
    fig, ax = plt.subplots(f_r, f_c, figsize=(f,f))
    ax = ax.ravel()
    for i in range(d.shape[1]):
        col= d.iloc[:,i]
        ax[i].hist(col,bins=b)
        ax[i].axvline(x=0,color='r',linestyle='dotted',linewidth=2)
        ax[i].set_title(col.name)

def corr(d,dig):
    plt.figure(figsize = (20,20))
    sns.heatmap(np.round(d.corr(),dig), cmap="YlGnBu", annot=True)
    plt.title('Corr_Matrix')
    

def pct_changes(d,days,col):
    n=pd.DataFrame()
    for i in days:
        n[[c.replace(' Index','_%ch_'+str(i)) for c in col]] = d.pct_change(periods=i)
        n[[c.replace(' Index','_diff_'+str(i)) for c in col]] = n.iloc[:,-len(col):].diff(i)
    return n

def ratio(df,list_col):
    for i in range(len(list_col)-1):
        num = list_col[i]
        for j in range(i+1,len(list_col)):
            den = list_col[j]
            try:
                df[num+'/'+den]=df[num]/df[den]
            except:
                pass
    return df

def diff_in_time(d,col,rang):
    for r in rang:
        new_col = [c+'_dif_'+str(r) for c in col]
        d[new_col] = d[col].diff(r)
    return d

def diff_between(d,list_col):
    for i in range(len(list_col)-1):
        num = list_col[i]
        for j in range(i+1,len(list_col)):
            den = list_col[j]
            d[num+'_gap_'+den]=d[num]-d[den]
    return d

def colu(d):
    return d.columns.tolist()

def compare_rows(df1,df2):
    compare1 = set(df1.index)-set(df2.index)
    compare2 = set(df2.index)-set(df1.index)
    inter = list(set(df1.index) & set(df2.index))
    return sorted(list(compare1)),sorted(list(compare2)),sorted(inter)

def compare_col(d1,d2):
    d1_col, d2_col = d1.columns.tolist(), d2.columns.tolist()
    dif1 = set(sorted(d1_col)) - set(sorted(d2_col))  
    dif2 = set(sorted(d2_col)) - set(sorted(d1_col))
    inter = list(set(d1_col) & set(d2_col))
    return dif1,dif2,inter

def mov_av(d,col,rang):
    for r in rang:
        d[[c+'_MA_'+str(r) for c in col]]=d[col].ewm(span=r,min_periods=r).mean()
    return d

def toy_df(mn,mx,l,w,col):
    df = pd.DataFrame(np.random.randint(mn,mx,size=(l,w)), columns=list(col))
    return df

def toy_df_fl(mn,mx,l,w,col):
    df = pd.DataFrame(np.random.uniform(mn,mx,size=(l,w)), columns=list(col))
    return df

def read_excel(f,s,h=0,path=None):
    if path==None:
        df = pd.read_excel(f,s,engine="openpyxl",header=h)
    else:
        df = pd.read_excel(path+f,s,engine="openpyxl",header=h)
    return df

def print_text(t,w=3,n_char=40):
    print('\n'+'#'*n_char+'\n'+t+'\n'+'#'*n_char)
    time.sleep(w)

def print_excel(df,f,s):
    wb = openpyxl.load_workbook(f)
    ws = wb[s]
    rows = dataframe_to_rows(df, index=False)
    for r_idx, row in enumerate(rows, 1):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)
    wb.save(f)
    print('\nFinished: Copying to Excel')

def ann_ret(tot,ite,i_per_an,dig):
    re_an = ((tot**(1/ite)-1)+1)**i_per_an-1
    return round(re_an,dig)

def null_rows(d):
    return d[d.isnull().any(axis=1)]

def compare_lists(df1,df2):
    compare1 = set(df1)-set(df2)
    compare2 = set(df2)-set(df1)
    inter = list(set(df1) & set(df2))
    return compare1,compare2,inter

def null_col_f(d,th):
    null_col = round(d.isnull().sum()/len(d),2)
    null_col2 = null_col[null_col>th].sort_values()
    null_col_nam = null_col2.index
    print('\nCol Below_Threshold:')
    print(null_col[(null_col<=th)*(null_col>0)].sort_values())
    print('\nCol Above_Threshold:')
    print(null_col2)
    return null_col_nam

def read_pickle(p,path=None):
    df = pd.read_pickle(path+p)
    return df
 
def save_pickle(df,p,path=None):
    pickle_out = open(path+p,"wb")
    pickle.dump(df, pickle_out)
    pickle_out.close()


def IR(df,p,b,pe,a):
    dif_r = (df[p+'_Cum'].iloc[-1]**(1/pe)-1)-(df[b+'_Cum'].iloc[-1]**(1/pe)-1)
    sd_r = np.std(df[p]-df[b])
    ir = round(dif_r/sd_r*a,1)
    return ir

def df_dty(df):
    dty = df.dtypes
    print(dty.unique())
    return dty

def calc_vol(d,ran):
    return d.rolling(ran).std()#/d.rolling(ran).mean()

    
    
def plot_veloc_acceler(d,fr,to,nro):
    plt.plot(d[[c for c in colu(d) if 'PC_'+str(nro) in c]].iloc[fr:to,:])
    plt.axhline(y = 0.0, color = 'red', linestyle = 'dotted',linewidth= 2)
    plt.tick_params(axis="x",rotation=45,labelsize='x-small')
    plt.legend([str(nro),'dif','dif_of_dif'],fontsize=8)
    plt.show()

    
def plot_return_2axis(d,cols,t):
    fig,ax = plt.subplots()
    ax.plot(d[cols[0]],color="orange")
    ax.legend([d[cols[0]].name])
    ax.tick_params(axis="x",rotation=45,labelsize='small')
    ax2=ax.twinx()
    ax2.plot(d[cols[1]],color="blue")
    ax2.legend([d[cols[1]].name],loc='lower right')
    plt.title(t)
    
def plot_2_axis(df,col1,col2,m):
    fig,ax = plt.subplots()
    ax.plot(df[col1],color="red")
    ax.axhline(y = 0.0, color = 'orange', linestyle = 'dotted',linewidth= 2)
    ax.set_ylabel('Weight',color="red")
    ax.set_title(m)
    ax.tick_params(axis="x",rotation=45,labelsize='x-small')
    ax2=ax.twinx()
    ax2.plot(df[col2],color="blue")
    ax2.axhline(y = 0.0, color = 'cyan', linestyle = 'dotted',linewidth= 2)
    ax2.set_ylabel('Bench_Ret',color="blue")

def make_df_gr(l_r,l_b,l_a,col_w,idx):
    l_rew,l_ben,l_act_n = np.array(l_r),np.array(l_b),np.array(l_a)    
    df_gr = pd.DataFrame({'Rew':np.reshape(l_rew,len(l_rew)),'Bench':np.reshape(l_ben,len(l_ben))})
    df_gr['Rew_Cum'] = (1+df_gr['Rew']).cumprod()
    df_gr['Bench_Cum'] = (1+df_gr['Bench']).cumprod()
    df_gr = pd.concat([df_gr,pd.DataFrame(l_act_n, columns=col_w)], axis=1)
    df_gr.index=idx
    return df_gr.fillna(0)

def plot_w(d,ass,m='Weight',f=10):
    fig,ax = plt.subplots(figsize=(f,f))
    ax.plot(d[ass],linewidth= 3)
    ax.set_ylabel('Weights',color="black")
    ax.set_title(m)
    ax.legend([ass])
    ax.tick_params(axis="x",rotation=45,labelsize='x-small')
    ax.axhline(y = 0.0, color = 'red', linestyle = 'dotted',linewidth=4)
    plt.tight_layout()
    

def wei_subpl(d,cols):
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    ax = ax.ravel()
    ii=0
    for c in cols:
        ax[ii].plot(d.index, d[c],label=c)
        ax[ii].legend(loc='upper left')
        ax[ii].axhline(y = 0, linewidth=4,linestyle='dotted',color='y')
        ax[ii].tick_params(axis="x",rotation=45,labelsize='x-small')
        ii+=1
    fig.suptitle('100% weight distributed betw 3 Assets')
    plt.savefig('w.png')
    
    


def weights_plot_divided(d,cols,q=4):
    not_all_0=d.apply(lambda x: np.all(x == 0))
    cols=[c for c in cols if c in not_all_0[not_all_0==0]]
    for i in range(0,len(cols),q):
        wei_subpl(d,cols[i:(i+q)])
    
        

def plot_return(d,cols,t):
    fig,ax = plt.subplots()
    ax.plot(d[cols])    
    ax.legend(cols)
    ax.tick_params(axis="x",rotation=45,labelsize='x-small')
    fig.text(0.5,0.5,t[t.find("IR",0):t.find("Corr",0)][:-2],color='green', fontsize=15)
    plt.title(t)
    plt.savefig(t[-4:]+'.png')
    
    
def save_to_doc():
    doc = Document('Validation.docx')
    table = doc.add_table(rows=1, cols=2)
    row_cells = table.rows[0].cells
    row_cells[0].paragraphs[0].add_run().add_picture('Port.png', width=Inches(3))  # Adjust width as needed
    row_cells[1].paragraphs[0].add_run().add_picture('w.png', width=Inches(3))  # Adjust width as needed
    doc.add_paragraph()
    doc.save('Validation.docx')

    
    
def array_equal(ar):
    print('Min in Array: {}'.format(np.min(ar)))
    print('Max in Array: {}'.format(np.max(ar)))
    print('All elements equal: {}'.format(np.all(ar == ar[0])))
    

class Assets(Env):
    def __init__(self,x,y,l_rew,l_act,l_tar,l_i,l_obs,l_act_n,l_ben,idx,cum_p=np.array([1]),cum_b=np.array([1])):
        self.x,self.y ,self.idx,self.cum_p,self.cum_b = x,y,idx,cum_p,cum_b
        self.l_rew,self.l_i,self.l_act_n,self.l_tar,self.l_obs,self.l_act,self.l_ben= l_rew,l_i,l_act_n,l_tar,l_obs,l_act,l_ben
        #self.action_space = spaces.Box(low=-1, high=1,shape=(self.y.shape[1],), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3 for _ in range(self.y.shape[1])])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self.x.shape[1],), dtype=np.float32)
        self.reset()
    def app(self,r,t,i,a_n,o,a):
        self.l_rew.append(r),self.l_i.append(i),self.l_act_n.append(np.round(a_n,dig)),self.l_tar.append(t),self.l_obs.append(o),self.l_act.append(np.round(a,dig)),self.l_ben.append(t[self.idx])    
    def step(self,action):
        action=action-1
        if list(action).count(0)==self.y.shape[1]:
            self.reward=0
        else:
            self.reward = round(np.dot(action / sum(abs(action)),self.target.T).item(),dig)
        self.app(self.reward,self.target,self.i,(action/sum(abs(action))),self.observation,action)
        print('Rew {} Bench_W {}'.format(np.round(self.reward*100,dig),round((action[self.idx]/sum(abs(action))),2))) # np.round(action,2),   ,np.round(100*((action/sum(abs(action))).std()),0)
        self.cum_p=np.append(self.cum_p,self.cum_p[-1]*(1+self.reward))
        self.cum_b=np.append(self.cum_b,self.cum_b[-1]*(1+self.target[self.idx]))
        """
        if self.cum_p[-1]<0.8*np.max(self.cum_p) or self.cum_p[-1]<0.8*self.cum_b[-1]:
            self.reward-=1
        else:
            pass
        """
        self.i+=1    
        if self.i==self.y.shape[0]:
            self.done=True
        else:
            self.done=False
            self.target = np.round(self.y[self.i],dig)
            self.observation = np.round(self.x[self.i],dig)
        info={}
        return self.observation, self.reward, self.done, info
    def reset(self):
        self.done,self.reward,self.i,self.sum,self.cum_p,self.cum_b = False,0,0,0,np.array([1]),np.array([1])
        self.target = np.round(self.y[self.i],dig)    
        self.observation = np.round(self.x[self.i],dig)
        return self.observation
    def render(self,action):
        print('I{} Act:{} Rew:{} Tar:{}'.format(self.i,action,self.reward,self.target),'*'*10)
        print('L_i:{} L_Act:{} L_Rew:{} L_Tar{}'.format(l_i,l_act,l_rew,l_tar),'*'*20)



"""

class Assets(Env):
    def __init__(self,x,y,l_rew,l_act,l_tar,l_i,l_obs,l_act_n,l_ben,idx,cum_p=np.array([1]),cum_b=np.array([1])):
        self.x,self.y ,self.idx,self.cum_p,self.cum_b = x,y,idx,cum_p,cum_b
        self.l_rew,self.l_i,self.l_act_n,self.l_tar,self.l_obs,self.l_act,self.l_ben= l_rew,l_i,l_act_n,l_tar,l_obs,l_act,l_ben
        self.action_space = spaces.Box(low=-1, high=1,shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self.x.shape[1],), dtype=np.float32)
        self.reset()
    def app(self,r,t,i,a_n,o,a):
        self.l_rew.append(r),self.l_i.append(i),self.l_act_n.append(np.round(a_n,dig)),self.l_tar.append(t),self.l_obs.append(o),self.l_act.append(np.round(a,dig)),self.l_ben.append(t[self.idx])
    def step(self,action):
        self.reward = round((action*self.target[self.idx]).item(),dig)
        self.app(self.reward,self.target,self.i,action,self.observation,action)
        print('Rew {} Oil_W {}'.format(np.round(self.reward*100,dig),action))
        self.cum_p=np.append(self.cum_p,self.cum_p[-1]*(1+self.reward))
        self.cum_b=np.append(self.cum_b,self.cum_b[-1]*(1+self.target[self.idx]))
        self.i+=1    
        if self.i==self.y.shape[0]:
            self.done=True
        else:
            self.done=False
            self.target = np.round(self.y[self.i],dig)
            self.observation = np.round(self.x[self.i],dig)
        info={}
        return self.observation, self.reward, self.done, info
    def reset(self):
        self.done,self.reward,self.i,self.cum_p,self.cum_b = False,0,0,np.array([1]),np.array([1])
        self.target = np.round(self.y[self.i],dig)    
        self.observation = np.round(self.x[self.i],dig)
        return self.observation
    def render(self,action):
        print('I{} Act:{} Rew:{} Tar:{}'.format(self.i,action,self.reward,self.target),'*'*10)
        print('L_i:{} L_Act:{} L_Rew:{} L_Tar{}'.format(l_i,l_act,l_rew,l_tar),'*'*20)

"""