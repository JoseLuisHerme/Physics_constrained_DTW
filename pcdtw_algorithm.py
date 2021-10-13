#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:19:31 2021

@author: Jose Luis Hernandez Mejia
"""

import numpy as np #Array operations
import matplotlib.pyplot as plt #Plotting 
import pandas as pd #To handle dataframes
import scipy.spatial #Spatial calculations
import math #Mathematical Operations 
from sklearn.metrics import mean_squared_error #Evalute MSE between water injection and optimal match 


df1 = pd.read_csv('Prod_Data.csv') #Data loading
df1['Oil_complete'] = df1['Oil(bbls)'].fillna(0)

df2 = pd.read_csv('Injection_Data.csv') #Data loading

water_inj=df2['Volume Injected'].to_numpy()[0:176]
oil_prod=df1['Oil_complete'].to_numpy()[0:176]


prd_time=np.arange(0,len(oil_prod),1).reshape(-1,1)
inj_time=np.arange(0,len(water_inj),1).reshape(-1,1)

#Normalized time series plot 
#iny=np.arange(0,len(Y),1)
#inx=np.arange(0,len(X),1)
fig, ax1 = plt.subplots(figsize=(10,4))
#ax2 = ax1
lns1 =ax1.plot(prd_time,oil_prod,color='Black',linestyle='-',marker='o',markersize='2',linewidth='1',label='Water injection')
#lns2 =ax2.plot(iny,Y,color='black',marker='o',markersize='2',linewidth='1',label='Oil production')
#lns = lns1+lns2
#labs = [l.get_label() for l in lns]
#ax1.legend(lns, labs, loc=0)
ax1.set_xlabel('Time Index (Month)')
ax1.set_ylabel('Oil production, Bls',color='black')
plt.show()

#Normalized time series plot 
#iny=np.arange(0,len(Y),1)
#inx=np.arange(0,len(X),1)
fig, ax1 = plt.subplots(figsize=(10,4))
#ax2 = ax1
lns1 =ax1.plot(inj_time,water_inj,color='Black',linestyle='-',marker='o',markersize='2',linewidth='1',label='Water injection')
#lns2 =ax2.plot(iny,Y,color='black',marker='o',markersize='2',linewidth='1',label='Oil production')
#lns = lns1+lns2
#labs = [l.get_label() for l in lns]
#ax1.legend(lns, labs, loc=0)
ax1.set_xlabel('Time Index (Month)')
ax1.set_ylabel('Water Injected, Bls',color='black')
plt.show()



X=water_inj
Y=oil_prod

#Data normlization
X=norm(X).reshape(-1,1)
Y=norm(Y).reshape(-1,1)
time_index=np.arange(0,len(X),1)

#Normalized time series plot 
iny=np.arange(0,len(Y),1)
inx=np.arange(0,len(X),1)
fig, ax1 = plt.subplots(figsize=(10,4))
ax2 = ax1
lns1 =ax1.plot(inx,X,color='gray',linestyle='--',marker='o',markersize='2',linewidth='1',label='Water injection')
lns2 =ax2.plot(iny,Y,color='black',marker='o',markersize='2',linewidth='1',label='Oil production')
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
ax1.set_xlabel('Time Index')
ax1.set_ylabel('Normalized oil production and water injection',color='black')
plt.show()


C = scipy.spatial.distance.cdist(X, Y, 'euclidean') 
plt.figure(figsize=(10,4))
im2=plt.imshow(C, origin='lower', aspect='equal',cmap='viridis')
cbar2=plt.colorbar(im2)
cbar2.set_label('Cost', rotation=270, fontsize = 11, labelpad=15)
plt.xlabel('Normalized oil production', fontsize = 11)
plt.ylabel('Normalized water injected', fontsize = 11);

P_cdtw, D_cdtw=dtw_constrain(X,Y,2,19)
#plot cummulative cost matrix
plt.figure(figsize=(10,4))
im3=plt.imshow(D_cdtw, origin='lower', aspect='equal',cmap='viridis')
cbar2=plt.colorbar(im3)
cbar2.set_label('Constrain cumulative cost', rotation=270, fontsize = 11, labelpad=15)
plt.xlabel('Normalized oil production', fontsize = 11)
plt.ylabel('Normalized water injected', fontsize = 11);

optimal_path_plot(X,Y,P_cdtw,D_cdtw)
aligment_plot(X,Y,P_cdtw,1)

time_lag=[]
for i in range(len(P_cdtw)):
    time_lag.append(P_cdtw[:,1][i]-P_cdtw[:,0][i])
print ('Mean lag time between water injection and oil production: ',round(np.mean(np.array(time_lag)),2),'Months')
print ('Water break-trhought time :',np.min(np.array(P_cdtw[:,1])),'Months')

ang=np.arange(1,47,2)
ss=np.arange(1,22,1)
cost=[]
mse=[]
cosim=[]
matched_points=[]
for i in ss:
    for j in ang:
        P_norm, D_norm=dtw_constrain(X,Y,i,j)
        matched_points.append(len(P_norm))        
        C_pn = scipy.spatial.distance.cdist(X,Y, 'euclidean')
        wpc=[]
        for (n, m) in P_norm:
            wpc.append(C_pn[n, m])
        c_P=np.sum(wpc)
        cost.append(c_P)
        
        match_X=[]
        match_Y=[]
        for k in range(len(P_norm)):
            match_X.append(X[P_norm[:,0][k]])
            match_Y.append(Y[P_norm[:,1][k]])
        err=mean_squared_error(match_X,match_Y)
        mse.append(err)
        
        u=np.array(match_X).reshape(-1,)
        v=np.array(match_Y).reshape(-1,)
    
        simm=cosine_similarity(u, v)
        print ('Omega: ',i,'Alpha: ',j, 'Cost: ',c_P, 'MSE: ',err,'Cosine Simularity: ',simm)
        cosim.append(simm)



cost=np.array(cost).reshape(len(ss),len(ang))
mse=np.array(mse).reshape(len(ss),len(ang))
sim=np.array(cosim).reshape(len(ss),len(ang))
matched_points=np.array(matched_points).reshape(len(ss),len(ang))
hyperparameter_tunning_plots(cost,mse,sim,matched_points)

np.where(mse == np.min(mse))


P_cdtw, D_cdtw=dtw_constrain(X,Y,5,35)
#plot cummulative cost matrix
plt.figure(figsize=(10,4))
im3=plt.imshow(D_cdtw, origin='lower', aspect='equal',cmap='viridis')
cbar2=plt.colorbar(im3)
cbar2.set_label('Constrain cumulative cost', rotation=270, fontsize = 11, labelpad=15)
plt.xlabel('Normalized oil production', fontsize = 11)
plt.ylabel('Normalized water injected', fontsize = 11);

optimal_path_plot_fix_2(X,Y,P_cdtw,D_cdtw)

optimal_path_plot(X,Y,P_cdtw,D_cdtw)
aligment_plot(X,Y,P_cdtw,1)

time_lag=[]
for i in range(len(P_cdtw)):
    time_lag.append(P_cdtw[:,1][i]-P_cdtw[:,0][i])
print ('Mean lag time between water injection and oil production: ',round(np.mean(np.array(time_lag)),2),'Months')
print ('Water break-trhought time :',np.min(np.array(P_cdtw[:,1])),'Months')


def norm(X):
    """Normilize time series from 0 to 1 range
    Args:
        X: Numpy array of the time series
    Returns:
        Xn: Normilized time series from 0 to 1 
    """
    maxx=np.max(X)
    minn=np.min(X)
    rangee=maxx-minn
    Xn=(X-minn)/rangee
    return Xn

def optimal_path_plot(X,Y,P,D):
    """Creates a vizual representation of the optimal warping path
    Args:
        X: Normilized water injection time series
        Y: Normilized oil production time series
        P: Optimal Warping path
        D: Cumulative cost matrix
    Returns:
    Tree way plot of the optimal warping path 
    """
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2,2,width_ratios=[1, 4])

    ax_joint = fig.add_subplot(gs[0,1])
    ax_marg_x =  fig.add_subplot(gs[1,1])
    ax_marg_y = fig.add_subplot(gs[0,0])

    iny=np.arange(0,len(Y),1)
    prd=np.arange(0,len(X),1)
    ax_joint.imshow(D, origin='lower', aspect='equal',cmap='viridis')
    ax_joint.plot(P[:, 1], P[:, 0], color='red',lw=2,label='Optimal Warping Path')
    ax_joint.legend()
    ax_marg_x.plot(iny,Y,color='black',marker='o',markersize='2',linewidth='1')
    ax_marg_y.plot(X,prd,color='gray',linestyle='--',marker='o',markersize='2',linewidth='1')

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=True)
    plt.setp(ax_marg_y.get_yticklabels(), visible=True)

    # Set labels on marginals
    ax_marg_y.set_ylabel('Normilized water injection time index')
    ax_marg_x.set_xlabel('Normilized oil production time index')
    plt.show()
    
def compute_accumulated_cost_matrix(C,step_size=1):
    """Compute the accumulated cost matrix given the cost matrix with a different step size constrain 21
    Args:
        C: cost matrix
        step_size: Step_size in the minimization distance
    Returns:
        D: Accumulated cost matrix
    """
    N = C.shape[0]
    M = C.shape[1]
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n-1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m-1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D

def compute_optimal_warping_path_subsecuent(D,step_size=1):
    """Given an accumulated cost matrix, Compute the warping path for clasical subsequence dynamic time warping for
    step sizes
    Args:
        D: Accumulated cost matrix
        step_size: Step_size of the minimazion process

    Returns
        P: Warping path (list of index pairs)
    """
    if step_size==1:
        N, M = D.shape
        n = N - 1
        m = D[N - 1, :].argmin()
        P = [(n, m)]

        while n > 0:
            if m == 0:
                cell = (n - 1, 0)
            else:
                val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
                if val == D[n-1, m-1]:
                    cell = (n-1, m-1)
                elif val == D[n-1, m]:
                    cell = (n-1, m)
                else:
                    cell = (n, m-1)
            P.append(cell)
            n, m = cell
        P.reverse()
        return np.array(P)
    
    elif step_size==2:
        N, M = D.shape
        n = N - 1
        m = D[N - 1, :].argmin()
        P = [(n, m)]

        while n > 0 or m > 0:
            if n == 0:
                cell = (0, m - 1)
            elif m == 0:
                cell = (n - 1, 0)
            else:
                val = min(D[n-1, m-1], D[n-2, m-1], D[n-1, m-2])
                if val == D[n-1, m-1]:
                    cell = (n-1, m-1)
                elif val == D[n-2, m-1]:
                    cell = (n-2, m-1)
                else:
                    cell = (n-1, m-2)
            P.append(cell)
            (n, m) = cell
        P.reverse()
        return np.array(P)
    
    elif step_size==3:
        N, M = D.shape
        n = N - 1
        m = D[N - 1, :].argmin()
        P = [(n, m)]

        while n > 1 or m > 1:
            if n == 0:
                cell = (0, m - 1)
            elif m == 0:
                cell = (n - 1, 0)
            else:
                val = min(D[n-3, m-1], D[n-1, m-1], D[n-1, m-3])

                if val == D[n-3, m-1]:
                    cell = (n-3, m-1)

                elif val == D[n-1, m-1]:
                    cell = (n-1, m-1)

                else:
                    cell = (n-1, m-3)
            P.append(cell)
            (n, m) = cell
        P.reverse()
        return np.array(P)
    
def aligment_plot(X,Y,P,offset):
    """Plot the time series aligment given the optimal warping path
    Args:
        X: Normilized water injection
        Y: Normilized oil production
        P: Optimal warping path
        offset: Re-scale the time series for better vizualuzation. Recomended value = 1 
    Returns
        Optimal time series aligment plot
    """
    xmin=np.min(X)
    xmax=np.max(X)
    xrange=xmax-xmin
    xn=(X-xmin)/(xrange)
    iny=np.arange(0,len(Y),1)
    inx=np.arange(0,len(X),1)
    yn=((Y-xmin)/(xrange))+offset
    plt.figure(figsize=(10, 4))
    plt.plot(inx,xn,color='gray',linestyle='--',marker='o',markersize='2',linewidth='1',label='Normilized water injection')
    plt.plot(iny,yn,color='black',marker='o',markersize='2',linewidth='1',label='Normilized oil production ')
    plt.xlabel('Time Index')
    plt.ylabel('Offset normilized oil prodiction and water injection')
    
    for i in range(len(P)):
        plt.plot([P[:,0][i], P[:,1][i]],[xn[P[:,0][i]],yn[P[:,1][i]]],color='Black',ls='--',lw=1)
    plt.legend();
    
def slope(lp,sp):
    """Calculate the slope between a starting point and all the subsecuent points of the allowed serch space.
    Function used inside the constrain DTW Algorithm 
    Arguments:
        lp: Startin point          
        sp: search space 
    Returns:
        angles: Deviation angle between the starting point and all the subsecuent points in the allowed serch space
        sp: Slopes between a starting point and all the points in the allowed search space
    """
    slopes=[]
    for i in sp:     
        m=(lp[0]-i[0])/(lp[1]-i[1])
        slopes.append(m)
        
    angles=[]
    for m in slopes:
        ang=math.degrees(math.atan(m))
        angles.append(ang)
    
    angles=np.array(angles).reshape(-1,1)    
    return angles,sp

def dtw_constrain(X,Y,alpha,ome):
    """Calculates the optimal warping path between normilized water injection and oil production given
        hyperparamters alpha and omega 
    Args:
        X: Normilized water injection
        Y: Normilized oil production
        alpha: Search space constrain
        ome: Angle constrain
    Returns
        P: Warping path (list of index pairs)
    """
    C = scipy.spatial.distance.cdist(X, Y, 'euclidean') 
    #Construct cummulative cost matrix depending on alpha value
    N, M = C.shape
    delta=(alpha-1)
    Ni=N+delta
    Mi=M+delta
    Di = np.zeros((Ni, Mi))
    Di[delta:,delta] = np.cumsum(C[:, 0])
    Di[delta, delta:] = C[0, :]
    for n in range(delta+1, Ni):
        for m in range(delta+1, Mi):
            s_p=[]
            for i in range(alpha):
                for j in range(alpha):
                    s_p.append([n-1-i,m-1-j])
            sp=np.array(s_p)
            val=[]
            for i in range(len(sp)):
                val.append(Di[sp[i][0],sp[i][1]])
            Di[n, m] = C[n-delta, m-delta] + min(val)
    Di_beta= Di[delta:, delta:]
    
    ##Constrain back-in time
    for i in range(Di_beta.shape[0]):
        for j in range(Di_beta.shape[0]):
            if i<j:
                Di_beta[j,i]=np.inf
            else:
                pass
    ##Warping path
    N, M =  Di_beta.shape
    n = N - 1
    m = Di_beta[N - 1, :].argmin() #Warping path initizalization
    
    P_beta = [(n, m)]
    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            s_p=[]
            for i in range(alpha):
                for j in range(alpha):
                    s_p.append([n-1-i,m-1-j])
            search_space=np.array(s_p)
            
            ip=[n,m]
            angles,sp_upd=slope(ip,search_space)
            
            sp_new=[]
            up_boundary=45+ome
            lower_boundary=45-ome
            for a,b in zip(angles,sp_upd):
                if (a<=up_boundary) and (a>=lower_boundary): 
                    sp_new.append(b)
                else:
                    pass
            sp=np.array(sp_new) 
                    
            val=[]
            for i in range(len(sp)):
                val.append(Di_beta[sp[i][0],sp[i][1]])
            
            sor_val=pd.DataFrame(val,columns=['Val'])
            sor_val['cell']=sp_new
            sor_val_sort=sor_val.sort_values(by=['Val']).reset_index()
            cell_in=sor_val_sort['cell'][0]
            n=cell_in[0]
            m=cell_in[1]
            cell = (n, m)
        P_beta.append(cell)
        n, m = cell
    P_beta.reverse()
    P_beta=np.array(P_beta)        
    return  P_beta, Di_beta


def cosine_similarity(u, v):
    """Cosine similarity reflects the degree of similarity between X and Y
    Arguments:
        X: Normilized water injection         
        Y: Normilized oil production
    Returns:
        cosine_similarity:  the cosine similarity between X and Y.
    """
    # Compute the dot product between u and v 
    dot = np.dot(u,v)
    # Compute the L2 norm of u
    norm_u = np.sqrt(np.sum(u**2))
    # Compute the L2 norm of v
    norm_v =  np.sqrt(np.sum(v**2))
    # Compute the cosine similarity 
    cosine_similarity = dot/(norm_v*norm_u)
    return cosine_similarity

def hyperparameter_tunning_plots(cost,mse,sim,matched_points):
    """plot the cost, mse, sim and number of matched_points during hyperparameter tunning
    Arguments:
        cost: Cost matrix with dimentions dim(search_space,angle constrains)        
        mse: Mean squared error matrix with dimentions dim(search_space,angle constrains)  
        sim: cosine similarity matrix with dimentions dim(search_space,angle constrains) 
        matched_points: Number of points mateched during DTW matrix with dimentions dim(search_space,angle constrains)
    Returns:
        Plots of evaluation metrics vs hyperparamters 
    """
    
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(221)
    im2=plt.imshow(cost.T, aspect='equal',cmap='viridis')
    cbar2=plt.colorbar(im2,fraction=0.046, pad=0.04)
    cbar2.set_label('Cost', rotation=270, fontsize = 11, labelpad=10)
    plt.ylabel('Angle constrain', fontsize = 11)
    plt.autoscale(True)
    plt.yticks(range(len(ang)),np.round(ang, 1), fontsize = 8)
    plt.xticks(range(len(ss)),ss, fontsize = 8)

    ax2 = fig.add_subplot(222)
    im2=plt.imshow(mse.T, aspect='equal',cmap='viridis')
    cbar2=plt.colorbar(im2,fraction=0.046, pad=0.04)
    cbar2.set_label('MSE', rotation=270, fontsize = 11, labelpad=10)
    plt.autoscale(True)
    plt.yticks(range(len(ang)),np.round(ang, 1), fontsize = 8)
    plt.xticks(range(len(ss)),ss, fontsize = 8)

    ax3 = fig.add_subplot(223)
    im2=plt.imshow(sim.T, aspect='equal',cmap='viridis')
    cbar2=plt.colorbar(im2,fraction=0.046, pad=0.04)
    cbar2.set_label('Cosine Similarity', rotation=270, fontsize = 11, labelpad=10)
    plt.xlabel('Allowed search space', fontsize = 11)
    plt.ylabel('Angle constrain', fontsize = 11)
    plt.autoscale(True)
    plt.yticks(range(len(ang)),np.round(ang, 1), fontsize = 8)
    plt.xticks(range(len(ss)),ss, fontsize = 8)

    ax4 = fig.add_subplot(224)
    im2=plt.imshow(matched_points.T, aspect='equal',cmap='viridis')
    cbar2=plt.colorbar(im2,fraction=0.046, pad=0.04)
    cbar2.set_label('Matched points', rotation=270, fontsize = 11, labelpad=10)
    plt.xlabel('Allowed search space', fontsize = 11)
    plt.autoscale(True)
    plt.yticks(range(len(ang)),np.round(ang, 1), fontsize = 8)
    plt.xticks(range(len(ss)),ss, fontsize = 8); 

def optimal_path_plot_fix_2(X,Y,P,D):
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    N, M =  D.shape
    m = D[N - 1, :].argmin() 
    fig2 = plt.figure(figsize=(12, 10), dpi= 100,constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=2, nrows=3,width_ratios=[1, 4])
    f2_ax2 = fig2.add_subplot(spec2[0, 1])
    f2_ax3 = fig2.add_subplot(spec2[1, 0])
    f2_ax4 = fig2.add_subplot(spec2[1, 1])
    f2_ax6 = fig2.add_subplot(spec2[2, 1])
    iny=np.arange(0,len(Y),1)
    prd=np.arange(0,len(X),1)
    
    im = f2_ax4.imshow(D, origin='lower', aspect='auto',cmap='viridis')
    f2_ax4.plot(P[:, 1], P[:, 0], color='red',label='Optimal Warping path')
    #f2_ax4.plot(P_norm[:, 1], P_norm[:, 0], color='red',label='Signal')
    f2_ax4.legend()
    
    f2_ax6.plot(iny,Y,color='black',marker='o',markersize='2',linewidth='1')
    f2_ax3.plot(X,prd,color='gray',linestyle='--',marker='o',markersize='2',linewidth='1')
    
    f2_ax2.plot(D[N - 1, :],color='black',linewidth='1')
    f2_ax2.scatter(m,np.min(D[N - 1, :]),color='black',label='Global minimun')
    f2_ax2.legend()
    
    
    axins = inset_axes(f2_ax4, # here using axis of the lowest plot
                   width="2%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 340% good for a (4x4) Grid
                   loc='lower left',
                   bbox_to_anchor=(1.02,1.2,1, 1),
                   bbox_transform=f2_ax6.transAxes,
                   borderpad=0,
                   )
    cb = fig.colorbar(im, cax=axins)
    cb.set_label('Cumulative cost', rotation=270, fontsize = 11, labelpad=15)
    
    # Set labels on marginals
    f2_ax3.set_ylabel('Time index')
    f2_ax2.set_xlim(0, max(iny))
    f2_ax3.set_xlabel('Normalized water injection')
    f2_ax2.set_ylabel('Cumulative cost at N')
    f2_ax6.set_ylabel('Normalized oil production')
    f2_ax6.margins(x=0)
    f2_ax6.set_xlabel('Time index')
    f2_ax3.margins(y=0)
    f2_ax2.margins(x=0)
    plt.show()