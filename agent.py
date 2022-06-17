from dataset import *
import numpy as np
class BC_Agent():
    def __init__(self, hor, card_s, card_a, d_r, env,n_demon,n_add,mu_max):
        self.hor = hor
        self.card_s = card_s
        self.card_a = card_a
        self.d_r = d_r
        self.env =env
        self.n_demon = n_demon
        self.n_add = n_add
        self.alpha = 0.5
        self.delta = 500
        self.mu_max = mu_max

    def train(self,datasets):
        
        pi = np.zeros([self.hor, self.card_s, self.card_a])
        N = np.zeros([self.hor, self.card_s])
        for traj in datasets.traj:
            for h, tran in enumerate(traj):
                s, a, s_next = tran
                pi[h, s, a] += 1
                N[h,s]+=1
        for h in range(self.hor):
            for s in range(self.card_s):
                for a in range(self.card_a):
                    if N[h,s] == 0:
                        pi[h,s,a] = 1./self.card_a
                    else:
                        pi[h, s, a] = pi[h, s, a]/N[h,s]
        print("BC Performance: ", self.env._eval(pi))
        return self.env._eval(pi)

import math
class Agent():
    def __init__(self, hor, card_s, card_a, d_r, env,n_demon,n_add,mu_max,kappa,alpha,verbose,stepsize):
        self.hor = hor
        self.card_s = card_s
        self.card_a = card_a
        self.d_r = d_r
        self.env =env
        self.n_demon = n_demon
        self.n_add = n_add
        self.alpha = 1.0
        self.delta = 0.1
        self.mu_max = mu_max
        self.stepsize = 100
        self.kappa = kappa
        self.alpha = alpha
        self.verbose = verbose
        self.stepsize = stepsize
        
    def init_construct(self,datasets):
    
        tot = np.zeros([self.hor, self.card_s, self.card_a])
        vis = np.zeros([self.hor, self.card_s, self.card_a, self.card_s])
        est_p = np.zeros_like(vis)
        uq_p = np.zeros_like(tot)
        for traj in datasets.traj:
            for h, tran in enumerate(traj):
                s, a, s_next = tran
                vis[h, s, a, s_next] += 1
                tot[h, s, a] += 1
        for h in range(self.hor):
            for s in range(self.card_s):
                for a in range(self.card_a):
                    for s_next in range(self.card_s):
                        est_p[h, s, a, s_next] = vis[h, s, a, s_next] / max(tot[h, s, a],1)
                    uq_p[h, s, a] = self.hor * self.card_s * np.sqrt(np.log(2 * self.hor * self.card_s 
                        * self.card_a * self.card_s / self.delta) / 2 / max(tot[h, s, a],1))
        return uq_p, est_p
    def train(self,train_num):
        returns_list = []
       
        mu = self.env._init_mu(False)
        
        expert_demon = demonstration(self.hor,self.card_s,self.card_a,self.d_r,self.env,self.n_demon)
        self.expert_demon = expert_demon
        add_data = additional_dataset(self.hor,self.card_s,self.card_a,self.d_r,self.env,self.n_add)
        uq , P = self.init_construct(add_data)

        
        Q = np.zeros([self.hor, self.card_s, self.card_a])
        V = np.zeros([self.hor, self.card_s])
        pi = np.ones([self.hor, self.card_s, self.card_a])*(1./self.card_a)
        self.add_data = add_data
        if self.verbose:
            print(uq)
        for k in range(train_num):
            
            #policy_improve
            for h in range(self.hor):
                for s in range(self.card_s):
                    tmp = 0.0  # normalization constant
                    for a in range(self.card_a):
                        pi[h,s,a] *= np.exp(self.alpha * Q[h,s,a]/math.sqrt((k)+1))
                        tmp += pi[h,s,a]
                    for a in range(self.card_a):
                        pi[h,s,a] /= tmp
                         
            # policy_eval()
            Q = np.zeros([self.hor, self.card_s, self.card_a])
            V = np.zeros([self.hor, self.card_s])
            for h in reversed(range(self.hor)):
                for s in range(self.card_s):
                    for a in range(self.card_a):
                        PV = 0
                        if h < self.hor-1:
                            for s_ in range(self.card_s):
                                PV += P[h,s,a,s_]*V[h+1,s_]
                        Q[h,s,a] = np.clip(self.env._calculate_reward(mu,h,s,a)-self.kappa*uq[h,s,a]+PV,0,1000)
                        if (Q[h,s,a]>50):
                            print("Big Q",Q[h,s,a])
                        #Q[h,s,a] = np.clip(self.env._true_reward(h,s,a)-self.kappa*uq[h,s,a]+PV,0,100)
                        V[h,s] += pi[h,s,a]*Q[h,s,a]
            
            # update mu _ batch_grad

            mu_diff = expert_demon.eval_grad() -add_data.eval_grad_sim(P,pi,Q)
            mu = np.clip(mu + self.stepsize/math.sqrt(k//2+1)* mu_diff,0,self.mu_max)
  
            # Logger
            
            if k % 1 == 0:
                print(k,"-th return:",self.env._eval(pi))
            returns_list.append(self.env._eval(pi))
        
       
        return returns_list
    