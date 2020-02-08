import numpy as np
import scipy.stats as st






def Q(ks,n,p_k,p_n,lam):
	q= p_k * np.exp(-(p_n/lam) * (n-ks)**2.)
	q = (q+1e-10)/np.sum(q+1e-10)
	return q



def KL(p,q):
	return np.sum(q * (np.log2(q) - np.log2(p)), axis=1)


def compute_q_nk(ns, ks, p_n,p_k, lam, sm=1e-10):
    lam = lam.reshape((len(lam),1))
    p_n = p_n.reshape((len(p_n),1))

    q_nk = -((p_n * (ns-ks)**2.)/(lam)) 



    q_nk = np.exp(q_nk)
    q_nk = q_nk * p_k 
    
    q_nk = q_nk/ np.sum(q_nk, axis=1).reshape(len(q_nk), 1)

    
    q_nk += sm
    
    q_nk = (q_nk)/ np.sum(q_nk, axis=1).reshape(len(q_nk), 1)


    return q_nk


def find_q_nk(ns,ks,p_n,p_k,info, n_steps=1500):

    lams = np.ones_like(ns)*0.5
    q_nk = compute_q_nk( ns, ks, p_n,p_k, lams)

    ents = KL(p_k, q_nk)



    for i in range(n_steps):

        diffs = ents - info
        deltas = diffs *0.025

        lams = np.exp(np.log(lams) + deltas.reshape(len(deltas),1))


        q_nk = compute_q_nk(ns,ks,p_n,p_k,lams)
        ents = KL(p_k,q_nk)



    return q_nk





def P(x,a):
	p = 1./(x**a)

	return p/np.sum(p)



if __name__ == "__main__":


	step_n = 1
	step_k = 1
	a = 1

	min_n = 1
	max_n = 15
	min_k = step_k
	max_k = 100

	ks = np.arange(min_k,max_k,step_k)
	ns = np.arange(min_n,max_n+1,step_n)
	ns = ns.reshape((len(ns),1))

	p_ks = P(ks,float(a))
	p_ns = P(ns,float(a))

	
	f.close()
