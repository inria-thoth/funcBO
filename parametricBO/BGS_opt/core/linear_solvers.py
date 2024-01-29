import torch 
import numpy as np

from utils.helpers import get_gpu_usage


class LinearSolverAlg(object):
	def __call__(self):
		raise NotImplementedError
class GD(LinearSolverAlg):
	## performs gd/sgd on quadratic loss 0.5 xAx+bx 
	def __init__(self,lr=0.1,n_iter=1):
		super(GD,self).__init__()
		self.n_iter= n_iter
		self.lr= lr
	def __call__(self,linear_op,b_vector,init,compute_latest=False):
		out_lower = init
		for i in range(self.n_iter):
			out_upper, update = linear_op(out_lower)
			out_lower = tuple([ x - self.lr*(ax+b) if ax is not None else x - self.lr*b for x,ax,b in zip(out_lower,update,b_vector)])
		if compute_latest:
			out_upper,_ = linear_op(out_lower,retain_graph=False, which='upper')
		return out_upper,out_lower

class Normal_GD(LinearSolverAlg):
	## performs gd/sgd on normal loss 0.5 || Ax+b||^2
	def __init__(self,lr=0.1,n_iter=1):
		super(Normal_GD,self).__init__()
		self.n_iter= n_iter
		self.lr= lr
	def __call__(self,linear_ p,b_vector,init,compute_latest=False):
		out_lower = init
		if linear_op.stochastic:
			retain_graph = False
		else:
			retain_graph = True
		for i in range(self.n_iter):
			out_upper, update = linear_op(out_lower,retain_graph=retain_graph)
			
			update = tuple([ax+b if ax is not None else b for ax,b in zip(update,b_vector)])
			if i == self.n_iter-1 and not compute_latest:
				retain_graph = False
			out_upper, update = linear_op(update,retain_graph=retain_graph)
			out_lower = tuple([ x - self.lr*ax if ax is not None else x  for x,ax in zip(out_lower,update)])
		if compute_latest:
			out_upper,_ = linear_op(out_lower,retain_graph=False, which='upper')

		return out_upper,out_lower



class CG(LinearSolverAlg):
	## performs gd/sgd on quadratic loss 0.5 xAx+bx 
	def __init__(self,n_iter=1, epsilon=1.0e-5):
		super(CG,self).__init__()
		self.n_iter= n_iter
		self.epsilon = epsilon
	def __call__(self,linear_op,b_vector,init,compute_latest=False):
		
		##### reverse 
		b_vector = tuple([-b for b in b_vector])
		if linear_op.stochastic:
			retain_graph = False
		else:
			retain_graph = True

		def Ax(x):
			_, update = linear_op(x,retain_graph=retain_graph)
			return update
		x_last, _ = cg(Ax,b_vector, init,max_iter=self.n_iter, epsilon=self.epsilon)
		x_last = tuple(x_last)
		out_upper,_ = linear_op(x_last,retain_graph=False, which='upper')
		return out_upper,x_last



class Identity(LinearSolverAlg):
	def __init__(self,n_iter=1, epsilon=1.0e-5):
		super(Identity,self).__init__()
		self.n_iter= n_iter
		self.epsilon = epsilon
	def __call__(self,linear_op,b_vector,init,compute_latest=False):
		
		##### reverse 
		b_vector = tuple([-b for b in b_vector])

		out_upper,_ = linear_op(b_vector,retain_graph=False, which='upper')
		return out_upper,b_vector


# adapted from https://github.com/JunjieYang97/stocBiO/blob/master/Hyperparameter-optimization/hypergrad/CG_torch.py
def cg(Ax, b, x, max_iter=100, epsilon=1.0e-5):
		""" Conjugate Gradient
			Args:
				Ax: function, takes list of tensors as input
				b: list of tensors
			Returns:
				x_star: list of tensors
		"""

		x_last = x
		init_Ax = Ax(x_last)
		r_last = tuple([bb-ax for bb,ax in zip(b,init_Ax)])
		p_last = tuple([torch.zeros_like(rr).copy_(rr) for rr in r_last])
		counter_hess = 1
		for ii in range(max_iter):
			Ap = Ax(p_last)
			counter_hess +=1
			Ap_vec = cat_list_to_tensor(Ap)
			p_last_vec = cat_list_to_tensor(p_last)
			r_last_vec = cat_list_to_tensor(r_last)
			rTr = torch.sum(r_last_vec * r_last_vec)
			pAp = torch.sum(p_last_vec * Ap_vec)
			alpha = rTr / pAp

			x = tuple([xx + alpha * pp for xx, pp in zip(x_last, p_last)])
			r = tuple([rr - alpha * pp for rr, pp in zip(r_last, Ap)])
			r_vec = cat_list_to_tensor(r)

			if float(torch.norm(r_vec)) < epsilon:
				break

			beta = torch.sum(r_vec * r_vec) / rTr
			p = [rr + beta * pp for rr, pp in zip(r, p_last)]

			x_last = x
			p_last = p
			r_last = r

		return x_last,counter_hess


def cat_list_to_tensor(list_tx):
		return torch.cat([xx.view([-1]) for xx in list_tx])


# class MinresQLP(LinearSolverAlg):
# 	def __init__(self,n_iter=1):
# 		super(MinresQLP,self).__init__()

# 		self.z_0 = None
# 		self.z_1 = None
# 		self.q_1 = None

# 		self.w_0 = 0.
# 		self.w_prev = 0.
# 		self.phi_0 = None
# 		self.beta_1= None

# 		self.x_0 = None
# 		self.x_prev = None
# 		self.x_prev_2 = None

# 		self.c_1 = -1
# 		self.c_2 = -1
# 		self.c_3 = -1

# 		self.s_1 = 0.
# 		self.s_2 = 0.
# 		self.s_3 = 0.


# 		self.tau = 0.
# 		self.omega = 0.
# 		self.xi_prev_2 = 0.
# 		self.xi_prev = 0.
# 		self.xi_0 = 0.

# 		self.kappa_0 = 1.

# 		self.alpha_0 = 0.
# 		self.delta = 0.
# 		self.gamma_prev = 0.
# 		self.gamma_0 = 0.

# 		self.eta_prev = 0.
# 		self.eta = 0.
# 		self.eta_1 = 0.

# 		self.v_prev = 0.
# 		self.v_0 = 0.
# 		self.v_1 = 0.

# 		self.mu_prev = 0.
# 		self.mu_0 = 0.

# 	def __call__(self,res_op,init):
# 		sol = init
# 		## initialize None variables


# 		for i in range(self.n_iter):
						
# 			self.p_1 = res_op(self.q_1) 
# 			alpha_1 = torch.sum(torch.cat([torch.einsum('i,i->',self.q_1,self.p_1)],axis=0))/self.beta_1**2

# 			if self.beta_0 is None:
# 				self.z_1_new = [(p- (alpha_k)* z)/self.beta_1 for p,z in zip(self.p_1,self.z_1)]
# 			else:
# 				self.z_1_new = [(p- (alpha_k)* z)/self.beta_1-self.beta_1/self.beta_0*z_p for p,z,z_p in zip(self.p_1,self.z_1,self.z_0)]

# 			self.z_0 = self.z_1
# 			self.z_1 = self.z_1_new

# 			self.beta_0 = self.beta_1
# 			self.beta_1 = norm(self.z_1)

# 			##### Finish this based on https://arxiv.org/pdf/1003.4042.pdf




# 		return sol,out
		



def dot(a,b):
	return torch.sum(torch.cat([torch.einsum('i,i->',u,v) for u,v in zip(a,b)],axis=0))

def norm(a):
	return torch.norm(torch.cat([torch.norm(u) for u in a ],axis=0))






# 	def __call__(self,res_op,init):
# 		sol = init
# 		for i in range(self.n_iter):
# 			vhp,out = res_op(sol)
# 			sol = [ ag - self.lr*g if g is not None else 1.*ag for ag,g in zip(sol,vhp)]
# 		return sol,out


# def MinresQLP(A, b, rtol, maxit, shift=None, maxxnorm=None,
#               Acondlim=None, TranCond=None):
    
#     #A = aslinearoperator(A)
#     if shift is None:
#         shift = 0
#     if maxxnorm is None:
#         maxxnorm = 1e7
#     if Acondlim is None:
#         Acondlim = 1e15
#     if TranCond is None:
#         TranCond = 1e7
        
    
#     #n = len(b) 
#     #b = b.reshape(n,1)
#     r2 = b
#     r3 = r2
#     beta1 = torch.norm(torch.cat([torch.norm(_r) for _r in r2],axis=0)) 
#     noprecon = True

#     ## Initialize
#     flag0 = -2
#     flag = -2
#     iters = 0
#     QLPiter = 0
#     beta = 0.
#     tau = 0.
#     taul = 0.
#     phi = beta1
#     betan = beta1
#     gmin = 0.
#     cs = -1.
#     sn = 0.
#     cr1 = -1.
#     sr1 = 0.
#     cr2 = -1.
#     sr2 = 0.
#     dltan = 0.
#     eplnn = 0.
#     gama = 0.
#     gamal = 0.
#     gamal2 = 0.
#     eta = 0.
#     etal = 0.
#     etal2 = 0.
#     vepln = 0.
#     veplnl = 0.
#     veplnl2 = 0.
#     ul3 = 0.
#     ul2 = 0.
#     ul = 0.
#     u = 0.
#     rnorm = betan
#     xnorm = 0.
#     xl2norm = 0.
#     Axnorm = 0.
#     Anorm = 0.
#     Acond = 1.
#     relres = rnorm / (beta1 + 1e-50)
#     x = torch.zeros_like(b)
#     w = torch.zeros_like(b)
#     wl = torch.zeros_like(b)
        
#     #b = 0 --> x = 0 skip the main loop
#     if beta1 == 0:
#         flag = 0
    
#     while flag == flag0 and iters < maxit:
#         #lanczos
#         iters += 1
#         betal = beta
#         beta = betan
#         v = r3/beta
#         r3 = res_op(v)
#         if shift == 0:
#             pass
#         else:
#             r3 = r3 - shift*v
        
#         if iters > 1:
#             r3 = r3 - r1*beta/betal
        
#         alfa = torch.real(r3.T.dot(v))
#         r3 = r3 - r2*alfa/beta
#         r1 = r2
#         r2 = r3
        

#         betan = torch.norm(r3)
#         if iters == 1:
#             if betan == 0:
#                 if alfa == 0:
#                     flag = 0
#                     break
#                 else:
#                     flag = -1
#                     x = b/alfa
#                     break
#         pnorm = torch.sqrt(betal ** 2 + alfa ** 2 + betan ** 2)
        
#         #previous left rotation Q_{k-1}
#         dbar = dltan
#         dlta = cs*dbar + sn*alfa
#         epln = eplnn
#         gbar = sn*dbar - cs*alfa
#         eplnn = sn*betan
#         dltan = -cs*betan
#         dlta_QLP = dlta
#         #current left plane rotation Q_k
#         gamal3 = gamal2
#         gamal2 = gamal
#         gamal = gama
#         cs, sn, gama = SymGivens(gbar, betan)
#         gama_tmp = gama
#         taul2 = taul
#         taul = tau
#         tau = cs*phi
#         Axnorm = torch.sqrt(Axnorm ** 2 + tau ** 2)
#         phi = sn*phi
#         #previous right plane rotation P_{k-2,k}
#         if iters > 2:
#             veplnl2 = veplnl
#             etal2 = etal
#             etal = eta
#             dlta_tmp = sr2*vepln - cr2*dlta
#             veplnl = cr2*vepln + sr2*dlta
#             dlta = dlta_tmp
#             eta = sr2*gama
#             gama = -cr2 *gama
#         #current right plane rotation P{k-1,k}
#         if iters > 1:
#             cr1, sr1, gamal = SymGivens(gamal, dlta)
#             vepln = sr1*gama
#             gama = -cr1*gama
        
#         #update xnorm
#         xnorml = xnorm
#         ul4 = ul3
#         ul3 = ul2
#         if iters > 2:
#             ul2 = (taul2 - etal2*ul4 - veplnl2*ul3)/gamal2
#         if iters > 1:
#             ul = (taul - etal*ul3 - veplnl *ul2)/gamal
#         xnorm_tmp = torch.sqrt(xl2norm**2 + ul2**2 + ul**2)
#         if torch.abs(gama) > np.finfo(np.double).tiny and xnorm_tmp < maxxnorm:
#             u = (tau - eta*ul2 - vepln*ul)/gama
#             if torch.sqrt(xnorm_tmp**2 + u**2) > maxxnorm:
#                 u = 0
#                 flag = 6
#         else:
#             u = 0
#             flag = 9
#         xl2norm = torch.sqrt(xl2norm**2 + ul2**2)
#         xnorm = torch.sqrt(xl2norm**2 + ul**2 + u**2)
#         #update w&x
#         #Minres
#         if (Acond < TranCond) and flag != flag0 and QLPiter == 0:
#             wl2 = wl
#             wl = w
#             w = (v - epln*wl2 - dlta_QLP*wl)/gama_tmp
#             if xnorm < maxxnorm:
#                 x += tau*w
#             else:
#                 flag = 6
#         #Minres-QLP
#         else:
#             QLPiter += 1
#             if QLPiter == 1:
#                 xl2 = torch.zeros_like(b)
#                 if (iters > 1):  # construct w_{k-3}, w_{k-2}, w_{k-1}
#                     if iters > 3:
#                         wl2 = gamal3*wl2 + veplnl2*wl + etal*w
#                     if iters > 2:
#                         wl = gamal_QLP*wl + vepln_QLP*w
#                     w = gama_QLP*w
#                     xl2 = x - wl*ul_QLP - w*u_QLP
                    
#             if iters == 1:
#                 wl2 = wl
#                 wl = v*sr1
#                 w = -v*cr1                
#             elif iters == 2:
#                 wl2 = wl
#                 wl = w*cr1 + v*sr1
#                 w = w*sr1 - v*cr1
#             else:
#                 wl2 = wl
#                 wl = w
#                 w = wl2*sr2 - v*cr2
#                 wl2 = wl2*cr2 +v*sr2
#                 v = wl*cr1 + w*sr1
#                 w = wl*sr1 - w*cr1
#                 wl = v
#             xl2 = xl2 + wl2*ul2
#             x = xl2 + wl*ul + w*u         

#         #next right plane rotation P{k-1,k+1}
#         gamal_tmp = gamal
#         cr2, sr2, gamal = SymGivens(gamal, eplnn)
#         #transfering from Minres to Minres-QLP
#         gamal_QLP = gamal_tmp
#         #print('gamal_QLP=', gamal_QLP)
#         vepln_QLP = vepln
#         gama_QLP = gama
#         ul_QLP = ul
#         u_QLP = u
#         ## Estimate various norms
#         abs_gama = torch.abs(gama)
#         Anorml = Anorm
#         Anorm = torch.max(torch.cat([Anorm, pnorm, gamal, abs_gama]),axis=0)
#         if iters == 1:
#             gmin = gama
#             gminl = gmin
#         elif iters > 1:
#             gminl2 = gminl
#             gminl = gmin
#             gmin = torch.min(torch.cat([gminl2, gamal, abs_gama],axis=0))
#         Acondl = Acond
#         Acond = Anorm / gmin
#         rnorml = rnorm
#         relresl = relres
#         if flag != 9:
#             rnorm = phi
#         relres = rnorm / (Anorm * xnorm + beta1)
#         rootl = torch.sqrt(gbar ** 2 + dltan ** 2)
#         Arnorml = rnorml * rootl
#         relAresl = rootl / Anorm
#         ## See if any of the stopping criteria are satisfied.
#         epsx = Anorm * xnorm * np.finfo(float).eps
#         if (flag == flag0) or (flag == 9):
#             t1 = 1 + relres
#             t2 = 1 + relAresl
#             if iters >= maxit:
#                 flag = 8 #exit before maxit
#             if Acond >= Acondlim:
#                 flag = 7 #Huge Acond
#             if xnorm >= maxxnorm:
#                 flag = 6 #xnorm exceeded
#             if epsx >= beta1:
#                 flag = 5 #x = eigenvector
#             if t2 <= 1:
#                 flag = 4 #Accurate Least Square Solution
#             if t1 <= 1:
#                 flag = 3 #Accurate Ax = b Solution
#             if relAresl <= rtol:
#                 flag = 2 #Trustful Least Square Solution
#             if relres <= rtol:
#                 flag = 1 #Trustful Ax = b Solution
#         if flag == 2 or flag == 4 or flag == 6 or flag == 7:
#             #possibly singular
#             iters = iters - 1
#             Acond = Acondl
#             rnorm = rnorml
#             relres = relresl
                              
    
#     return x

