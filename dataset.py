import numpy as np
class dataset():
	def __init__(self,hor, card_s, card_a,d_r,n):
		self.hor = hor
		self.card_s = card_s
		self.card_a = card_a
		self.traj = None
		self.n = n
		self.collect()
	def eval_grad(self):
		raise ('Not Implemented')
	def collect(self):
		raise ('Not Implemented')
class demonstration(dataset):
	def __init__(self,hor, card_s, card_a,d_r,env,n):
		self.hor = hor
		self.card_s = card_s
		self.card_a = card_a
		self.n = n
		self.env= env
		self.collect()
	def collect(self):
		expert = self.env.expert
		self.traj = self.env.sample(expert,self.n)
	def eval_grad(self):
		grad_traj = []
		for traj in self.traj:
			grad_traj_temp = np.zeros_like(self.env.r_map[:,0,0,0])
			h = 0
			for (s, a, s_next) in traj:
				grad_traj_temp += self.env.r_map[:,h,s,a]
				h += 1
			
		return grad_traj_temp/self.n

class additional_dataset(dataset):
	def __init__(self,hor, card_s, card_a,d_r,env,n):
		self.hor = hor
		self.card_s = card_s
		self.card_a = card_a
		self.n = n
		self.env= env
		self.collect()
	def collect(self):
		self.traj = self.env.sample(None,self.n)
	def mix(self,data):
		self.traj.extend(data.traj)
		return self
	def eval_grad(self):
		grad_traj = []
		for traj in self.traj:
			grad_traj_temp = np.zeros_like(self.env.r_map[:,0,0,0])
			h = 0
			for (s, a, s_next) in traj:
				grad_traj_temp += self.env.r_map[:,h,s,a]
				h += 1
		return grad_traj_temp/self.n
	def eval_grad_sim(self,p,pi,Q):
		prob_s = np.zeros([self.card_s])
		prob_s[self.env.init_s] = 1.0
		grad_traj_temp  = np.zeros_like(self.env.r_map[:,0,0,0])
		for h in range(self.hor):
			prob_s_next = np.zeros([self.card_s])
			for s in range(self.card_s):
				for a in range(self.card_a):
					for s_next in range(self.card_s):
						prob_s_next[s_next] += prob_s[s] * pi[h, s, a] * p[h, s, a, s_next]
					grad_traj_temp += self.env.r_map[:,h,s,a]*int(Q[h,s,a]>0)
			prob_s = prob_s_next
		return grad_traj_temp /self.n