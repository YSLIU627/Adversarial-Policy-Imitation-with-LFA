import numpy as np
from env_old import GridWorld

class Env():
	def __init__(self, hor, card_s, card_a,d_r,expert_sim_time,noise_factor=1.0,n=3,mu_init_factor = 0):
		if n != 3:
			hor = n**2
			card_s = n**2
		self.hor = hor
		self.card_s = card_s
		self.card_a = card_a
		self.init_s = 0
		self.p = np.abs(np.random.rand(hor, card_s, card_a, card_s))
		self.subenv = GridWorld(n)
		self.p = np.zeros([self.hor, self.card_s, self.card_a, self.card_s])
		self.d_r = d_r
		self.mu_init_factor =mu_init_factor
		assert card_a == 5 and card_s == n**2 and hor == n**2
		# Generate kernels
		for h in range(self.hor):
			for s in range(self.card_s):
				for a in self.subenv.action_space:
					for ns in range(self.card_s):
					
						self.p[h,s,a,ns] = self.subenv._get_transition(self.subenv.state_space[s],a,self.subenv.state_space[ns])
						
						if self.p[h,s,a,ns] == 1:
							self.p[h,s,a,ns] = 0.9
						else:
							self.p[h,s,a,s] = 0.1
	
					normalize = np.sum(self.p[h,s,a,:])
					for ns in range(self.card_s):
						self.p[h,s,a,ns] /= normalize
					
					assert np.sum(self.p[h,s,a,:]) ==1.0
		self.r_map = np.zeros((self.d_r,self.hor,self.card_s,self.card_a))
		# Generate reward functions
		for h in range(self.hor):
			for s in range(self.card_s):
				for a in self.subenv.action_space:
					index = (s+1)*(a+1)-1
					self.r_map[index,h,s,a] = 1
					for _ in range(self.d_r):
						self.r_map[_,h,s,a] += 0.01/self.d_r
					self.r_map[:,h,s,a] /= np.sum(self.r_map[:,h,s,a])
					
					
		self.mu_true = self._init_mu(True)

		self.train_num_expert = expert_sim_time
		self.expert = self._generate_expert()
	def _init_mu(self,true_model = False):
		mu = np.zeros(self.d_r)
		if true_model:
			factor = 1.0
		else:
			factor = self.mu_init_factor
		for s in range(self.card_s):
			for a in self.subenv.action_space:
				index = (s+1)*(a+1)-1
				mu[index] = factor*self.subenv._get_reward(self.subenv.state_space[s],a)
		return mu
	def _calculate_reward(self,r_mu,h,s,a):
		return np.dot(self.r_map[:,h,s,a],r_mu)
		
	def _true_reward(self,h,s,a):
		return self._calculate_reward(self.mu_true,h,s,a)
		
	def _eval(self, pi=None):
		if pi is None:
			# assume a uniform policy
			pi = np.ones([self.hor, self.card_s, self.card_a])*(1./self.card_a)
		prob_s = np.zeros([self.card_s])
		prob_s[self.init_s] = 1.0
		returns = 0
		for h in range(self.hor):
			prob_s_next = np.zeros([self.card_s])
			for s in range(self.card_s):
				for a in range(self.card_a):
					for s_next in range(self.card_s):
						prob_s_next[s_next] += prob_s[s] * pi[h, s, a] * self.p[h, s, a, s_next]
					returns += prob_s[s] * pi[h, s, a]  *self._true_reward(h,s,a)
			prob_s = prob_s_next
		return returns
	
	def sample(self, pi ,num_data):
		if pi is None:
			# assume a uniform policy
			pi = np.ones([self.hor, self.card_s, self.card_a])*(1./self.card_a)
		dataset = []
		for _ in range(num_data):
			s = self.init_s
			traj = []
			for h in range(self.hor):
				a = np.random.choice(range(self.card_a),
                                              p=pi[h,s,:])
				s_next = np.random.choice(np.arange(self.card_s), p=self.p[h, s,a].reshape(-1))
				traj.append((s, a, s_next))
				s = s_next
				dataset.append(traj)
		
		return dataset
	

	def _generate_expert(self):
		alpha = 0.5
		P = self.p
		Q = np.zeros([self.hor, self.card_s, self.card_a])
		V = np.zeros([self.hor, self.card_s])
		pi = np.ones([self.hor, self.card_s, self.card_a])*(1./self.card_a)
		for k in range(self.train_num_expert):
			
			
            #policy_improve()
			for h in range(self.hor):
				
				for s in range(self.card_s):
					tmp = 0.0  # normalization constant
					for a in range(self.card_a):
						pi[h,s,a] *= np.exp(alpha * Q[h,s,a])
						tmp += pi[h,s,a]
					for a in range(self.card_a):
						pi[h,s,a] /= tmp      
						
            # policy_eval()
			Q = np.zeros([self.hor, self.card_s, self.card_a])
			V = np.zeros([self.hor, self.card_s])
			for h in reversed(range(self.hor)):
				
				for s in range(self.card_s):
					tmp = 0.0  # normalization constant
					for a in range(self.card_a):
						PV = 0
						if h < self.hor-1:
							for s_ in range(self.card_s):
								PV += P[h,s,a,s_]*V[h+1,s_]
						Q[h,s,a] = np.clip(self._true_reward(h,s,a)+PV,0,100)
						V[h,s] += pi[h,s,a]*Q[h,s,a]

	
		print("expert performance",self._eval(pi))

		return pi 	
