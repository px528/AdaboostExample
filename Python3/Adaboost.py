

import numpy as np 
from WeakClassifier import *

class AdaboostClassifier:

	#calculate new Weight
	def cal_W(self,W,alpha,y,pred):
		ret=0
		new_W=[]
		for i in range(len(y)):
			new_W.append(W[i]*np.exp(-alpha*y[i]*pred[i]))
		return np.array(new_W/sum(new_W)).reshape([len(y),1])

	#calculate error rate per iteration
	def cal_e(self,y,pred,W):
		ret=0
		for i in range(len(y)):
			if y[i]!=pred[i]:
				ret+=W[i]
		return ret

	#calculate alpha
	def cal_alpha(self,e):
		if e==0:
			return 10000
		elif e==0.5:
			return 0.001
		else:
			return 0.5*np.log((1-e)/e)

	#calculate final predict value
	def cal_final_pred(self,i,alpha,weak,y):
		ret=np.array([0.0]*len(y))
		for j in range(i+1):
			ret+=alpha[j]*weak[j].pred
		return np.sign(ret)

	#calculate final error rate
	def cal_final_e(self,y,cal_final_predict):	
		ret=0
		for i in range(len(y)):
			if y[i]!=cal_final_predict[i]:
				ret+=1
		return ret/len(y)

	#train
	def fit(self,X,y,M=15):
		W={}
		self.weak={}
		alpha={}
		pred={}

		for i in range(M):
			W.setdefault(i)
			self.weak.setdefault(i)
			alpha.setdefault(i)
			pred.setdefault(i)

		#per iteration (all:M times)
		for i in range(M):
			#for the first iteration,initial W
			if i == 0:
				W[i]=np.array([1]*len(y))/len(y)
				W[i]=W[i].reshape([len(y),1])
			#if not the first iteration,calculate new Weight
			else:
				W[i]=self.cal_W(W[i-1],alpha[i-1],y,pred[i-1])

			#using train weak learner and get this learner predict value
			self.weak[i]=WeakClassifier()
			self.weak[i].fit(X,y,W[i])
			pred[i]=self.weak[i].pred

			#calculate error rate this iteration
			e=self.cal_e(y,pred[i],W[i])
			#calculate alpha this iteration
			alpha[i]=self.cal_alpha(e)
			#calculate the final predict value
			cal_final_predict=self.cal_final_pred(i,alpha,self.weak,y)

			print('iteration:%d'%(i+1))
			print('self.decision_key=%s'%(self.weak[i].decision_key))
			print('self.decision_feature=%d'%(self.weak[i].decision_feature))
			print('decision_threshold=%f'%(self.weak[i].decision_threshold))
			print('W=%s'%(W[i]))
			print('pred=%s'%(pred[i]))
			print('e:%f alpha:%f'%(e,alpha[i]))
			print('cal_final_predict:%s'%(cal_final_predict))
			print('cal_final_e:%s%%'%(self.cal_final_e(y,cal_final_predict)*100))
			print('')

			#calculate the final error rate,if it is zero,stop iteration.
			if self.cal_final_e(y,cal_final_predict)==0 or e==0:
				break
		#return the iteration times,from 1 on.
		return i+1

