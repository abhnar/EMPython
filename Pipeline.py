# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:41:54 2020

@author: Abhijith
"""

class Pipeline:
    vals=[None] * 50
    size=50
    pos=0
    
    def push(self,a):
        self.vals[self.pos]=a
        self.pos+=1
        if(self.pos==self.size):
            self.pos=0
        #print(self.vals)
    def get(self,ix):
        ix=self.pos-ix
        if(ix<0):
            ix=ix+self.size
        return self.vals[ix]
    def print(self):
        print(self.vals)
        
p=Pipeline()
for i in range(15):
    p.push(i)
    
