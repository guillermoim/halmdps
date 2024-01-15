
import numpy as np

class LearningRateScheduler:

    def __init__(self, schedule1=200, schedule2=300, schedule3=400, factor1=.99, factor2=.9, factor3=.75, init_lr1=1, init_lr2=1, init_lr3=1,):

        self.lr1 = init_lr1
        self.lr2 = init_lr2
        self.lr3 = init_lr3

        self.schedule1 = schedule1
        self.schedule2 = schedule2
        self.schedule3 = schedule3

        self.factor1 = factor1
        self.factor2 = factor2
        self.factor3 = factor3

        self.samples = 0

    def update(self):

        self.samples += 1
        
        if self.samples % self.schedule1 == 0:
            self.lr1 *= self.factor1 
        if self.samples % self.schedule2 == 0:
            self.lr2 *= self.factor2
        if self.samples % self.schedule3 == 0:
            self.lr3 *= self.factor3 

    def get_learning_rates(self):
        return self.lr1, self.lr2, self.lr3
