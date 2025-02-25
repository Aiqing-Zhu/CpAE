import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .nn import DynNN
from .utils import timing, cross_entropy_loss

class ImageBrain:
    '''Runner based on torch.
    '''
    brain = None
    
    @classmethod
    def Init(cls, dataset, net, criterion, optimizer='adam', filename=None, lr=0.01, betas=(0.9, 0.999), lr_decay=1, iterations=1, batch_size=None, print_every=1000, save=False, minimum=False, callback=None, dtype='float', device='cpu', shuffle=True, num_workers=0):
        cls.brain = cls(dataset, net, criterion, optimizer, filename, lr, betas, lr_decay, iterations, batch_size, 
                         print_every, save, minimum, callback, dtype, device, shuffle, num_workers)
        
    @classmethod
    def Run(cls):
        cls.brain.run()
        
    @classmethod
    def Restore(cls):
        cls.brain.restore()
        
    @classmethod
    def Output(cls, dataset=True, best_model=True, loss_history=True, info=None, path=None, **kwargs):
        cls.brain.output(dataset, best_model, loss_history, info, path, **kwargs)
    
    @classmethod
    def Loss_history(cls):
        return cls.brain.loss_history
    
    @classmethod
    def Encounter_nan(cls):
        return cls.brain.encounter_nan
    
    @classmethod
    def Best_model(cls):
        return cls.brain.best_model
    
    def __init__(self, dataset, net, criterion, optimizer, filename, lr, betas, lr_decay, iterations, batch_size, 
                 print_every, save, minimum, callback, dtype, device, shuffle, num_workers):
        
        self.dataset = dataset
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.filename=filename if filename is not None else time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        self.lr = lr
        self.betas = betas
        self.lr_decay=lr_decay
        self.iterations = iterations
        self.batch_size = batch_size
        self.print_every = print_every
        self.save = save
        self.minimum=minimum
        self.callback = callback
        self.dtype = dtype
        self.device = device
        
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None
        
        self.__optimizer = None
        self.__criterion = None
        self.__train_dataloader = None
    
    @timing
    def run(self): 
        self.__init_brain()
        print('Training...', flush=True)
        loss_history = []
        if not os.path.isdir('./'+'training file/'+self.filename+'/model'): os.makedirs('./'+'training file/'+self.filename+'/model')
        t=time.time()
        for i in range(self.iterations + 2):
            if i!=self.iterations + 1:
                # t1=time.time()
                for inputs, targets in self.__train_dataloader:
                    inputs, targets = self.Preprocessing(inputs, targets)
                    loss = self.__criterion(self.net(inputs), targets)
                    regu = self.net.regularization(self.net(inputs), targets)
                    if i < self.iterations:
                        self.__optimizer.zero_grad()
                        (loss+regu).backward()
                        self.__optimizer.step()
                    if torch.any(torch.isnan(loss)):
                        self.encounter_nan = True
                        print('Encountering nan, stop training', flush=True)
                        return None              
                # print('optimization time:', time.time()-t)
                # t=time.time()


                # t2 =time.time()  
                if i % self.print_every == 0 or i == self.iterations:
                    loss_train, loss_test = self.accuracy()
                    loss_history.append([i, loss_train, loss_test])
                    print('{:<9}Train loss: {:<25}Test loss: {:<25}'.format(i, loss_train, loss_test), flush=True)
                    print('lr',self.__optimizer.param_groups[0]['lr'])

                    if self.save:
                        torch.save(self.net, 'training file/'+self.filename+'/model/model{}.pkl'.format(i))
                    if self.callback is not None: 
                        to_stop = self.callback(self.data, self.net)
                        if to_stop: break 
                # print('compute loss time:', time.time()-t2)
                # t2=time.time()

                loss_record = np.array(loss_history)
                np.savetxt('training file/'+self.filename+'/loss.txt', loss_record)            



                if self.lr_decay!=1:
                    self.__optimizer.param_groups[0]['lr']=self.__optimizer.param_groups[0]['lr']/self.lr_decay**(1/self.iterations)

                self.net.hyperparameter_update(i)
            else:
                self.loss_history = np.array(loss_history)
                print('Done!', flush=True)
        return self.loss_history
    
    def restore(self):
        if self.loss_history is not None and self.save == True:
            if self.minimum:
                best_loss_index = np.argmin(self.loss_history[:, 1])
            else:
                best_loss_index = -1
            iteration = int(self.loss_history[best_loss_index, 0])
            loss_train = self.loss_history[best_loss_index, 1]
            loss_test = self.loss_history[best_loss_index, 2]
            print('Best model at iteration {}:'.format(iteration), flush=True)
            print('Train loss:', loss_train, 'Test loss:', loss_test, flush=True)
            
            path = './outputs/' + self.filename
            if not os.path.isdir('./outputs/'+self.filename): os.makedirs('./outputs/'+self.filename)
            f = open(path +'/output.txt',mode='a')
            f.write('\n\n'
                    +'Train completion time  '
                    + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
                    + '\n'
                    + 'Best model at iteration: {}'.format(iteration)
                    + '\n'
                    + 'Train loss: %s'%(loss_train)
                    + '\n'
                    + 'Test loss: %s'%(loss_test)
                    )
            f.close()
            
            self.best_model = torch.load('training file/'+self.filename+'/model/model{}.pkl'.format(iteration))
        else:
            raise RuntimeError('restore before running or without saved models')
        return self.best_model
    
    def output(self, data, best_model, loss_history, info, path, **kwargs):
        if path is None:
            path = './outputs/' + self.filename
        if not os.path.isdir(path): os.makedirs(path)

        if best_model:
            torch.save(self.best_model, path + '/model_best.pkl')
        if loss_history:
            np.savetxt(path + '/loss.txt', self.loss_history)
        if info is not None:
            with open(path + '/info.txt', 'w') as f:
                for item in info:
                    f.write('{}: {}\n'.format(item[0], str(item[1])))
        for key, arg in kwargs.items():
            np.savetxt(path + '/' + key + '.txt', arg)
        
            
    def __init_brain(self):
        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None
        
        # self.data.device = self.device
        # self.data.dtype = self.dtype
        self.Device = "cpu" if self.device=='cpu' else 'cuda'
        self.__init_dataloader()
        
        self.net.device = self.device
        self.net.dtype = self.dtype
        self.net.init_auxi_modu()
        
        self.__init_optimizer()
        self.__init_criterion()
        

    def __init_optimizer(self):
        if self.optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas)
        else:
            raise NotImplementedError
            
    def __init_criterion(self):
        if isinstance(self.net, DynNN):
            self.__criterion = self.net.criterion
            self.__test_criterion = self.net.test_criterion
            if self.criterion is not None:
                raise Warning('loss-oriented neural network has already implemented its loss function')
        elif self.criterion == 'MSE':
            self.__criterion = torch.nn.MSELoss()
            self.__test_criterion = torch.nn.MSELoss()
        elif self.criterion == 'CrossEntropy':
            self.__criterion = cross_entropy_loss
            self.__test_criterion = cross_entropy_loss
        else:
            raise NotImplementedError   
    
    def __init_dataloader(self):
        self.__train_dataloader = DataLoader(self.dataset[0],
                                             batch_size=self.batch_size,     # 输出的batch size
                                             shuffle=self.shuffle,     # 随机输出
                                             num_workers=self.num_workers)  
        self.__test_dataloader = DataLoader(self.dataset[1],
                                            batch_size=self.batch_size,     # 输出的batch size
                                            shuffle=self.shuffle,     # 随机输出
                                            num_workers=self.num_workers)    
    
    def Preprocessing(self, inputs, targets):
        return (inputs/255.0).to(torch.float32).to(self.Device), (targets/255.0).to(torch.float32).to(self.Device)
        
    
    def accuracy(self, batch=False):
        with torch.no_grad():
            
            if batch:
                inputs, targets = next(iter(self.__train_dataloader))
                inputs, targets = self.Preprocessing(inputs, targets)
                total_train_correct = self.__criterion(self.net(inputs), targets)

                inputs, targets = next(iter(self.__test_dataloader))
                inputs, targets = self.Preprocessing(inputs, targets)
                total_test_correct = self.__test_criterion(self.net(inputs), targets)
                number_train, number_test = 1,1
            else:
                total_train_correct = 0
                for inputs, targets in self.__train_dataloader:
                    inputs, targets = self.Preprocessing(inputs, targets)
                    total_train_correct += self.__criterion(self.net(inputs), targets)
                number_train = len(self.__train_dataloader.dataset)/self.__train_dataloader.batch_size
                total_test_correct = 0
                for inputs, targets in self.__test_dataloader:
                    inputs, targets = self.Preprocessing(inputs, targets) 
                    total_test_correct += self.__test_criterion(self.net(inputs), targets)
                number_test = len(self.__test_dataloader.dataset)/self.__test_dataloader.batch_size 
        return ((total_train_correct / number_train).item(), 
                (total_test_correct / number_test).item() )