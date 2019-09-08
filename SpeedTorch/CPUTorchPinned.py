import torch
import os
import numpy as np

class _CPUPytorchCommon():
    def getTorchVar(self):
        return self.TorchFSet

    def saveTorch(self, saveFileName):
        torch.save(self.pytorchCPUVar, saveFileName)

    def getNumpyVersion(self):
        return self.pytorchCPUVar

    def _getReshapedRetrieval( self, retrievedPosIndexes , retrievedNegIndexes = None):
        if not retrievedNegIndexes is None:
            reshapedRetrieval =  np.concatenate( [ retrievedPosIndexes.reshape(-1) , retrievedNegIndexes.reshape(-1) ] )
        else:
            reshapedRetrieval = retrievedPosIndexes.reshape(-1)
        return reshapedRetrieval

class CPUPytorchModelFactory(_CPUPytorchCommon):

    def __init__(self, model_variable,  total_classes,  embed_dimension, datatype = torch.float ):
        self.model_variable = model_variable
        self.total_classes = total_classes
        self.embed_dimension = embed_dimension
        self.dtype = datatype

    def zerosInit(self):
        self.pytorchCPUVar = torch.zeros( size=(self.total_classes, self.embed_dimension), dtype=self.dtype, device = 'cpu', pin_memory = True)

    def customInit(self, initFunction, *args):
        self.pytorchCPUVar = torch.empty(size=(self.total_classes, self.embed_dimension), dtype=self.dtype, device='cpu' , pin_memory = True )
        initFunction( self.pytorchCPUVar, *args )

    def uniformDistributionInit(self, low, high):
        self.pytorchCPUVar = torch.empty(size=(self.total_classes, self.embed_dimension), dtype=self.dtype, device='cpu' ,  pin_memory = True)
        torch.nn.init.uniform_(self.pytorchCPUVar, a=low, b=high)
                
    def normalDistributionInit(self, mean, stdDev ):
        self.pytorchCPUVar = torch.empty(size=(self.total_classes, self.embed_dimension), dtype=self.dtype, device='cpu' ,  pin_memory = True)
        torch.nn.init.normal_(self.pytorchCPUVar, mean=mean, std=stdDev)
                
    def variableTransformer(self, batchSize, posPerBatch,  negPerBatch = None ):
        if not negPerBatch == None:
            return (np.arange( batchSize*posPerBatch ).reshape( batchSize , posPerBatch), 
                np.arange(start = batchSize*posPerBatch, 
                    stop = batchSize*posPerBatch + batchSize*negPerBatch ).reshape(batchSize, negPerBatch) )
        else:
            return np.arange( batchSize*posPerBatch ).reshape( batchSize, posPerBatch )

    def beforeForwardPass(self, retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        self.model_variable.weight.data = self.pytorchCPUVar[ reshapedRetrieval ].cuda()

    def afterOptimizerStep(self,retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        self.pytorchCPUVar[ reshapedRetrieval ] = self.model_variable.weight.data.detach().cpu().pin_memory() 
        
class CPUPytorchOptimizerFactory(_CPUPytorchCommon): #to do later, able to load matrixes to continue training
#take into account different size embedding matrices 

    def __init__(self, given_optimizer,  total_classes,  embed_dimension, model, variable_name, dtype=torch.float ):
        self.given_optimizer = given_optimizer
        self.total_classes = total_classes
        self.embed_dimension = embed_dimension
        self.model = model
        self.variable_name = variable_name
        self.dtype = dtype
        optimizer_index = None

        #Some optiizers do not initialize its state until after first step
        #So they need to initialized here
        for group in given_optimizer.param_groups:
            for p in group['params']:
                state = given_optimizer.state[p]
                # State initialization

                if given_optimizer.__str__().split(' ', 1)[0] == 'SparseAdam':
                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                    self.optVarList = [ 'exp_avg', 'exp_avg_sq']
                elif given_optimizer.__str__().split(' ', 1)[0] == 'Adagrad':
                    self.optVarList = [ 'sum' ]
                elif given_optimizer.__str__().split(' ', 1)[0] == 'Adadelta':
                    if len(state) == 0:
                        state['step'] = 0
                        state['square_avg'] = torch.zeros_like(p.data)
                        state['acc_delta'] = torch.zeros_like(p.data)
                    self.optVarList = [ 'square_avg', 'acc_delta']
                elif given_optimizer.__str__().split(' ', 1)[0] == 'Adamax':
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_inf'] = torch.zeros_like(p.data)
                    self.optVarList = [ 'exp_avg', 'exp_inf']
                elif given_optimizer.__str__().split(' ', 1)[0] == 'RMSprop':
                    if len(state) == 0:
                        state['step'] = 0
                        state['square_avg'] = torch.zeros_like(p.data)
                        if group['momentum'] > 0:
                            state['momentum_buffer'] = torch.zeros_like(p.data)
                        if group['centered']:
                            state['grad_avg'] = torch.zeros_like(p.data)
                    self.optVarList = [ 'square_avg']
                    if group['momentum'] > 0:
                         self.optVarList.append( 'momentum_buffer' )
                    if group['centered']:
                        self.optVarList.append( 'grad_avg' )
                elif given_optimizer.__str__().split(' ', 1)[0] == 'Rprop':
                    if p.grad is None:
                        print('Error, gradients are empty')
                        print('For Rprop, need to first run at least 1 training step that has gradients')
                        return
                    if len(state) == 0:
                        state['step'] = 0
                        state['prev'] = torch.zeros_like(p.data)
                        #For now, do now know how to Not initialize this due to len(state)==0 in optimizer
                        state['step_size'] = grad.new().resize_as_(grad).fill_(group['lr'])
                    self.optVarList = [ 'prev']
                elif given_optimizer.__str__().split(' ', 1)[0] == 'ASGD': 
                    if len(state) == 0:
                        state['step'] = 0
                        state['eta'] = group['lr']
                        state['mu'] = 1
                        state['ax'] = torch.zeros_like(p.data)
                        self.optVarList = [ 'ax']
                elif given_optimizer.__str__().split(' ', 1)[0] == 'AdamW': 
                    amsgrad = group['amsgrad']
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        if amsgrad:
                            state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    self.optVarList = [ 'exp_avg', 'exp_avg_sq']
                    if amsgrad:
                        self.optVarList.append('max_exp_avg_sq')
                elif given_optimizer.__str__().split(' ', 1)[0] == 'Adam': 
                    amsgrad = group['amsgrad']
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        if amsgrad:
                            state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    self.optVarList = [ 'exp_avg', 'exp_avg_sq']
                    if amsgrad:
                        self.optVarList.append( 'max_exp_avg_sq' )
                else:
                    print('This optimizer is not currently supported. Please choose a different optimizer')
                    return

        #Figure out which index for given variable 
        for i, item in enumerate( self.model.named_parameters() ):
            if item[0][:-7] == self.variable_name:
                optimizer_index = i
        if optimizer_index == None:
            print( 'Error: No variable with that name is in Model. Please initialize again with correct name' ) 
            return

        optimizerKeyList = list(self.given_optimizer.state_dict()['state'].keys())
        self.optimizerKey = optimizerKeyList[ optimizer_index ]

    def optInit(self):
        self.pytorchCPUVar = []
        for optVar in self.optVarList:
            self.pytorchCPUVar.append( torch.zeros( size=(self.total_classes, self.embed_dimension), dtype=self.dtype, device = 'cpu' , pin_memory = True) )
            
    def beforeForwardPass(self, retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        for idx, optVar in enumerate(self.optVarList):
            self.given_optimizer.state_dict()['state'][ self.optimizerKey ][optVar] = self.pytorchCPUVar[idx][ reshapedRetrieval ].cuda()

    def afterOptimizerStep(self, retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        for idx, optVar in enumerate(self.optVarList):
            self.pytorchCPUVar[idx][ reshapedRetrieval ] =  self.given_optimizer.state_dict()['state'][ self.optimizerKey ][optVar].detach().cpu().pin_memory()  
            
class CPUPytorchCOM(_CPUPytorchCommon):

    def __init__(self, total_classes, datatype = torch.int32 ):
        self.total_classes = total_classes
        self.dtype = datatype

    def comInit(self):
        self.pytorchCPUVar = torch.zeros( size=(self.total_classes, self.embed_dimension), dtype=self.dtype, device = 'cpu' , pin_memory = True )

