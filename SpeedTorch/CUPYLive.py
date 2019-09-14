
import numpy as np
import cupy
import torch
import os

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

class PMemory(cupy.cuda.memory.BaseMemory):
    def __init__(self, size):
        self.size = size
        self.device_id = cupy.cuda.device.get_device_id()
        self.ptr = 0
        if size > 0:
            self.ptr = cupy.cuda.runtime.hostAlloc(size, 0)
    def __del__(self):
        if self.ptr:
            cupy.cuda.runtime.freeHost(self.ptr)

def my_pinned_allocator(bsize):
    return cupy.cuda.memory.MemoryPointer(PMemory(bsize),0)

class _Common():

    def getCupyVar(self):
        return self.CUPYcorpus

    def saveCupy(self, saveFileName):
        cupy.save( saveFileName, self.CUPYcorpus)

    def getNumpyVersion(self):
        return cupy.asnumpy(self.CUPYcorpus)

    def _getReshapedRetrieval( self, retrievedPosIndexes , retrievedNegIndexes = None):
        if not retrievedNegIndexes is None:
            reshapedRetrieval =  np.concatenate( [ retrievedPosIndexes.reshape(-1) , retrievedNegIndexes.reshape(-1) ] )
        else:
            reshapedRetrieval = retrievedPosIndexes.reshape(-1)
        return reshapedRetrieval

class ModelFactory(_Common):

    def __init__(self, model_variable,  total_classes,  embed_dimension, datatype = 'float32', CPUPinn = False):
        self.model_variable = model_variable
        self.total_classes = total_classes
        self.embed_dimension = embed_dimension
        self.datatype = datatype
        self.CPUPinn = CPUPinn

    def zerosInit(self):
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)
        self.CUPYcorpus = cupy.zeros(shape=(self.total_classes, self.embed_dimension), dtype=self.datatype)
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)

    def uniformDistributionInit(self, low, high):
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)
        self.CUPYcorpus = cupy.random.uniform(low=low, high=high, size=(self.total_classes, self.embed_dimension), dtype=self.datatype )
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)
                
    def normalDistributionInit(self, mean, stdDev):
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)
        self.CUPYcorpus = cupy.random.normal(loc=mean, scale=stdDev, size=(self.total_classes, self.embed_dimension), dtype=self.datatype )
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)
                
    def variableTransformer(self, batchSize, posPerBatch,  negPerBatch = None ):
        if not negPerBatch == None:
            return (np.arange( batchSize*posPerBatch ).reshape( batchSize , posPerBatch), 
                np.arange(start = batchSize*posPerBatch, 
                    stop = batchSize*posPerBatch + batchSize*negPerBatch ).reshape(batchSize, negPerBatch) )
        else:
            return np.arange( batchSize*posPerBatch ).reshape( batchSize, posPerBatch )

    def beforeForwardPass(self, retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        self.model_variable.weight.data = (
            from_dlpack(self.CUPYcorpus[ reshapedRetrieval ].toDlpack() ) )

    def afterOptimizerStep(self,retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        self.CUPYcorpus[ reshapedRetrieval ] = (
            cupy.fromDlpack( to_dlpack( self.model_variable.weight.data ) ) )
        
    
class OptimizerFactory(_Common): #to do later, able to load matrixes to continue training
#take into account different size embedding matrices 

    def __init__(self, given_optimizer,  total_classes,  embed_dimension, model, variable_name, dtype='float32', CPUPinn = False):
        self.given_optimizer = given_optimizer
        self.total_classes = total_classes
        self.embed_dimension = embed_dimension
        self.model = model
        self.variable_name = variable_name
        self.datatype = dtype
        optimizer_index = None
        self.CPUPinn = CPUPinn

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
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)

        self.CUPYcorpi = []
        for optVar in self.optVarList:
            self.CUPYcorpi.append( cupy.zeros( shape =( self.total_classes, self.embed_dimension), dtype=self.datatype) )

        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)
            
    def beforeForwardPass(self, retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        for idx, optVar in enumerate(self.optVarList):
            self.given_optimizer.state_dict()['state'][ self.optimizerKey ][optVar] = (
                from_dlpack( self.CUPYcorpi[idx][ reshapedRetrieval ].toDlpack() )   )

    def afterOptimizerStep(self, retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        for idx, optVar in enumerate(self.optVarList):
            self.CUPYcorpi[idx][ reshapedRetrieval ] = (
                cupy.fromDlpack( to_dlpack( self.given_optimizer.state_dict()['state'][ self.optimizerKey ][optVar] ) )  )
    
class COM(_Common):

    def __init__(self, total_classes, datatype = 'uint32', CPUPinn = False  ):
        self.total_classes = total_classes
        self.datatype = datatype
        self.CPUPinn = CPUPinn

    def comInit(self, CPUPinn=False):
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)
        self.CUPYCom = cupy.zeros( shape =( self.total_classes, self.total_classes), dtype=self.datatype)
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)
    
class DataGadget(_Common):
    def __init__(self, fileName, CPUPinn=False):
        self.fileName = fileName
        self.CPUPinn = CPUPinn

    def gadgetInit(self):
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)

        self.CUPYcorpus = cupy.load( self.fileName)

        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)

    def getData(self, indexes):
        return from_dlpack( self.CUPYcorpus[indexes].toDlpack() )

    def insertData(self, dataObject, indexes):
         self.CUPYcorpus[indexes] =  (
            cupy.fromDlpack( to_dlpack( dataObject ) ) )
    
    
    
