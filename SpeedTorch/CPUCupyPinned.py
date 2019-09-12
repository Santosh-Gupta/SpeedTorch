import numpy as np
import cupy
# cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.memory.malloc_managed).malloc)
import torch
import os

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

class PMemoryMM(cupy.cuda.memory.BaseMemory):
    def __init__(self, size):
        self.size = size
        self.device_id = cupy.cuda.device.get_device_id()
        self.ptr = 0
        if size > 0:
            self.ptr = cupy.cuda.runtime.hostAlloc(size, 0)
    def __del__(self):
        if self.ptr:
            cupy.cuda.runtime.freeHost(self.ptr)

def my_pinned_allocatorMM(bsize):
    return cupy.cuda.memory.MemoryPointer(PMemory(bsize),0)

class _CommonMM():
    def _preInit(self):
        fileNumber = 0
        while os.path.isfile( self.diskname + str(fileNumber) + '.memmap.cpy.npy'  ) == True:
            fileNumber = fileNumber + 1
        else:
            self.fileName = self.diskname + str(fileNumber) + '.memmap' 
        numpyMemmap = np.memmap( self.fileName, dtype='float32', mode='w+', shape=(self.total_classes ,self.embed_dimension ))
        np.save( self.fileName + '.cpy' , numpyMemmap, allow_pickle=True)
        del numpyMemmap
        os.remove(self.fileName)

    def getCupyMM(self):
        return self.CUPYmemmap

    def saveCupy(self, saveFileName):
        cupy.save( saveFileName, self.CUPYmemmap)

    def getNumpyVersion(self):
        return cupy.asnumpy(self.CUPYmemmap)

    def _getReshapedRetrieval( self, retrievedPosIndexes , retrievedNegIndexes = None):
        if not retrievedNegIndexes is None:
            reshapedRetrieval =  np.concatenate( [ retrievedPosIndexes.reshape(-1) , retrievedNegIndexes.reshape(-1) ] )
        else:
            reshapedRetrieval = retrievedPosIndexes.reshape(-1)
        return reshapedRetrieval

class ModelFactoryMM(_CommonMM):

    def __init__(self, model_variable,  total_classes,  embed_dimension, diskname = 'variable', datatype = 'float32', CPUPinn = False):
        self.model_variable = model_variable
        self.total_classes = total_classes
        self.embed_dimension = embed_dimension
        self.diskname = diskname
        self.datatype = datatype
        self.CPUPinn = CPUPinn

    def zerosInit(self ):
        #Initialize the memmap with just zeros
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)
        self._preInit()
        self.CUPYmemmap = cupy.load( self.fileName+'.cpy.npy' , mmap_mode = 'r+' )
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)

    def uniformDistributionInit(self, low, high):
        #Initialize the memmap with a uniform distribution 
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)

        self._preInit()
        self.CUPYmemmap = cupy.load( self.fileName+'.cpy.npy' , mmap_mode = 'r+' )

        if self.total_classes  > 100000:    
            for i in range( int( self.total_classes/100000) ):
                j=i*100000   
                self.CUPYmemmap[j:j+100000] = cupy.random.uniform(low=low, high=high, size=(100000, self.embed_dimension), dtype=self.datatype)
            
            for i in range( int( self.total_classes/100000)*100000,  int( self.total_classes/10000) ):
                j=i*10000   
                self.CUPYmemmap[j:j+10000] = cupy.random.uniform(low=low, high=high, size=(10000, self.embed_dimension), dtype=self.datatype)
            
            for i in range( int( self.total_classes /10000)*10000 , self.total_classes ):
                self.CUPYmemmap[i] = cupy.random.uniform(low=low, high=high, size=(self.embed_dimension), dtype=self.datatype)
        
        elif self.total_classes  > 10000:    
            for i in range( int(self.total_classes/10000) ):
                j=i*10000
                self.CUPYmemmap[j:j+10000] = cupy.random.uniform(low=low, high=high, size=(10000, self.embed_dimension), dtype=self.datatype)
            
            for i in range( int( self.total_classes/10000)*10000 , self.total_classes ):
                self.CUPYmemmap[i] = cupy.random.uniform(low=low, high=high, size=(self.embed_dimension), dtype=self.datatype)

        else:
            for i in range( self.total_classes  ):
                self.CUPYmemmap[i] = cupy.random.uniform(low=low, high=high, size=(self.embed_dimension), dtype=self.datatype)

        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)
                
    def normalDistributionInit(self, mean, stdDev):
        #Initialize the memmap with a normal distribution 
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)

        self._preInit()
        self.CUPYmemmap = cupy.load( self.fileName+'.cpy.npy' , mmap_mode = 'r+' )

        if self.total_classes > 100000: 
            for i in range( int(self.total_classes/100000) ):
                j=i*100000
                self.CUPYmemmap[j:j+100000] = cupy.random.normal(loc=mean, scale=stdDev, size=(100000, self.embed_dimension), dtype=self.datatype )
                
            for i in range( int(self.total_classes/100000)*100000, int(self.total_classes/10000) ):
                j=i*10000
                self.CUPYmemmap[j:j+10000] = cupy.random.normal(loc=mean, scale=stdDev, size=(10000, self.embed_dimension), dtype=self.datatype )

            for i in range( int(self.total_classes/10000)*10000 , self.total_classes ):
                self.CUPYmemmap[i] = cupy.random.normal(loc=mean, scale=stdDev, size=(self.embed_dimension), dtype=self.datatype )

        elif self.total_classes > 10000:
            for i in range( int(self.total_classes/10000) ):
                j=i*10000
                self.CUPYmemmap[j:j+10000] = cupy.random.normal(loc=mean, scale=stdDev, size=(10000, self.embed_dimension), dtype=self.datatype )

            for i in range( int(self.total_classes/10000)*10000 , self.total_classes ):
                self.CUPYmemmap[i] = cupy.random.normal(loc=mean, scale=stdDev, size=(self.embed_dimension), dtype=self.datatype )

        else:
            for i in range( self.total_classes ):
                self.CUPYmemmap[i] = cupy.random.normal(loc=mean, scale=stdDev, size=(self.embed_dimension), dtype=self.datatype )

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
            from_dlpack(self.CUPYmemmap[ reshapedRetrieval ].toDlpack() ) )

    def afterOptimizerStep(self,retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        self.CUPYmemmap[ reshapedRetrieval ] = (
            cupy.fromDlpack( to_dlpack( self.model_variable.weight.data ) ) )
        
    
class OptimizerFactoryMM(_CommonMM): #to do later, able to load matrixes to continue training
#take into account different size embedding matrices 

    def __init__(self, given_optimizer,  total_classes,  embed_dimension, model, variable_name, dtype='float32' , CPUPinn = False):
        self.given_optimizer = given_optimizer
        self.total_classes = total_classes
        self.embed_dimension = embed_dimension
        self.model = model
        self.variable_name = variable_name
        self.dtype = dtype
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
                self.diskname = item[0][:-7] + given_optimizer.__str__().split(' ', 1)[0]
        if optimizer_index == None:
            print( 'Error: No variable with that name is in Model. Please initialize again with correct name' ) 
            return

        optimizerKeyList = list(self.given_optimizer.state_dict()['state'].keys())
        self.optimizerKey = optimizerKeyList[ optimizer_index ]

    def _preInit(self):
        for optVar in self.optVarList:
            fileNumber = 0
            while os.path.isfile( self.diskname + str(fileNumber) + '.memmap.cpy.npy'  ) == True:
                fileNumber = fileNumber + 1
            else:
                self.fileName = self.diskname + str(fileNumber) + '.memmap' 
            numpyMemmap = np.memmap( self.fileName+optVar, dtype='float32', mode='w+', shape=(self.total_classes ,self.embed_dimension ))
            np.save( self.fileName + optVar + '.cpy' , numpyMemmap, allow_pickle=True)
            del numpyMemmap
            os.remove(self.fileName+optVar)

    def optInit(self):
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)

        self._preInit()
        self.CUPYmemmap = []
        for optVar in self.optVarList:
            self.CUPYmemmap.append( cupy.load( self.fileName+optVar+'.cpy.npy' , mmap_mode = 'r+' )  )

        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)
            
    def beforeForwardPass(self, retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        for idx, optVar in enumerate(self.optVarList):
            self.given_optimizer.state_dict()['state'][ self.optimizerKey ][optVar] = (
                from_dlpack( self.CUPYmemmap[idx][ reshapedRetrieval ].toDlpack() )   )

    def afterOptimizerStep(self, retrievedPosIndexes , retrievedNegIndexes = None):
        reshapedRetrieval = self._getReshapedRetrieval( retrievedPosIndexes, retrievedNegIndexes )

        for idx, optVar in enumerate(self.optVarList):
            self.CUPYmemmap[idx][ reshapedRetrieval ] = (
                cupy.fromDlpack( to_dlpack( self.given_optimizer.state_dict()['state'][ self.optimizerKey ][optVar] ) )  )
    
class COMMM(_CommonMM):

    def __init__(self, total_classes, diskname = 'COM', datatype = 'uint32', CPUPinn = False  ):
        self.total_classes = total_classes
        self.datatype = datatype
        self.diskname = diskname
        self.CPUPinn = CPUPinn

    def _preInit(self): #Can't depend on inherited since the shape is different 

        fileNumber = 0
        while os.path.isfile(diskself.disknamename+str(fileNumber) + 'memmap' ) == false:
            fileNumber = fileNumber + 1
        else:
            self.fileName = self.diskname+str(fileNumber)
        numpyMemmap = np.memmap(self.fileName, dtype=datatype, mode='w+', shape=(total_classes , total_classes ))
        np.save( self.fileName+'.cpy' , numpyMemmap, allow_pickle=True)
        del numpyMemmap
        os.remove(self.fileName)

    def comInit(self, CPUPinn=False):
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)

        self._preInit()
        self.CUPYmemmap = cupy.load( fileName+'.cpy.npy'  , mmap_mode = 'r+' )

        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)
        
class DataGadgetMM(_CommonMM):
    def __init__(self, fileName, CPUPinn=False):
        self.Numpyfilename = Numpyfilename
        self.CPUPinn = CPUPinn

    def gadgetInit(self):
        if self.CPUPinn == True:
            cupy.cuda.set_allocator(my_pinned_allocator)

        self.CUPYmemmap = cupy.load( self.fileName , mmap_mode = 'r+' )

        if self.CPUPinn == True:
            cupy.cuda.set_allocator(None)

    def getData(self, retrievedPosIndexes , retrievedNegIndexes = None):
        return from_dlpack( self.CUPYmemmap[ reshapedRetrieval ].toDlpack() )
