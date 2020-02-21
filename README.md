<p align="center">
  <img src="https://i.imgur.com/wr4VaUV.png?1">
</p>

# SpeedTorch

[![Join the chat at https://gitter.im/SpeedTorch/community](https://badges.gitter.im/SpeedTorch/community.svg)](https://gitter.im/SpeedTorch/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Downloads](https://pepy.tech/badge/speedtorch)](https://pepy.tech/project/speedtorch) [![Downloads](https://pepy.tech/badge/speedtorch/week)](https://pepy.tech/project/speedtorch/week)

Faster pinned CPU tensor <-> GPU Pytorch variabe transfer and GPU tensor <-> GPU Pytorch variable transfer, in certain cases. 

## Update 9-29-19

Since for some systems, using the pinned Pytorch CPU tensors is faster than using Cupy tensors (see 'How It Works' section for more detail), I created general Pytorch tensor classes `PytorchModelFactory` and `PytorchOptimizerFactory` which can specifiy either setting the tensors to `cuda` or `cpu`, and if using `cpu`, if its memory should be pinned. The original `GPUPytorchModelFactory` and `GPUPytorchOptimizerFactory` classes are still in the library, so no existing code using SpeedTorch should be affected. The documentation has been updated to include these new classes. 

## What is it?

This library revovles around Cupy tensors pinned to CPU, which can achieve **3.1x** faster CPU -> GPU transfer than regular Pytorch Pinned CPU tensors can, and **410x** faster GPU -> CPU transfer. Speed depends on amount of data, and number of CPU cores on your system (see the How it Works section for more details)

The library includes functions for embeddings training; it can host embeddings on CPU RAM while they are idle, sparing GPU RAM. 

## Inspiration

I initially created this library to help train large numbers of embeddings, which the GPU may have trouble holding in RAM. In order to do this, I found that by hosting some of the embeddings on the CPU can help achieve this. Embedding systems use sprase training; only fraction of the total prameters participate in the forward/update steps, the rest are idle. So I figured, why not keep the idle parameters off the GPU during the training step? For this, I needed fast CPU -> GPU transfer. 

For the full backstory, please see the Devpost page

https://devpost.com/software/speedtorch-6w5unb

## What can fast CPU->GPU do for me? (more that you might initially think)

With fast CPU->GPU, a lot of fun methods can be developed for functionalities which previously people thought may not have been possible. 

🏎️    Incorporate SpeedTorch into your data pipelines for fast data transfer to/from CPU <-> GPU

🏎️    Augment training parameters via CPU storage. As long as you have enough CPU RAM, you can host any number of embeddings without having to worry about the GPU RAM.

🏎️    Use Adadelta, Adamax, RMSprop, Rprop, ASGD, AdamW, and Adam optimizers for sparse embeddings training. Previously, only SpraseAdam, Adagrad, and SGD were suitable since only these directly support sparse gradients. 

<p align="center">
  <img src="https://i.imgur.com/6o8C1BP.gif">
</p>

## Benchmarks

### Speed

(Edit 9-20-19, one of the Pytorch developers pointed out some minor bugs in the original bench marking code, the values and code have been updated)

Here is a notebook comparing transfer via SpeedTorch vs Pytorch tensors, with both pinned CPU and Cuda tensors. All tests were done with a colab instance with a Tesla K80 GPU, and 2 core CPU. 

UPDATE 10-17-19: Google Colab is now standard with 4 core CPUs, so this notebook will give different results than what is reported below, since Pytorch's indexing kernals get more efficient as the number of CPU cores increase. 

https://colab.research.google.com/drive/1PXhbmBZqtiq_NlfgUIaNpf_MfpiQSKKs

This notebook times the data transfer of 131,072 float32 embeddings of dimension 128, to and from the Cupy/Pytorch tensors and Pytorch variables, with n=100. Google Colab's CPU has 4 cores, which has an impact on the transfer speed. CPU's with a higher number of cores will see less of an advatage to using SpeedTorch. 

The table below is a summary of the results. Transfering data from  Pytorch cuda tensors to the Cuda Pytorch embedding variable is faster than the SpeedTorch equivalent, but for all other transfer types, SpeedTorch is faster. And for the sum of both steps transferring to/from the Cuda Pytorch embedding, SpeedTorch is faster than the Pytorch equivalent for both the regular GPU and CPU Pinned tensors. 

I have noticed that different instances of Colab result in different speed results, so keep this in mind while reviewing these results. A personal run of the colab notebook may result in different values, though the order of magnetude of the results are generally the same. 

The transfer times in the following tables are given in seconds. This benchmarking was preformed with a colab instance whose CPU has 2 cores. Colab has a Pro version of paid instances which are 4 core CPUs, so the following benchmarking would not reflect for those instances. 

| Tensor Type	| To Cuda Pytorch Variable	| Comparison |
| --- | --- | --- |
| SpeedTorch(cuda)	| 0.0087	| 6.2x Slower than Pytorch Equivalent |
| SpeedTorch(PinnedCPU)	| 0.0154	| 3.1x Faster than Pytorch Equivalent |
| Pytorch(cuda)	| 0.0014	| 6.2x Faster than SpeedTorch Equivalent |
| Pytorch(PinnedCPU)	| 0.0478	| 3.1x Slower than SpeedTorch Equivalent |
		
| Tensor Type	| From Cuda Pytorch Variable	| Comparison |
| --- | --- | --- |
| SpeedTorch(cuda)	| 0.0035	| 9.7x Faster than Pytorch Equivalent |
| SpeedTorch(PinnedCPU)	| 0.0065	| 410x Faster than Pytorch Equivalent |
| Pytorch(cuda)	| 0.0341	| 9.7x Slower than SpeedTorch Equivalent |
| Pytorch(PinnedCPU)	| 2.6641	| 410x Slower than SpeedTorch Equivalent |
		
| Tensor Type	| Sum  of to/from Cuda Pytorch Variable	| Comparison |
| --- | --- | --- |
| SpeedTorch(cuda)	| 0.0122	| 2.9x Faster than Pytorch Equivalent |
| SpeedTorch(PinnedCPU)	| 0.0219	| 124x Faster than Pytorch Equivalent |
| Pytorch(cuda)	| 0.0355	| 2.9x Slower than SpeedTorch Equivalent |
| Pytorch(PinnedCPU)	| 2.7119	| 124x Slower than SpeedTorch Equivalent |

Similar benchmarks were calculated for transferring to/from Pytorch Cuda optimizers. The results are basically the same, here is the notebook used for the optimizers benchmarking

https://colab.research.google.com/drive/1Y2nehd8Xj-ixfjkj2QWuA_UjQjBBHhJ5

### Memory 

Although SpeedTorch's tensors are generally faster than Pytorch's, the drawback is SpeedTorch's tensors use more memory. However, because transferring data can happen more quickly, you can use SpeedTorch to augment the number of embeddings trained in your architecture by holding parameters in both the GPU And CPU. 

This table is a summary of benchmarking done in Google Colab. From my experience, there seems to be some variation in the reported memory values in Colab, +-0.30 gb, so keep this in mind while reviewing these numbers. The values are for holding a 10,000,000x128 float32 tensor. 

|Tensor Type	| CPU (gb)	| GPU (gb)|
| --- | --- | --- |
|Cupy PinnedCPU |	9.93 |	0.06|
|Pytorch PinnedCPU |	6.59 |	0.32|
|Cupy Cuda |	0.39 |	9.61|
|Pytorch Cuda |	1.82 |	5.09|

Although Pytorch's time to/from for Pytorch GPU tensor <-> Pytorch cuda Variable is not as fast as the Cupy equivalent, the speed is still workable. So if memory is still a concern, a best of both worlds approach would be to SpeedTorch's Cupy CPU Pinned Tensors to store parameters on the CPU, and SpeedTorch's Pytorch GPU tensors to store parameters on the GPU. 

This is the notebook I used for measuring how much memory each variable type takes. 
https://colab.research.google.com/drive/1ZKY7PyuPAIDrnx2HdtbujWo8JuY0XkuE
If using this in Colab, you will need to restart the enviroment after each tensor creation, to get a measure for the next tensor. 

## What systems get a speed advantage?

For the CPU<->GPU transfer, it depends on the amount of data being transfered, and the number of cores you have. Generally for 1-2 CPU cores SpeedTorch will be much faster. But as the number of CPU cores goes up, Pytorch's CPU<->GPU indexing operations get more efficient. For more details on this, please see the next 'How it works' section. For an easy way to see if you get a speed advantage in your system, please run the benchmarking code on your system, but change the amount of data to reflect the amount that you will be working with in your application. 

For the  GPU <-> GPU transfer, if using ordinary indexing notations in vanilla Pytorch, all systems will get a speed increase because SpeedTorch bypasses a bug in Pytorch's indexing operations. But this bug can be avoided if using the nightly version, or just using different indexing notions, please see the 'How it works' section for more details. 

## How it works?

Update 9-20-19: I initially had no idea why this is faster than using Pytorch tensors; I stumbled upon the speed advantage by accident. But one of the Pytorch developers on the Pytorch forum pointed it out. 

As for the better CPU<->GPU transfer, it's because SpeedTorch avoids a CPU indexing operation by masquarding CPU tensors as GPU tensors. The CPU index operation may be slow if working on with very few CPU cores, such as 2 in Google Colab, but may be faster if you have many cores.  It depends on how much data you're transfering and how many cores you have. 

As for the better GPU<->GPU transfer, it's because SpeedTorch avoids a bug in the indexing operation. This bug can also be avoided by using the nightly builds, or using `index_select` / `index_copy_` instead of `a[idx]` notation in 1.1/1.2. 

For more details of this, please see this Pytorch post

https://discuss.pytorch.org/t/introducing-speedtorch-4x-speed-cpu-gpu-transfer-110x-gpu-cpu-transfer/56147/2

where a Pytorch engineer gives a detailed analysis on how the Cupy indexing kernals are resulting in speed ups in certain cases. It's not the transfer itself that is getting faster, but the indexing kernals which are being used. 

As for how the memory management in Cupy works, I direct to these two stackoverflow questions I asked, where brilliant user Robert Crovella not only gave detailed explanations, but also figured out how to allocate pinned memory to Cupy arrays by developing his own memory allocator for Cupy. This is basically the core technology behind SpeedTorch. 

https://stackoverflow.com/questions/57750125/cupy-outofmemoryerror-when-trying-to-cupy-load-larger-dimension-npy-files-in-me

https://stackoverflow.com/questions/57752516/how-to-use-cuda-pinned-zero-copy-memory-for-a-memory-mapped-file

## Guide

### Geting Started

SpeedTorch is pip installable. You need to have Cupy installed and imported before you import SpeedTorch. 

```
!pip install SpeedTorch
import cupy
import SpeedTorch
```

### Using SpeedTorch to increase speed transfer of data from CPU to GPU

This colab notebook shows how to load data into SpeedTorch using its Data Gadget, and how to transfer this data to/from a Pytorch cuda variable. 

https://colab.research.google.com/drive/185Z5Gi62AZxh-EeMfrTtjqxEifHOBXxF

<p align="center">
  <img src="https://i.imgur.com/Dc2DKsC.png">
</p>

Please see the speed benchmarking notebook to see the speed advantage of using SpeedTorch. 

### Using SpeedTorch to use non-sparse optimizers (in this case, Adamax) for sparse training

For people first trying to figure out how to use SpeedTorch, I recommend following this example, since word2vec is one of the more commonly known algorithms in machine learning. 

https://colab.research.google.com/drive/1ApJR3onbgQWM3FBcBKMvwaGXIDXlDXOt

The notebook shows how to train word2vec the regular way, then shows how to use SpeedTorch to train on the same data, using one of the optimizers normally not supported for sparse training. This is possible because since all the embeddings contained in the embedding variable has an update during each step, you can set `sparse=False` during initialization. 

### Augment training parameters via CPU storage

tl;dr:

**Normal training**: Pytorch embedding variables contain all embeddings. Pytorch optimizer contain all the corresponding parameter weights for each embedding. 

**SpeedTorch traing**: Pytorch embeddng variables only contain a batch of embeddings. Pytorch optimizer only contains all the corresponding parameter weights for that batch. SparseTorch tensors contain the rest, and exchanges the embeddings/weights with the Pytorch variable at each step. 

In sparse training algorithms like word2vec, GloVe, or Neural Collaborative Filtering, only a fraction of the total parameters (embeddngs) are trained during every step. If your GPU can not handle all of your embeddings at a desired embedding size, an option would be to host some of your parameters on pinned CPU Cupy arrays, and transfer those parameters to your model tensors as needed. Doing this primary in Pytorch would be very slow, especially because transferring parameters between a Cuda mounted Pytorch variable and a pinned CPU pytorch tensor can take 2.5-3 seconds (on Google Colab). fortunately,
this step only takes 0.02-0.03 seconds with SpeedTorch!

**Case Uses :**

--2,829,853 book embeddings--

SpeedTorch was used in training 2,829,853 books for a rare book recommender.

https://github.com/Santosh-Gupta/Lit2Vec2

https://devpost.com/software/lit2vec2

Each book had an embedding of size of 400, but an embedding size of 496 could have been used, the 400 embedding size was due to limits of space on my Google Drive to store the trained embeddings :(. But the limits of the GPU RAM is no longer an issue :)
Here is a direct link to a demo training notebook, which trains with a 496 embedding size using SpeedTorch

NOTE: You need the version of the Colab notebook that has 25 gb of RAM, instead of the usual 12 gb. To get this type of instance, you need to crash your current instance due to overwhelming the RAM, and then a note in the bottom left corner asking if you would like to upgrade. You can do this by making a loop that keeps doubling the size of a numpy float matrix. 

https://colab.research.google.com/drive/1AqhT-HetihXMET1wJQROrC3Q9tFJqJ19

Here is a directly link with the same model and data, but doesn't use SpeedTorch

https://colab.research.google.com/drive/1idV1jBOUZVPCfdsy40wIrRPHeDOanti_

Using the orthodox training method, the largest embedding size that colab is able to handle is 255-260, any higher than that and a cuda error will occur

`RuntimeError: CUDA out of memory. Tried to allocate 2.74 GiB (GPU 0; 11.17 GiB total capacity; 8.22 GiB already allocated; 2.62 GiB free; 5.05 MiB cached)`

--14,886,544 research paper embeddings--

https://github.com/Santosh-Gupta/Research2Vec2

SpeedTorch can allow me to train 14,886,544 research paper embeddings at an embedding size of 188, by allowing my to store my target embeddings on the CPU, while keeping my context embeddings on the GPU (SGD optimizer was used, so there are no optimizer weights). 

Here is a direct link to the notebook. 

https://colab.research.google.com/drive/1saKzsaHoy6O_U1DF_z15_Qkr5YLNI_GR

NOTE: You need the version of the Colab notebook that has 25 gb of RAM, instead of the usual 12 gb. To get this type of instance, you need to crash your current instance due to overwhelming the RAM, and then a note in the bottom left corner asking if you would like to upgrade. You can do this by making a loop that keeps doubling the size of a numpy float matrix. 

Without SpeedTorch, only an embedding size of 94-96 can be used on Google Colab Tesla K80 GPU before an `RuntimeError: CUDA out of memory` error. Here is a version of the training without using SpeedTorch. 

https://colab.research.google.com/drive/1jh7RUgeajhdWdGNfWG3Twm1ZjyTQU0KR

### Best Practices

1) Whenever using the Cupy GPU tensors, initialize these before any pinned CPU tensors. This is because the initialization of the Cupy GPU tensors seem to uses a solid amount of CPU RAM. So if you're limited on CPU RAM, and you already have your pinned CPU tensors in memory, then initializing the cupy GPU tensors may cause a crash. 

2) If you're able to fit all of your parameters in your GPU memory, use pure Pytorch since this is the fastest option for training. But if you can't fit all your parameters in memory, split your parameters (keep in mind that your optimizers also have weights) between SpeedTorch's Cupy cuda tensors and SpeedTorch's Cupy pinned CPU tensors; this is the 2nd fastest options. But, if you're still not able to fit all your parameters into memory that way, then split your parameters between SpeedTorch's Cupy pinned CPU tensors, and SpeedTorch's Pytorch cuda tensors; this is slower than both options, but uses less GPU memory. For the 3rd option, here are two notebooks which shows an example of this https://colab.research.google.com/drive/1AqhT-HetihXMET1wJQROrC3Q9tFJqJ19 , https://colab.research.google.com/drive/1saKzsaHoy6O_U1DF_z15_Qkr5YLNI_GR

3) After training, saving any cuda variables will cause an increase in memory usage, and may cause a crash, especially with Cupy. If you're at the limits of your RAM. In this case, use the `getNumpyVersion` method to get a numpy version of your tensor, and then use use numpy.save or hdpy/pytables to save your numpy array. Numpy save is more lightweight. 

## Need Help?

Either open an issue, or chat with me directory on Gitter here https://gitter.im/SpeedTorch

## Future Work

I am looking incoporate more functionalities around the fast CPU -> GPU transfer. If you have an idea, please post a Github Issue. 

## Experimental

In addition the the Cupy GPU/pinned CPU and Pytorch GPU tensors, SpeedTorch also has Pytorch pinned CPU tensors, and Cupy memmap GPU/pinned CPU tensors. I have not found a solid use for these sorts of tensors, but they're fully coded and availible for use. 

https://github.com/Santosh-Gupta/SpeedTorch/tree/master/SpeedTorch

One area I would like to look at is if there is a way to have RAM memory reduction by using Cupy Memmaps. So far they use just as much memory as the live versions. 

## Documentation 

### Class ModelFactory

```python
ModelFactory(model_variable,  total_classes,  embed_dimension, datatype = 'float32', CPUPinn = False)
```

Creates switchers for model variables using Cupy. Switches variables from your full embedding collection and your model batch collection. Each variable needs its own switcher. 

Example:

```python
uEmbed_switcher = SpeedTorch.ModelFactory( skip_gram_modelSparse.u_embeddings, total_classes=50000, embed_dimension=128)
```

Arguments:

`model_variable`: Specific variable from your model you would like to create a switcher for.

`total_classes`: The total amount of embeddings to be trained. 

`embed_dimension`: Dimension of the embeddings.

`datatype` (optional): Datatype for the variable. Default is 'float32'. 

`CPUPinn` (optional): Pin your full embedding collection to CPU. Spares GPU memory, but data transfer will be slower. Default is False. 

Methods:

`zerosInit()` : Initializes the variable switcher full collection with zeros:

`uniformDistributionInit(low, high)`: Initializes the variable switcher full collection with a uniform distribution from `low` to `high`

`normalDistributionInit(mean, stdDev)`: Initializes the variable switcher full collection with a normal distribution with a mean of `mean` and a standard deviation of `stdDev`

`variableTransformer( batchSize, posPerBatch,  negPerBatch = None )`: Sets up a dummy input to be used for the forward step of you model. `batchSize` is the size of your batch, and `posPerBatch` is the number of positive examples per batch. If a second dummy input is needed for the negative examples, `negPerBatch` (optional) can be set to the number of negative examples, and two dummy inputs will be returned instead of one. 

`beforeForwardPass(retrievedPosIndexes , retrievedNegIndexes = None)`: Switches embeddings from the full embeddings collection to your model embeddings. `retrievedPosIndexes` is the indexes of the positive samples to be retrieved. If negative samples are to be retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

`afterOptimizerStep( retrievedPosIndexes , retrievedNegIndexes = None)`: Switches updated embeddings from your model to the full embeddings collection. `retrievedPosIndexes` is the indexes of the positive samples that were retrieved. If negative samples were retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

`saveCupy(saveFileName)`: Save tensor to .npy file. 

`loadCupy(loadFileName)`: Load tensor from .npy file. 

`getNumpyVersion`: Get numpy version of tensor. 

### Class OptimizerFactory

```pyton
OptimizerFactory( given_optimizer,  total_classes,  embed_dimension, model, variable_name, dtype='float32' , CPUPinn = False)
```

Creates switchers for optimizer variables using Cupy. Switches variables from your full embedding collection and your optimizer batch collection. Each variable needs its own switcher. 

Example:

```python
uAdagrad_switcher = SpeedTorch.OptimizerFactory(given_optimizer,  total_classes,  embed_dimension, model, variable_name, dtype='float32', CPUPinn = False)
```

Arguments:

`given_optimizer`: The optimizer initialized with your model weights. If using for embeddings training, remember to set the `sparse` parameter to `False`. Currently, supported optimizers are SparseAdam, Adadelta, Adamax, Adam, AdamW, ASGD, and RMSprop. Rprop is also inlcuded, but needs the first forward pass, and `loss.backward()` step to be completed for initializing the OptimizerFactory instance. This is due to the Rprop optimizer needing gradients of its parameters for initialization.  

`total_classes`: The total amount of embeddings to be trained. 

`embed_dimension`: Dimension of the embeddings.

`model`: The instance of your model. 

`variable_name`: Exact name of the variable defined in your model. 

`dtype` (optional): Data type of your variable. Default is 'float32'

`CPUPinn` (optional): Pin your full optimizer variable weight collection to CPU. Spares GPU memory, but data transfer will be slower. Default is False. 

Methods:

`optInit`: Initializes the optimizer variable switcher. 

`beforeForwardPass(retrievedPosIndexes , retrievedNegIndexes = None)`: Switches optimizer variable weights from the full weights collection to optimizer weight tensor. `retrievedPosIndexes` is the indexes of the positive samples to be retrieved. If negative samples are to be retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

`afterOptimizerStep( retrievedPosIndexes , retrievedNegIndexes = None)`: Switches optimizer variable weights from your optimizer to the full weights collection. `retrievedPosIndexes` is the indexes of the positive samples that were retrieved. If negative samples were retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

### Class DataGadget

Creates a tensor whose main function is to transfer it's contents to a Pytorch cuda variable. 

```pyton
DataGadget( fileName, CPUPinn=False)
```

Arguments:

`fileName`: Location of data .npy file to be opened

`CPUPinn`:  (optional): Pin your data to CPU. Default is False. 

Methods:

`getData(indexes)`: Retrieves data in a format that is ready to be accepted by a Pytorch Cuda Variable. `indexes` is the indexes of the tensor from which to retrieve data from. 

`insertData(dataObject, indexes)`: Insert data from a Pytorch Cuda Variable. `dataObject` is the Pytorch cuda variable tensor data from which the data is is going to be retrived from, and `indexes` of the tensor from which to retrieve data from. 

`saveCupy(saveFileName)`: Save tensor to .npy file. 

`loadCupy(loadFileName)`: Load new tensor from .npy file. 

`getNumpyVersion`: Get numpy version of tensor. 

Please see this notebook on how to use the data gadget

https://colab.research.google.com/drive/185Z5Gi62AZxh-EeMfrTtjqxEifHOBXxF

### Class PytorchModelFactory

```python
PytorchModelFactory(model_variable,  total_classes,  embed_dimension, datatype = 'float32', deviceType = 'cuda', pinType = False)
```

Creates switchers for model variables using Pytorch tensors. Switches variables from your full embedding collection and your model batch collection. Each variable needs its own switcher. 

Example:

```python
uEmbed_switcher = SpeedTorch.PytorchModelFactory( skip_gram_modelSparse.u_embeddings, total_classes=50000, embed_dimension=128)
```

Arguments:

`model_variable`: Specific variable from your model you would like to create a switcher for.

`total_classes`: The total amount of embeddings to be trained. 

`embed_dimension`: Dimension of the embeddings.

`datatype` (optional): Datatype for the variable. Default is 'float32'. 

`deviceType` (optional): Set device either to 'cuda' or 'cpu'. Default is 'cuda'

`pinType` (optional): If device is set to 'cpu', you can specify using pinned memory. Default is 'False'. 


Methods:

`zerosInit()` : Initializes the variable switcher full collection with zeros:

`uniformDistributionInit(low, high)`: Initializes the variable switcher full collection with a uniform distribution from `low` to `high`

`normalDistributionInit(mean, stdDev)`: Initializes the variable switcher full collection with a normal distribution with a mean of `mean` and a standard deviation of `stdDev`

`customInit(initFunction, *args)`: Use any Pytorch initializer for the variable switchers full collection. Pass the initializer using `initFunction` and its corresponding arguments using `*args`.

`variableTransformer(batchSize, posPerBatch,  negPerBatch = None )`: Sets up a dummy input to be used for the forward step of you model. `batchSize` is the size of your batch, and `posPerBatch` is the number of positive examples per batch. If a second dummy input is needed for the negative examples, `negPerBatch` (optional) can be set to the number of negative examples, and two dummy inputs will be returned instead of one. 

`beforeForwardPass(retrievedPosIndexes , retrievedNegIndexes = None)`: Switches embeddings from the full embeddings collection to your model embeddings. `retrievedPosIndexes` is the indexes of the positive samples to be retrieved. If negative samples are to be retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

`afterOptimizerStep(retrievedPosIndexes , retrievedNegIndexes = None)`: Switches updated embeddings from your model to the full embeddings collection. `retrievedPosIndexes` is the indexes of the positive samples that were retrieved. If negative samples were retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

`saveTorch(saveFileName)`: Save tensor to file using torch.save

`loadTorch(loadFileName)`: Load tensor using torch.load

`getNumpyVersion`: Get numpy version of tensor. 

### Class PytorchOptimizerFactory

```pyton
PytorchOptimizerFactory( given_optimizer,  total_classes,  embed_dimension, model, variable_name, dtype='float32', deviceType = 'cuda', pinType = False)
```

Creates switchers for optimizer variables using Pytorch tensors. Switches variables from your full embedding collection and your optimizer batch collection. Each variable needs its own switcher. 

Example:

```python
uAdagrad_switcher = SpeedTorch.PytorchOptimizerFactory(given_optimizer,  total_classes,  embed_dimension, model, variable_name, dtype='float32')
```

Arguments:

`given_optimizer`: The optimizer initialized with your model weights. If using for embeddings training, remember to set the `sparse` parameter to `False`. Currently, supported optimizers are SparseAdam, Adadelta, Adamax, Adam, AdamW, ASGD, and RMSprop. Rprop is also inlcuded, but needs the first forward pass, and `loss.backward()` step to be completed for initializing the OptimizerFactory instance. This is due to the Rprop optimizer needing gradients of its parameters for initialization.  

`total_classes`: The total amount of embeddings to be trained. 

`embed_dimension`: Dimension of the embeddings.

`model`: The instance of your model. 

`variable_name`: Exact name of the variable defined in your model. 

`dtype` (optional): Data type of your variable. Default is 'float32'

`deviceType` (optional): Set device either to 'cuda' or 'cpu'. Default is 'cuda'

`pinType` (optional): If device is set to 'cpu', you can specify using pinned memory. Default is 'False'. 


Methods:

`optInit`: Initializes the optimizer variable switcher. 

`beforeForwardPass(retrievedPosIndexes , retrievedNegIndexes = None)`: Switches optimizer variable weights from the full weights collection to optimizer weight tensor. `retrievedPosIndexes` is the indexes of the positive samples to be retrieved. If negative samples are to be retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

`afterOptimizerStep( retrievedPosIndexes , retrievedNegIndexes = None)`: Switches optimizer variable weights from your optimizer to the full weights collection. `retrievedPosIndexes` is the indexes of the positive samples that were retrieved. If negative samples were retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

## Citing SpeedTorch:
If you use SpeedTorch in your research or wish to cite, please cite with:

@misc{

  title={SpeedTorch},
  
  author={Santosh Gupta},
  
  howpublished={\url{github.com/Santosh-Gupta/SpeedTorch}},
  
  year={2019}
  
}
