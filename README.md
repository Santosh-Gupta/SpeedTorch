<p align="center">
  <img src="https://i.imgur.com/wr4VaUV.png?1">
</p>

# SpeedTorch

[![Join the chat at https://gitter.im/SpeedTorch/community](https://badges.gitter.im/SpeedTorch/community.svg)](https://gitter.im/SpeedTorch/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Library for fastest pinned CPU -> GPU Pytorch transfer 

## What is it?

This library revovles around Cupy memmaps pinned to CPU, which can achieve **110x** faster CPU -> GPU transfer than regular Pytorch Pinned CPU tensors can. 

## Inspiration

I initially created this library to help train large numbers of embeddings, which the GPU may have trouble holding in RAM. In order to do this, I found that by hosting some of the embeddings on the CPU can help achieve this. Embedding systems use sprase training; only fraction of the total prameters participate in the forward/update steps, the rest are idle. So I figured, why not keep the idle parameters off the GPU during the training step? For this, I needed fast CPU -> GPU transfer. 

## What can fast CPU->GPU do for me? (more that you might initially think)

With fast CPU->GPU, a lot of fun methods can be developed for functionalities which previously people thought may not have been possible. 

🏎️    Incorporate SpeedTorch into your data pipelines for fast data transfer to/from CPU <-> GPU

🏎️    Augment training parameters via CPU storage, by nearly double.

🏎️    Use Adadelta, Adamax, RMSprop, Rprop, ASGD, AdamW, and Adam optimizers for sparse embeddings training. Previously, only SpraseAdam, Adagrad, and SGD were suitable since only these directly support sprase gradients. 

<p align="center">
  <img src="https://i.imgur.com/6o8C1BP.gif">
</p>

## Benchmarks

### Speed

Here is a notebook comparing transfer via SpeedTorch vs Pytorch tensors, with both pinned CPU and Cuda tensors. All tests were done with a colab instance with a Tesla K80 GPU.

https://colab.research.google.com/drive/1b3QpfSETePo-J2TjyO6D2LgTCjVrT1lu

This notebook times the data transfer of 131,072 float32 embeddings of dimension 128, to and from the Cupy/Pytorch tensors and Pytorch variables.

The table below is a summary of the results. Transfering data from  Pytorch cuda tensors to the Cuda Pytorch embedding variable is faster than the SpeedTorch equiviliant, but for all other transfer types, SpeedTorch is faster. And for the sum of both steps transfering to/from the Cuda Pytorch embedding, SpeedTorch is faster than the Pytorch equivilant for both the regular GPU and CPU Pinned tensors. 

I have noticed that different instances of Colab result in different speed results, so keep this in mind while reviewing these results. A personal run of the colab notebook may result in different values, though the order of magnetude of the results are generally the same. 

The transfer times in the following tables are given in seconds. 

| Tensor Type	| To Cuda Pytorch Variable	| Comparison |
| --- | --- | --- |
| SpeedTorch(cuda)	| 0.0143	| 6.8x Slower than Pytorch Equivalent |
| SpeedTorch(PinnedCPU)	| 0.0254	| 4.0x Faster than Pytorch Equivalent |
| Pytorch(cuda)	| 0.0021	| 6.8x Faster than SpeedTorch Equivalent |
| Pytorch(PinnedCPU)	| 0.1012	| 4.0x Slower than SpeedTorch Equivalent |
		
| Tensor Type	| From Cuda Pytorch Variable	| Comparison |
| --- | --- | --- |
| SpeedTorch(cuda)	| 0.0036	| 12.3x Faster than Pytorch Equivalent |
| SpeedTorch(PinnedCPU)	| 0.0255	| 110.6x Faster than Pytorch Equivalent |
| Pytorch(cuda)	| 0.0442	| 12.3x Slower than SpeedTorch Equivalent |
| Pytorch(PinnedCPU)	| 2.8211	| 110.6x Slower than SpeedTorch Equivalent |
		
| Tensor Type	| Sum  of to/from Cuda Pytorch Variable	| Comparison |
| --- | --- | --- |
| SpeedTorch(cuda)	| 0.0179	| 2.6x Faster than Pytorch Equivalent |
| SpeedTorch(PinnedCPU)	| 0.0509	| 57.4x Faster than Pytorch Equivalent |
| Pytorch(cuda)	| 0.0463	| 2.6x Slower than SpeedTorch Equivalent |
| Pytorch(PinnedCPU)	| 2.9223	| 57.4x Slower than SpeedTorch Equivalent |

Similar benchmarks were calculated for transfering to/from Pytorch Cuda optimizers. The results are basically the same, here is the notebook used for the optimizers benchmarking

https://colab.research.google.com/drive/1Y2nehd8Xj-ixfjkj2QWuA_UjQjBBHhJ5

### Memory 

Although SpeedTorch's tensors are generally faster than Pytorch's, the drawback is SpeedTorch's tensors use more memory. However, because tranferring data can happen more quickly, you can use SpeedTorch to augment the number of embeddings trained in your architecture by holding parameters in both the GPU And CPU. 

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

## How it works?

I am not exactly sure why Cupy arrays are generally better for data transfer to/from Pytorch variables. As for how the memory management in Cupy works, I direct to these two stackoverflow questions I asked, where brilliant user Robert Crovella not only gave detailed explanations, but also figured out how to allocate pinned memory to Cupy arrays by developing his own memory allocator for Cupy. This is basically the core technology behind SpeedTorch. 

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

2) If you're able to fit all of your parameters in your GPU memory, use pure Pytorch since this is the fastest option for training. But if you can't fit all your parameters in memory, split your parameters (keep in mind that your optimizers also have weights) between SpeedTorch's Cupy cuda tensors and SpeedTorch's Cupy pinned CPU tensors; this is the 2nd fastest options. But, if you're still not able to fit all your parameters into memory that way, then split your parameters between SpeedTorch's Cupy pinned CPU tensors, and SpeedTorch's Pytorch cuda tensors; this is slower than both options, but uses less GPU memory.

For the 3rd option, here are two notebooks which shows an example of this https://colab.research.google.com/drive/1AqhT-HetihXMET1wJQROrC3Q9tFJqJ19 , https://colab.research.google.com/drive/1saKzsaHoy6O_U1DF_z15_Qkr5YLNI_GR

3) After training, saving any cuda variables will cause an increase in memory usage, and may cause a crash, especially with Cupy. If you're at the limits of your RAM. In this case, use the `getNumpyVersion` method to get a numpy version of your tensor, and then use use numpy.save or hdpy/pytables to save your numpy array. Numpy save is more lightweight. 

## Need Help?

Either open an issue, or chat with me directory on Gitter here https://gitter.im/SpeedTorch

## Future Work

I am looking incoporate more functionalities around the fast CPU -> GPU transfer. If you have an idea, please post a Github Issue. 

## Experimental

In addition the the Cupy GPU/pinned CPU and Pytorch GPU tensors, SpeedTorch also has Pytorch pinned CPU tensors, and Cupy memmap GPU/pinned CPU tensors. I have not found a solid use for these sorts of tensors, but they're fully coded and availible for use. 

https://github.com/Santosh-Gupta/SpeedTorch/tree/master/SpeedTorch

One area I would like to look at is if there is a way to have RAM memory reduction by using Cupy Memmaps. So far they uses just as much memory as the live versions. 

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

`given_optimizer`: The optimizer initialized with your model weights. 

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

Please see this notebook on how to use the data gadget

https://colab.research.google.com/drive/185Z5Gi62AZxh-EeMfrTtjqxEifHOBXxF

### Class GPUPytorchModelFactory

```python
GPUPytorchModelFactory(model_variable,  total_classes,  embed_dimension, datatype = 'float32')
```

Creates switchers for model variables using Pytorch cuda tensors. Switches variables from your full embedding collection and your model batch collection. Each variable needs its own switcher. 

Example:

```python
uEmbed_switcher = SpeedTorch.ModelFactory( skip_gram_modelSparse.u_embeddings, total_classes=50000, embed_dimension=128)
```

Arguments:

`model_variable`: Specific variable from your model you would like to create a switcher for.

`total_classes`: The total amount of embeddings to be trained. 

`embed_dimension`: Dimension of the embeddings.

`datatype` (optional): Datatype for the variable. Default is 'float32'. 


Methods:

`zerosInit()` : Initializes the variable switcher full collection with zeros:

`uniformDistributionInit(low, high)`: Initializes the variable switcher full collection with a uniform distribution from `low` to `high`

`normalDistributionInit(mean, stdDev)`: Initializes the variable switcher full collection with a normal distribution with a mean of `mean` and a standard deviation of `stdDev`

`customInit( initFunction, *args)`: Use any Pytorch initializer for the variable switchers full collection. Pass the initializer using `initFunction` and its corresponding arguments using `*args`.

`variableTransformer( batchSize, posPerBatch,  negPerBatch = None )`: Sets up a dummy input to be used for the forward step of you model. `batchSize` is the size of your batch, and `posPerBatch` is the number of positive examples per batch. If a second dummy input is needed for the negative examples, `negPerBatch` (optional) can be set to the number of negative examples, and two dummy inputs will be returned instead of one. 

`beforeForwardPass(retrievedPosIndexes , retrievedNegIndexes = None)`: Switches embeddings from the full embeddings collection to your model embeddings. `retrievedPosIndexes` is the indexes of the positive samples to be retrieved. If negative samples are to be retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

`afterOptimizerStep( retrievedPosIndexes , retrievedNegIndexes = None)`: Switches updated embeddings from your model to the full embeddings collection. `retrievedPosIndexes` is the indexes of the positive samples that were retrieved. If negative samples were retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

### Class GPUPytorchOptimizerFactory

```pyton
OptimizerFactory( given_optimizer,  total_classes,  embed_dimension, model, variable_name, dtype='float32')
```

Creates switchers for optimizer variables using Cupy. Switches variables from your full embedding collection and your optimizer batch collection. Each variable needs its own switcher. 

Example:

```python
uAdagrad_switcher = SpeedTorch.OptimizerFactory(given_optimizer,  total_classes,  embed_dimension, model, variable_name, dtype='float32', CPUPinn = False)
```

Arguments:

`given_optimizer`: The optimizer initialized with your model weights. 

`total_classes`: The total amount of embeddings to be trained. 

`embed_dimension`: Dimension of the embeddings.

`model`: The instance of your model. 

`variable_name`: Exact name of the variable defined in your model. 

`dtype` (optional): Data type of your variable. Default is 'float32'


Methods:

`optInit`: Initializes the optimizer variable switcher. 

`beforeForwardPass(retrievedPosIndexes , retrievedNegIndexes = None)`: Switches optimizer variable weights from the full weights collection to optimizer weight tensor. `retrievedPosIndexes` is the indexes of the positive samples to be retrieved. If negative samples are to be retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

`afterOptimizerStep( retrievedPosIndexes , retrievedNegIndexes = None)`: Switches optimizer variable weights from your optimizer to the full weights collection. `retrievedPosIndexes` is the indexes of the positive samples that were retrieved. If negative samples were retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

