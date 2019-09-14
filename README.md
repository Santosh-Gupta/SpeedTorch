


<p align="center">
  <img src="https://i.imgur.com/IjG1h2v.png">
</p>

# SpeedTorch

[![Join the chat at https://gitter.im/SpeedTorch/community](https://badges.gitter.im/SpeedTorch/community.svg)](https://gitter.im/SpeedTorch/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Library for fastest pinned CPU -> GPU Pytorch transfer 

## What is it?

This library revovles around Cupy memmaps pinned to CPU, which can achieve _ % faster CPU -> GPU transfer than regular Pytorch Pinned CPU tensors can. 

## Inspiration

I initially created this library to help train large numbers of embeddings, which the GPU may have trouble holding in RAM. In order to do this, I found that by hosting some of the embeddings on the CPU can help achieve this. Embedding systems use sprase training; only fraction of the total prameters participate in the forward/update steps, the rest are idle. So I figured, 'why not keep the idle parameters off the GPU during the training step?' For this I need fast CPU -> GPU transfer. 

## What can fast CPU->GPU do for me? (more that you might initially think)

With fast CPU->GPU, a lot of fun methods can be developed for functionalities which previously people thought may not have been possible. 

🏎️    Incorporate SpeedTorch into your data pipelines for data transfer to GPU

🏎️    Augment training parameters via CPU storage

🏎️    Use Adadelta, Adamax, RMSprop, Rprop, ASGD, AdamW, and Adam optimizers for embeddings training. Previously, only SpraseAdam, Adagrad, and SGD were suitable since they directly support sprase gradients. 

## Benchmarks

### Speed

Here is a notebook comparing transfer via Cupy with Pytorch tensors, with both pinned CPU and Cuda. 
https://colab.research.google.com/drive/1xPtFMt-Mdq9FVEx9UrV_arpXKZ96xh0s
This notebook times data transfer of 131,072 float32 embeddings of dimension 128, to and from the Cupy and Pytorch variables, each holding 1,000,000 float32 embeddings of dimension 128. 

The table below is a summary of the results. Transfering data from Cuda Pytorch tensors to the Cuda Pytorch embedding variable is faster than the SpeedTorch equiviliant, but for all other transfer types, SpeedTorch is faster. For the sum of both steps transfering to/from the Cuda Pytorch embedding, SpeedTorch is faster than the Pytorch equivilant, for both the regular GPU and CPU Pinned tensors. 

| Tensor Type	| To Cuda Pytorch Variable	| Comparison |
| --- | --- | --- |
| Cupy(cuda)	| 0.0104	| 5.6x slower than Pytorch Equivilent |
| Cupy(PinnedCPU)	| 0.0154	| 4.4x faster than Pytorch Equivilent |
| Pytorch(cuda)	| 0.0019	| 5.6x faster than SpeedTorch Equivilent |
| Pytorch(PinnedCPU)	| 0.0678	| 4.4x slower than SpeedTorch Equivilent |
		
| Tensor Type	| From Cuda Pytorch Variable	| Comparison |
| --- | --- | --- |
| Cupy(cuda)	| 0.0047	| 6.4x faster than Pytorch Equivilent |
| Cupy(PinnedCPU)	| 0.0078	| 370x faster than Pytorch Equivilent |
| Pytorch(cuda)	| 0.0302	| 6.4x slower than SpeedTorch Equivilent |
| Pytorch(PinnedCPU)	| 2.8842	| 370x slower than SpeedTorch Equivilent |
		
| Tensor Type	| Sum  of to/from Cuda Pytorch Variable	| Comparison |
| --- | --- | --- |
| Cupy(cuda)	| 0.0151	| 2.1x faster than Pytorch Equivilent |
| Cupy(PinnedCPU)	| 0.0232	| 127x faster than Pytroch Equivilent |
| Pytorch(cuda)	| 0.0321	| 2.1x slower than SpeedTorch Equivilent |
| Pytorch(PinnedCPU)	| 2.952	| 127x slower than SpeedTorch Equivilent |

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

This is the notebook I used for measuring how much memory each variable type takes. 
https://colab.research.google.com/drive/1ZKY7PyuPAIDrnx2HdtbujWo8JuY0XkuE
If using this in Colab, you will need to restart the enviroment after each tensor creation, to get a measure for the next tensor. 

## How it works?

Somehow 

Speed up existing numpy -> gpu pipelines. 
Sometimes it can be tricky to completly convert your pipeline from numpy. Though converting your numpy indexes to cuda mounted int64 pytorch variables during each of your training steps can add non-trivial time to your training. Luckily SpeedTorch has a solution to speed up your training while keeping your existing pipelines. 

Cupy memaps can accept int32 numpy indexes, so you can just use SpeedTorch's variable switchers to just switch in and out embeddings during every training step, and you can just use a static dummy variable for inputs for each training step. The last sentence probably won't make sense the first time reading it ( I'm still working on reducing the learning curve for this library), but check out this example. And come to library's Gitter with your questions. 


## Guide

### Using SpeedTorch to use non-sparse optimizers (in this case, Adamax) in Word2vec

For people first trying to figure out how to use SpeedTorch, I recommend following this example, since word2vec is one of the more commonly known algorithms in machine learning. 

https://colab.research.google.com/drive/1ApJR3onbgQWM3FBcBKMvwaGXIDXlDXOt

The notebook shows how to train word2vec the regular way, then shows how to use SpeedTorch to train on the same data, using one of the optimizers normally not supported for sparse training. 

### Augment training parameters via CPU storage

In sparse training algorithms like word2vec, GloVe, or Neural Collaborative Filtering, only a fraction of the total parameters (embeddngs) are trained during every step. If your GPU can not handle all of your embeddings at a desired embedding size, an option would be to host some of your parameters on pinned CPU Cupy arrays, and transfer those parameters to your model tensors as needed. This has allowed me to increase the size of the embeddings I have used in two projects. 

## Examples

Applying SpeedTorch to word2vec

https://colab.research.google.com/drive/1cYb6f3DD1FP2PVSZaC8Jz8uP3BgoR7oe

## Need Help?

Either open an issue, or chat with me directory on Gitter here https://gitter.im/SpeedTorch

### Documentation 

## Class ModelFactory

```python
ModelFactory(model_variable,  total_classes,  embed_dimension, datatype = 'float32', CPUPinn = False)
```

Creates switchers for model variables. Switches variables from your full embedding collection and your model batch collection. Each variable needs its own switcher. 

Example:

```python
uEmbed_switcher = SpeedTorch.ModelFactory( skip_gram_modelSparse.u_embeddings, total_classes=50000, embed_dimension=128)
```

Arguments:

`model_variable`: Specific variable from your model you would like to create a switcher for.

`total_classes`: The total amount of embeddings to be trained. 

`embed_dimension`: Dimension of the embeddings.

`diskname` (optional): Name for how the variable is stored onto disk. 

`datatype` (optional): Datatype for the variable. Default is 'float32'. 

`CPUPinn` (optional): Pin your full embedding collection to CPU. Spares GPU memory, but data transfer will be slower. Default is False. 

Methods:

`zerosInit()` : Initializes the variable switcher full collection with zeros:

`uniformDistributionInit(low, high)`: Initializes the variable switcher full collection with a uniform distribution from `low` to `high`

`normalDistributionInit(mean, stdDev)`: Initializes the variable switcher full collection with a normal distribution with a mean of `mean` and a standard deviation of `stdDev`

`variableTransformer( batchSize, posPerBatch,  negPerBatch = None )`: Sets up a dummy input to be used for the forward step of you model. `batchSize` is the size of your batch, and `posPerBatch` is the number of positive examples per batch. If a second dummy input is needed for the negative examples, `negPerBatch` (optional) can be set to the number of negative examples, and two dummy inputs will be returned instead of one. 

`beforeForwardPass(retrievedPosIndexes , retrievedNegIndexes = None)`: Switches embeddings from the full embeddings collection to your model embeddings. `retrievedPosIndexes` is the indexes of the positive samples to be retrieved. If negative samples are to be retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

`afterOptimizerStep( retrievedPosIndexes , retrievedNegIndexes = None)`: Switches updated embeddings from your model to the full embeddings collection. `retrievedPosIndexes` is the indexes of the positive samples that were retrieved. If negative samples were retrieved as well, a value for `retrievedNegIndexes` (optional) can be passed as well. 

## Class OptimizerFactory

```pyton
OptimizerFactory( given_optimizer,  total_classes,  embed_dimension, model, variable_name, dtype='float32' , CPUPinn = False)
```

Creates switchers for optimizer variables. Switches variables from your full embedding collection and your optimizer batch collection. Each variable needs its own switcher. 

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
