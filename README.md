# SpeedTorch
Library for fastest pinned CPU -> GPU transfer 

## What is it?

This library revovles around Cupy memmaps pinned to CPU, which can achieve _ % faster CPU -> GPU transfer than regular Pytorch Pinned CPU tensors can. 

## Inspiration?

I initially created this library to help train large numbers of embeddings, which the GPU may have trouble holding in RAM. In order to do this, I found that by hosting some of the embeddings on the CPU can help achieve this. Embedding systems use sprase training; only fraction of the total prameters participate in the forward/update steps, the rest are idle. So I figured, 'why not keep the idle parameters off the GPU during the training step?' For this I need fast CPU -> GPU transfer. 

## What can fast CPU->GPU do for me? (more that you might initially think)

With fast CPU->GPU, a lot of fun methods can be developed for functionalities which previously people thought may not have been possible. 

🏎️    Incoporage SpeedTorch into your data pipelines for data transfer to GPU

🏎️    Increase training speed of existing pipelines (my favorite trick with SpeedTorch, see below for details)

🏎️    Augment training parameters via CPU storage

🏎️    Use any optimizer you want for embeddings training (Adamax, RMSProp, etc.). Previously, only SpraseAdam, Adagrad, and SGD were suitable since they support sprase gradients. 

## How it works?

Somehow 

## Benchmarks

## Guide

## Examples

### Documentation 

```python
import arxiv
```
