Notes to self learn about [efficient DL](https://hanlab.mit.edu/courses/2024-fall-65940). 
### Deep Learning 

**CNN Review**
- Linear layer
- Convolution layer
	- Padding
	- Receptive field
- Strided Convolution layer
- Grouped convolution layer
- Depthwise Convolution layer
- Pooling layer
- Normalization layer
- Activation Function

**Transformer**
- Encoding and decoding
- Attention:
	- $Softmax(\frac{QK^T}{\sqrt{d_k}}V)$

**Efficiency Metrics**
- Memory related: 
	- number of parameters 
	- model size: parameters * bit width
	- total / peak \# activations: memory bottleneck in CNN inference, not \#parameters
- Computation related: 
	- MAC (Multiply-Accumulate Operations):
		- $a \leftarrow a + b \cdot c$ 
		- Matrix-vector multiplication: MACs = $m\cdot n$
		- Matrix-matrix multiplication: MACs = $m\cdot n\cdot k$
	- FLOP, FLOPS (Floating point operations)
	- OP, OPS (Operations per second)
- Latency
- Throughput
- 
### Pruning


### Quantization


### Neural Architecture Search (NAS)

### Knowledge Distillation


### MCUNet: TinyML on Microcontrollers


### TinyEngine and Parallel Processing



---
### Transformer and LLM


### LLM Post Training


### Long Context LLM 


--- 
### Vision Transformer


### GAN, Video, and Point Cloud


### Diffusion Model 



---
### Distributed Training


### On-Device Training and Transfer Learning
---

### Quantum Machine Learning

