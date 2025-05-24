# AI System

AI系统自顶向下依次包括：算法应用、开发体系、AI训练/推理框架、AI编译器及AI硬件架构。

## 算法应用（2优先级）

### Transformer的前世今生：从RNN到Transformer

RNN存在梯度消失和梯度爆炸的问题

GRU

LSTM能够有效地捕捉长时间依赖关系，缓解了梯度消失和梯度爆炸问题

Transformer基于注意力机制，不依赖于序列顺序处理，能够捕捉序列中任意位置的依赖关系

### Encoder-Decoder、Encoder-only与Decoder-only

Encoder-Decoder起源与机器翻译模型，在[Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](vscode-file://vscode-app/e:/Software/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)的序列到序列RNN中首次提出，主要解决机器翻译中输入输出序列不等长的问题。

**Encoder（编码器）** 主要负责接收输入序列，将其编码为一个固定长度的上下文向量（或一系列向量），**Decoder（解码器）** 则根据编码器输出的上下文向量逐步生成目标序列。

根据对Encoder和Decoder的解释，不难理解：Encoder-Only主要用于理解类任务（如分类、检索等），不涉及序列生成；而Decoder-Only主要用于生成类任务（如文本生成、对话、代码生成等）；最全面的Encoder-Decoder则更多应用于输入输出依赖较强的场景（如机器翻译、文本摘要等）。

三种网络结构的一些常见模型如下：

|    网络结构    |               典型模型               |
| :-------------: | :----------------------------------: |
| Encoder-Decoder |     Transformer、T5、mT5、Gemini     |
|  Encoder-Only  |      BERT、ConvBERT、StructBERT      |
|  Decoder-Only  | GPT、LLaMA、Qwen、Baichuan、DeepSeek |

### MoE

### 大模型应用

## 训练/推理框架（3优先级）

### 推理加速

#### 量化

#### 张量并行与流水线并行

#### KV Cache

对于Encoder-Decoder和Decoder-Only模型的生成类任务，在自回归生成时，每生成一个新token，理论上需要重新计算所有历史token的Key和Value，非常低效，因此通过将历史token的Key和Value向量缓存起来，生成新token时只需要计算新token的Key和Value，然后与缓存拼接即可，无需重复计算历史部分。

#### PD分离

在使用了KV Cache技术的情况下，对于Encoder-Decoder和Decoder-Only这两种网络结构的模型，其推理过程可以根据计算特性分为两个不同的阶段：Prefill和Decode。

* Prefill：在生成任务推理过程中，模型一次性处理全部输入上下文（如用户输入的prompt），计算出所有输入token的隐藏状态和注意力缓存（KV Cache）。这一阶段通常是并行计算的，能够高效地利用硬件资源。
* Decode：在Prefill阶段完成后，模型进入自回归生成阶段。此时模型每次只生成一个新token，并利用KV Cache只计算与新token相关的内容，从而避免重复计算历史token的表示。Decode阶段通常是串行的。

下面会从大模型推理常用的两个评估指标出发，分析讨论PD分离的收益。

在大模型推理中，通常使用两项指标来评估推理的性能：

* TTFT（Time to First Token）：生成第一个token的时间，主要衡量Prefill阶段的性能；
* TPOT（Time per Output Token）：返回第一个token后自回归生成每个token的的时间，主要衡量Decode阶段的性能。

这两个指标分别对应用户体验的输出延迟和输出吞吐量，并且分别由Prefill和Decode阶段的性能决定。

而两者的计算特性有所不同，Prefill阶段是计算密集型（属于Compute-Bound），Decode阶段是访存密集型（属于Memory-Bound），在batch策略和并行策略等优化措施上具有不同的表现：

首先是batch size，随着batch size的增大，计算密度也是增大的。但Prefill阶段通常可以并行计算，而Decode阶段的自回归生成通常是串行的，且需要频繁访问KV Cache，因此相同batch size下，Prefill计算密度远大于Decode，随着batch size的增大（即计算密度增大），Prefill阶段会率先超过屋檐曲线的脊点。具体表现为：Prefill阶段吞吐量随batch size增加逐渐趋于平稳（Compute-Bound），而Decode阶段吞吐量随batch size增加而显著提升（Memory-Bound）。

然后是并行策略，**TODO**

### 算子开发

## AI编译器（4优先级）

TODO

## AI Chip

### DSA存在的历史必然性

> 提升能耗-性能-成本的唯一途径就是专用。

* **内因：** 定制的领域专用体系结构可以拥有更高的性能、更低的功耗，并且需要更少的面积
* **外因：** 通用处理器的性能提升速度已经非常缓慢，领域专用体系结构的优势在很长一段时间里都不会因通用处理器而变得过时，甚至永远不会过时
  * 随着摩尔定律的放缓和登纳德缩放比例定律的终结，晶体管不再大幅改进，
  * 微处理器的功耗预算不变
  * ？？？
* 

### DSA设计指导原则

* 使用专用存储器将数据移动距离缩至最短
* 减少微体系结构高级优化措施，将所节省的资源投入到更多的算术运算单元或更大的存储器中
* 使用与该领域相匹配的最简并行形式
* 缩小数据规模，减少数据类型，能满足该领域的最低需求即可
* 使用一种领域专用编程语言将代码移植到DSA

### DSA设计流程

### 典型AI芯片架构（1优先级）

#### nVidia GPU

##### 硬件架构

采用SIMT（Single Instruction Multiple Threads）架构，适合大规模并行计算，整体由多个SM（Streaming Multiprocessor）组成，每个SM包含若干CUDA Core、Tensor Core和RT Core。

对于芯片间互联，英伟达在Pascal架构中引入NVLink，Volta架构中引入NVSwitch。

* NVLink是一种先进的总线及其通信协议，采用点对点结构、串行传输，既可以用于CPU与GPU之间的连接，也可以用于GPU之间互相连接。
* NVSwitch是一种高速互联技术，作为一块独立的NVSwitch芯片，在第四代NVSwitch（Hopper架构）中可以提供高达18路的NVLink接口。

##### 编程模型

GPU线程层次从上到下依次为grid、block和thread：

* 一个grid对应一个GPU kernel，由多个线程块block组成
* 一个block由一组线程组成，共享片上共享内存（Shared Memory），支持线程间同步
* thread是最小执行单位

GPU上每个SM可以调度和执行一个或多个block，但一个block只能在一个SM上执行，block会被分配到可用的SM上，SM内部负责调度block内的线程。

其中线程调度的基本单位是warp，每个warp包含32个thread，一个block会被划分为若干个warp，warp内的线程同步执行同一条指令（SIMT）。

GPU内存层次结构分为寄存器（Register）、共享内存（Shared Memory）和全局内存（Global Memory）。

* 寄存器：每个线程独享，速度最快
* 共享内存：线程块内共享，延迟低，适合数据交换与缓存
* 全局内存：容量大但延迟高，所有线程都可以访问

#### Google TPU

##### 硬件架构

TPU是为深度学习定制的一款专用AI加速器，主要优化矩阵乘法和卷积等操作。整体架构由标量单元（Scalar Unit）、向量单元（Vector Unit）和矩阵乘法单元（MXU，采用脉动矩阵乘法阵列）组成。

TPU Pod支持成百上千块TPU互联，

##### 编程模型

主要通过TensorFlow、JAX和Pytorch-XLA等高层API进行开发，底层细节由XLA和TPU硬件自动处理。

#### 昇腾NPU

##### 硬件架构

昇腾NPU的AI Core（达芬奇架构，也称为达芬奇Core）负责执行标量、向量和张量相关的计算密集型算子，整体架构由标量单元（Scalar Unit）、向量单元（Vector Unit）和矩阵计算单元（Cube Unit）组成，同时还包含存储单元和控制单元。

根据Cube单元和Vector单元是否同核部署还分为耦合架构和分离架构

* 耦合架构：Cube计算单元和Vector计算单元共享同一个Scalar单元，统一加载所有的代码段
* 分离架构：将AI Core拆成矩阵计算（AI Cube，AIC）和向量计算（AI Vector，AIV）两个独立的核，每个核都有自己的Scalar单元，能独立加载自己的代码段，从而实现矩阵计算与向量计算的解耦，在系统软件的统一调度下互相配合达到计算效率优化的效果。AIV与AIC之间通过Global Memory进行数据传递。

##### 编程模型

AscendC算子编程采用SPMD（Single Program Multiple Data）架构，具体而言就是将需要处理的数据拆分并同时在多个计算核心上运行，从而获取更高的性能。多个AI Core共享相同的指令代码，每个核上的运行实例唯一的区别是block_idx不同，每个核通过不同的block_idx来识别自己的身份。

AscendC编程范式把算子的实现流程分为3个基本任务：CopyIn、Compute、CopyOut，通过队列（Queue）完成任务间通信和同步，因此三个任务可以同步执行，并支持Ping-Pong Buffer优化。

### SIMD vs SIMT vs SPMD

SIMD vs SIMT：

* SIMD与SIMD的基本原理是相同的，都是采用单指令多数据的思想
* 但SIMT比SIMD更灵活，SIMD在读入数据时数据是物理连续的，而SIMT在不同线程中可以根据blockid和threadid独立寻址

SIMT vs SPMD：

* 英伟达SIMT和华为昇腾SPMD在整体形式上也基本相似，GPU中一个block对应到一个SM中执行，而昇腾NPU中一个block对应到一个AI Core中执行
* 但GPU一个SM中采用的是SIMT形式，而昇腾NPU一个AI Core中采用的是SIMD形式，非连续寻址由MTE（Memory Transfer Engine）在CopyIn和CopyOut阶段实现

# 可能的面试题

* TPU的脉动阵列如何高效执行矩阵乘法？与GPU的Tensor Core以及昇腾TPU的Cube Unit有何区别？
* 如何通过硬件设计支持混合精度计算（FP16/INT8）？量化误差如何缓解？

# 问题

* Pytorch是如何给NPU提供接口的，TPU是基于Tensorflow的软件定义架构？
* 从RNN到LSTM再到Transformer的发展历程
* RISC-V与AI DSA
