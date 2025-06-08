# Multi-Scale Dilated Convolution Network for Long-Term Time Series Forecasting 

Feifei Li ${ }^{1,2}$, Suhan Guo ${ }^{1,3}$, Feng Han ${ }^{1,2}$ and Jian Zhao ${ }^{4}$,* Furao Shen ${ }^{1,3}$, ${ }^{*}$<br>${ }^{1}$ National Key Laboratory for Novel Software Technology,Nanjing University,China<br>${ }^{2}$ Department of Computer Science and Technology,Nanjing University,China<br>${ }^{3}$ School of Artificial Intelligence,Nanjing University,China<br>${ }^{4}$ School of Electronic Science and Engineering,Nanjing University,China<br>\{feifeili, shguo,fenghan\}@smail.nju.edu.cn, \{jianzhao,frshen \}@nju.edu.cn


#### Abstract

Accurate forecasting of long-term time series has important applications for decision making and planning. However, it remains challenging to capture the long-term dependencies in time series data. To better extract long-term dependencies, We propose Multi-Scale Dilated Convolution Network (MSDCN), a method that utilizes a shallow dilated convolution architecture to capture the period and trend characteristics of long time series. We design different convolution blocks with exponentially growing dilations and varying kernel sizes to sample time series data at different scales. Furthermore, we utilize traditional autoregressive model to capture the linear relationships within the data. To validate the effectiveness of the proposed approach, we conduct experiments on eight challenging long-term time series forecasting benchmark datasets. The experimental results show that our approach outperforms the prior state-of-the-art approaches and shows significant inference speed improvements compared to several strong baseline methods.


## 1 Introduction

Time series forecasting is widely used in real life, including power consumption prediction [Pang et al., 2018], economy forecasting [Qin et al., 2017], weather prediction [Karevan and Suykens, 2020], traffic prediction [Wu et al., 2020], and so on. Among these prediction demands, it is often necessary to forecast over a longer time window to make timely decisions or give early warnings. Through the analysis and modeling of historical data, long-term time series forecasting enables us to gain insights and forecast future trends, cycles, and potential influencing factors. For instance, by forecasting power consumption, the power system can be efficiently managed to ensure smooth operations and meet the demands of both production and daily life. At the same time, the speed of forecasting is critical. For example, traffic flow forecasting needs to display the real-time road occupancy for the next few minutes, so as to optimize the allocation of road resources and alleviate congestion. Therefore, timely and accurate time series forecasting plays a crucial role in practical applications.
![](https://cdn.mathpix.com/cropped/2025_06_08_98537bb7d76efdb4eb8cg-1.jpg?height=283&width=906&top_left_y=772&top_left_x=1062)

Figure 1: A time series can be considered as the linear or nonlinear superposition of components such as weekly trends, monthly trends, and yearly trends.

Times series can be regarded as the linear or nonlinear superposition of components such as weekly trends, monthly trends, yearly trends and random noise, as depicted in Figure 1. The key to time series modeling is to efficiently extract these components and model subsequent tasks based on the extracted components. Some methods [Zeng et al., 2023; Wu et al., 2021] based on time series decomposition apply operations like moving averages and pooling to decompose time series, partially ignoring small but crucial information. Moreover, the extraction operations are inflexible, making it challenging to capture multi-scale information.

Traditional time series forecasting methods [Box et al., 2015; Taylor and Letham, 2018] mainly adopt functions fitting on sequence data. However, these traditional methods have difficulties in dealing with big data and complex temporal patterns. In addition, methods based on recurrent neural network [Qin et al., 2017; Shih et al., 2019; Salinas et al., 2020] are prone to the problem of vanishing gradient, which makes it difficult to effectively capture longterm information.

For long-term time series forecasting, there are three main types of methods using neural networks: Transformer-based methods, MLP-based methods and CNN-based methods. Transformer [Vaswani et al., 2017] is a commonly-used model for sequence modeling. Nevertheless, the permutation invariance property inherent in its self-attention mechanism poses challenges in capturing the temporal characteristics of time series data effectively [Wen et al., 2023]. This limitation hinders the extraction of precise temporal representations from time series data. Moreover, it is worth noting that Transformer-based models exhibit high time and space com-
plexity, which can be a concern in resource-constrained scenarios. Many methods [Li et al., 2019; Kitaev et al., 2019; Liu et al., 2022c; Liu et al., 2022b; Zeng et al., 2022] focus on improving the redundant self-attention mechanism and point-to-point connections in the transformer. MLP-based methods [Oreshkin et al., 2020; Challu et al., 2023] first decouple data into trend and remainder using moving average operations and then forecast them separately. However, the extracted patterns are fixed and relatively single. For example, in DLinear [Zeng et al., 2023], the kernel size of the moving average operation is fixed, and it can only extract one scale of feature representation. CNN-based methods [Lai et al., 2018; Wu et al., 2023] use ordinary convolution operations to extract local dependencies, which can only extract short-term local dependencies. Some methods [Wang et al., 2022] employ causal convolution [van den Oord et al., 2016] to extract long-term global dependencies, but these methods lack the flexibility to extract multi-scale time information effectively.

In summary, current methods still face two major challenges when dealing with long-term time series forecasting. Firstly, long-term time series data often exhibit complex periodic and trend changes, making it difficult for existing methods to capture both the short-term local dependency and the long-term global dependency. Secondly, existing methods have high time complexity and space complexity, which may pose limitations in terms of computing and storage resources in real-world application scenarios.

To solve the two challenges mentioned above, we propose a multi-scale dilated convolution shallow network (MSDCN). Dilated convolution downsamples the time series at different scales. To capture a broader range of scales, we expand the receptive field of the convolution network by using different kernel sizes and exponentially changing the dilation size. In addition, our model incorporates the traditional autoregressive module to directly model the linear relationship of data and facilitate complex modeling. The contributions of our paper are summarized as follows:

- We propose a novel approach that utilize the intrinsic characteristics of dilated convolution to expand the receptive field for time series data.
- We introduce a multi-scale feature fusion shallow structure, which consists of two levels of multi-scale features. These features involve an exponentially growing number of dilations and different kernel sizes.
- We leverage traditional autoregressive models to capture linear dependencies inherent in the data, effectively streamlining the non-linear modeling process.
- We conduct experiments to demonstrate the effectiveness of each module in our model. Our method achieves state-of-the-art performance. Furthermore, our model exhibits superior speed in both the training and inference phases.


## 2 Related Works

### 2.1 Long-term Dependency Information

Transformer-based methods focus on reducing quadratic time complexity and quadratic memory usage of the canonical
self-attention mechanism when modeling the long sequence time series. Informer [Zhou et al., 2021] proposes ProbSparse self-attention mechanism to calculate the most similar queries to participate in the attention computation through KL divergence, hence reducing unnecessary calculations. However, these methods are still restricted by the permutation invariance characteristics of the Vanilla Transformer [Vaswani et al., 2017]. The numerical values in time series data often lack semantics, and the crucial aspect of modeling is to capture the temporal changes between consecutive sequences of points. To preserve time information, Autoformer [Wu et al., 2021] proposes Auto-Correlation mechanism to replace self-attention mechanism and discover sequence-level dependencies through Fast Fourier transform.

CNN-based models use convolutional neural networks [LeCun et al., 1998] to extract both local and global patterns. LSTNet [Lai et al., 2018] employs CNNs to capture short-term local dependency patterns and uses RNNs [Elman, 1990; Memory, 2010; Chung et al., 2014] to capture long-term patterns. However, the design of the skip length $p$ in the model can only extract a single scale of temporal pattern and becomes a bottleneck for modeling. MICN [Wang et al., 2022] uses mutil-scale isometric convolution to extract local and global features. The isometric convolution is similar to TCN [Bai et al., 2018]. However, the modeling maybe restricted by causal convolution for time series forecasting since the input does not contain future sequences. To overcome the limitations of TCN, SCINet [Liu et al., 2022a] proposes downsample convolution and interaction architecture to extract rich features from multiple resolutions. TimesNet [Wu et al., 2023] uses a parameter-efficient inception block to extract complex temporal variations. However, those models usually have high space complexity and time complexity.

### 2.2 The Relationship Between Multiple Variables

Recently, many methods have studied how to model the relationship between multiple variables. LightTS [Zhang et al., 2022] uses a linear layer to do channel projection, and conduct experiments to verify a linear layer is sufficient to model the variable relationship. Crossformer [Zhang and Yan, 2023] proposes a Two-stage Attention Layer, which calculates attention in the time dimension and the space dimension separately. Due to the phenomenon of distribution drift in the time series, DLinear [Zeng et al., 2023] and TimesNet [Wu et al., 2023] independently model multiple variables, and also achieve good performance. Recently, [Han et al., 2023] conducts experiments and provides theoretical evidence that independent prediction can alleviate the issue of distribution drift. Inspired by those methods, we also adopt the channel independency strategy in our model to achieve more robust predictions.

## 3 Method

### 3.1 Problem Formulation

Long-term time series forecasting can be formally defined as follows: the input is

$$
\begin{equation*}
\mathbf{X}=\left\{X_{1}^{t}, X_{2}^{t}, \cdots, X_{C}^{t}\right\}_{t=1}^{T} \in \mathbb{R}^{T \times C}, \tag{1}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/2025_06_08_98537bb7d76efdb4eb8cg-3.jpg?height=635&width=1722&top_left_y=181&top_left_x=199)

Figure 2: The overall architecture of MSDCN. The green box represents convolution module, and the yellow box represents autoregressive module.
where $T$ is the lookback window size, $C$ is the number of variables, $X_{i}^{t}$ represents the value of the $i$ th variable at time $t$. The output is

$$
\begin{equation*}
\mathbf{Y}=\left\{X_{1}^{t}, X_{2}^{t}, \cdots, X_{C}^{t}\right\}_{t=T+1}^{T+L} \in \mathbb{R}^{L \times C}, \tag{2}
\end{equation*}
$$

where $L$ is the prediction window size, We use previous $T$ steps to predict future $L$ steps, where $L>T$ in our research problem. Our target is to learn a mapping function $\mathbf{f}$ to predict future time series, which is

$$
\begin{equation*}
\mathbf{Y}=\mathbf{f}(\mathbf{X}) . \tag{3}
\end{equation*}
$$

### 3.2 Overview

Originating from image processing, dilated convolution [Yu et al., 2017] aims to solve the problem of information loss caused by traditional convolution or pooling during downsampling process. By introducing gaps (dilations) between convolution kernels, dilated convolution enlarges the receptive field of the convolution operation, allowing it to better capture pixel information from more distant locations. This property enables the network to effectively acquire more details during downsampling, thus alleviating the problem of information loss.

This paper proposes a method that utilizes dilated convolution to extract multi-scale information from time series through downsampling. The dilation rates in different convolution kernels grow exponentially, providing a more flexible method to capture information from various scales of sequence components.

We introduce a multi-scale dilated convolution structure, as shown in Figure 2. Convolution module downsamples the input data $\mathbf{X}$ at various scales. Long and short convolutional module comprises multiple blocks, which enable the generation of multi-scale feature representations. We then employ different weights for feature fusion. The fused features are further processed by a feed forward neural network layer. The final prediction is obtained as the sum of the output from the feed forward layer and the autoregressive component. In
addition, following the NLinear model [Zeng et al., 2023], we add normalization operations before and after the model, which is not shown in Figure 2.

### 3.3 Convolution Module

We use two distinct convolutional modules with the same underlying structure but different convolution kernel sizes to downsample the input data at different scales. The long-term convolutional module employs a larger kernel size, representing a larger receptive field, while the short-term module utilizes a smaller kernel size, representing a smaller receptive field. Each convolutional module consists of multiple different 1D convolution blocks arranged in parallel. Specifically, assuming the long convolution module comprises $n$ convolution blocks, the short convolution module has $m$ convolution blocks. These blocks respectively produce

$$
\begin{equation*}
h_{i}=\operatorname{ConvBlock}_{i}(\mathbf{X}) \quad i=1, \cdots, n, n+1 \cdots, n+m, \tag{4}
\end{equation*}
$$

where $h_{i} \in \mathbb{R}^{C \times T}$ is the output of the $i$ th ConvBlock. More specifically, each convolution block is a sequence structure composed of dilated convolution, batch normalization and relu activation function, that is

$$
\begin{align*}
u_{i} & =\text { DilatedConv1d }(\text { Padding }(\mathbf{X})), \\
h_{i} & =\operatorname{ReLU}\left(\operatorname{BatchNorm1d}\left(u_{i}\right)\right) . \tag{5}
\end{align*}
$$

The one-dimensional convolution is a depthwise convolution, where the number of input channels and output channels are the number of variables in the datasets. At the same time, we use dilated convolution to extract long-term time information. The dilation size denotes that the convolution kernel observes at a fixed interval. Different dilation sizes represent different scales of feature extraction. To gradually expand the receptive field, the number of dilation sizes in the parallel one-dimensional convolutions increases by a factor of 2 , that is,

$$
\begin{equation*}
\text { dilation factors }=\left\{2^{0}+1,2^{1}+1, \ldots, 2^{n}+1\right\} . \tag{6}
\end{equation*}
$$

Then for different convolution blocks, the weights $W \in$ $\mathbb{R}^{C \times(n+m)}$ are learnt to get fusion feature representations $M$ from different scales, as shown in Equation 7,

$$
\begin{align*}
H & =\left[h_{1}, h_{2}, \cdots, h_{n+m}\right] \\
M & =\sum_{i}^{n+m} H_{i} \odot W_{i}^{\prime} \tag{7}
\end{align*}
$$

where $H \in \mathbb{R}^{C \times T \times(n+m)}$ is the stack of $h_{i}, W^{\prime}$ is a repeat of $W$ along $T$ dimension, $\odot$ denotes the element-wise product, and $M \in \mathbb{R}^{C \times T}$ is the fusion feature representation. Then the fusion feature representation goes through a layer of feed forward neural network that produces:

$$
\begin{equation*}
\hat{Y}_{c}=W_{1} M^{T}+b_{1}, \tag{8}
\end{equation*}
$$

where $W_{1} \in \mathbb{R}^{L \times T}$ denotes the linear weights, $b_{1} \in \mathbb{R}^{L}$ is a bias parameters, $\hat{Y}_{c}$ is the first part of final prediction.

### 3.4 Autoregressive Module

To enhance the learning of repetitive patterns commonly observed in time series data, our model incorporates an classical autoregressive prediction module, which follows a similar concept to ResNet [He et al., 2016]. Autoregressive model is a regression process concerning the variable itself, capable of capturing underlying linear dynamic relationships in time series.

$$
\begin{equation*}
Y_{t}=\sum_{i=1}^{p} \phi_{i} Y_{i-1}+\epsilon_{t} \tag{9}
\end{equation*}
$$

where $\phi_{i}$ are coefficients and $\epsilon_{t}$ is white noise error. In the MSDCN architecture, we use $W$ as the coefficients of the autoregressive model.

$$
\begin{equation*}
\hat{Y}_{h}=W_{2} \mathbf{X}+b_{2} \tag{10}
\end{equation*}
$$

where $W_{2} \in \mathbb{R}^{L \times T}$ is the coefficient of the autoregressive operation, $b_{2} \in \mathbb{R}^{L}$ is a bias parameter, $\hat{Y}_{h}$ denotes the second part of final prediction. Compared to the LightTS [Zhang et al., 2022] and LSTNet [Lai et al., 2018] models which also incorporate this module, our model shows a positive effect and generates more accurate predictions. Detailed analysis is given in Section 4.

Finally, the final prediction of our model is

$$
\begin{equation*}
\hat{Y}=\hat{Y}_{c}+\hat{Y}_{h} . \tag{11}
\end{equation*}
$$

where $\hat{Y}_{c}$ is given in Equation 8, $\hat{Y}_{h}$ is given in Equation 10.
To reduce the influence of abnormal outliers, We choose Huber Loss [Huber, 1964] as the loss function in the training process, as follows:

$$
\text { loss }= \begin{cases}0.5\left(y_{i}-\hat{y}_{i}\right)^{2}, & \text { if }\left|y_{i}-\hat{y}_{i}\right|<\delta  \tag{12}\\ \delta *\left(\left|y_{i}-\hat{y}_{i}\right|-0.5 * \delta\right), & \text { otherwise, }\end{cases}
$$

where $\delta$ is a positive and adjustable threshold.

Table 1: Experimental Datasets

| Datasets | Variables | Frequency | Observations |
| :--- | :--- | :--- | :--- |
| ETTm1 | 7 | 15 Minutes | 69,680 |
| ETTm2 | 7 | 15 Minutes | 69,680 |
| ETTh1 | 7 | 1 hour | 17,420 |
| ETTh2 | 7 | 1 hour | 17,420 |
| Illness | 7 | 1 week | 966 |
| Weather | 21 | 10 Minutes | 52,696 |
| Electricity | 321 | 1 hour | 26,304 |
| Traffic | 862 | 1 hour | 17,544 |

## 4 Experiments

### 4.1 Setup

Dataset We conduct experiments on eight benchmark datasets for time series forecasting, including (1) ETT datasets [Zhou et al., 2021] are collected from electricity transformers, including oil temperature and load. (2) Electricity dataset [Trindade, 2015] contains electricity consumption from 321 clients. (3) Illness dataset [CDC, 2021] contains influenza-like illness patients data. (4) Weather dataset [Wetterstation, 2021] comprises 21 weather indicators from the Weather Station of the Max Planck Biogeochemistry Institute. (5) Traffic dataset [California, 2017] contains the road occupancy rates from 862 sensors.

The characteristics of each dataset are shown in Table 1. However, we do not conduct experiments on Exchange-rate dataset, due to the reasons metioned in PatchTST [Nie et al., 2023]. To ensure the fairness of our experiments, we strictly follow the standard protocol for dataset splitting. Specifically, we divide all datasets into training, validation, and test sets in chronological order. For the ETT dataset, we use a split ratio of 6:2:2, while for other datasets, we adopt a ratio of $7: 1: 2$. For the illness dataset, the model's input length is 36 and its output length can be $24,36,48$ or 60 . For the other datasets, the model's input length is fixed at 96, and its output length can be $96,192,336$ or 720. To evaluate our model performance, we adopt two commonly used metrics: mean squared error (MSE) and mean absolute error (MAE). MSE and MAE are computed as:

$$
\begin{align*}
& M S E=\frac{1}{N} \sum_{j=1}^{N}\left(\hat{y}_{j}-y_{j}\right)^{2}  \tag{13}\\
& M A E=\frac{1}{N} \sum_{j=1}^{N}\left|\hat{y}_{j}-y_{j}\right| \tag{14}
\end{align*}
$$

where $N$ is the number of variables, $\hat{y}$ is the prediction and $y$ is the ground truth.

Baselines We compare our method with the following baselines:

- Transformer-based models: Informer [Zhou et al., 2021], Autoformer [Wu et al., 2021], ETSformer [Woo et al., 2022].
- MLP-based models: DLinear [Zeng et al., 2023], NLinear [Zeng et al., 2023], LightTS [Zhang et al., 2022].
- CNN-based models: TimesNet [Wu et al., 2023].

Table 2: Multivariate long-term forecasting results.

| Models |  | MSDCN |  | TimesNet |  | NLinear |  | DLinear |  | LightTS |  | ETSformer |  | Autoformer |  | Informer |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Metric |  | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE |
| ETTm2 | 96 | 0.174 | 0.256 | 0.187 | 0.267 | 0.182 | 0.265 | 0.193 | 0.292 | 0.209 | 0.308 | 0.189 | 0.280 | 0.255 | 0.339 | 0.365 | 0.453 |
|  | 192 | 0.239 | 0.298 | 0.249 | 0.309 | 0.246 | 0.304 | 0.284 | 0.362 | 0.311 | 0.382 | 0.253 | 0.319 | 0.281 | 0.340 | 0.533 | 0.563 |
|  | 336 | 0.296 | 0.332 | 0.321 | 0.351 | 0.306 | $\underline{0.341}$ | 0.369 | 0.427 | 0.442 | 0.466 | 0.314 | 0.357 | 0.339 | 0.372 | 1.363 | 0.887 |
|  | 720 | 0.395 | 0.392 | 0.408 | 0.403 | 0.408 | $\underline{0.398}$ | 0.554 | 0.522 | 0.675 | 0.587 | 0.414 | 0.413 | 0.433 | 0.432 | 3.379 | 1.338 |
| ETTh1 | 96 | 0.379 | 0.390 | 0.384 | 0.402 | 0.393 | 0.400 | 0.386 | $\underline{0.400}$ | 0.424 | 0.432 | 0.494 | 0.479 | 0.449 | 0.459 | 0.865 | 0.713 |
|  | 192 | 0.428 | 0.417 | $\underline{0.436}$ | $\underline{0.429}$ | 0.449 | 0.433 | 0.437 | 0.432 | 0.475 | 0.462 | 0.538 | 0.504 | 0.500 | 0.482 | 1.008 | 0.792 |
|  | 336 | 0.465 | 0.436 | 0.491 | 0.469 | 0.485 | $\underline{0.449}$ | 0.481 | 0.459 | 0.518 | 0.488 | 0.574 | 0.521 | 0.521 | 0.496 | 1.107 | 0.809 |
|  | 720 | 0.468 | 0.453 | 0.521 | 0.500 | 0.471 | 0.462 | 0.519 | 0.516 | 0.547 | 0.533 | 0.562 | 0.535 | 0.514 | 0.512 | 1.181 | 0.865 |
| ECL | 96 | 0.175 | 0.265 | 0.168 | $\underline{0.272}$ | 0.198 | 0.275 | 0.197 | 0.282 | 0.207 | 0.307 | 0.187 | 0.304 | 0.201 | 0.317 | 0.274 | 0.368 |
|  | 192 | 0.183 | 0.271 | 0.184 | 0.289 | 0.198 | $\underline{0.278}$ | 0.196 | 0.285 | 0.213 | 0.316 | 0.199 | 0.315 | 0.222 | 0.334 | 0.296 | 0.386 |
|  | 336 | $\underline{0.199}$ | 0.287 | 0.198 | 0.300 | 0.212 | $\underline{0.293}$ | 0.209 | 0.301 | 0.230 | 0.333 | 0.212 | 0.329 | 0.231 | 0.338 | 0.300 | 0.394 |
|  | 720 | $\underline{0.238}$ | 0.320 | 0.220 | 0.320 | 0.254 | 0.326 | 0.245 | 0.333 | 0.265 | 0.360 | 0.233 | 0.345 | 0.254 | 0.361 | 0.373 | 0.439 |
| Traffic | 96 | 0.619 | 0.366 | 0.593 | 0.321 | 0.647 | 0.388 | 0.650 | 0.396 | $\underline{0.615}$ | 0.391 | 0.607 | 0.392 | 0.613 | 0.388 | 0.719 | 0.391 |
|  | 192 | 0.577 | 0.342 | 0.617 | 0.336 | 0.600 | 0.364 | $\underline{0.598}$ | 0.370 | 0.601 | 0.382 | 0.621 | 0.399 | 0.616 | 0.382 | 0.696 | 0.379 |
|  | 336 | 0.591 | 0.348 | 0.629 | 0.336 | 0.607 | 0.367 | $\underline{0.605}$ | 0.373 | 0.613 | 0.386 | 0.622 | 0.396 | 0.622 | 0.337 | 0.777 | 0.420 |
|  | 720 | 0.630 | 0.365 | 0.640 | 0.350 | 0.645 | 0.387 | 0.645 | 0.394 | 0.658 | 0.407 | 0.632 | 0.396 | 0.660 | 0.408 | 0.864 | 0.472 |
| Weather | 96 | 0.169 | 0.217 | 0.172 | 0.220 | 0.202 | 0.240 | 0.196 | 0.255 | 0.182 | 0.242 | 0.197 | 0.281 | 0.266 | 0.336 | 0.300 | 0.384 |
|  | 192 | 0.215 | 0.259 | $\underline{0.219}$ | 0.261 | 0.248 | 0.277 | 0.237 | 0.296 | 0.227 | 0.287 | 0.237 | 0.312 | 0.307 | 0.367 | 0.598 | 0.544 |
|  | 336 | 0.269 | 0.299 | 0.280 | 0.306 | 0.300 | 0.313 | 0.283 | 0.335 | 0.282 | 0.334 | 0.298 | 0.353 | 0.359 | 0.395 | 0.578 | 0.523 |
|  | 720 | $\underline{0.350}$ | 0.352 | 0.365 | 0.359 | 0.373 | 0.360 | 0.345 | 0.381 | 0.352 | 0.386 | 0.352 | 0.388 | 0.419 | 0.428 | 1.059 | 0.741 |
| 브g | 24 | 2.222 | 0.936 | 2.317 | 0.934 | 2.662 | 1.054 | 2.398 | 1.040 | 8.313 | 2.144 | 2.527 | 1.020 | 3.483 | 1.287 | 5.764 | 1.677 |
|  | 36 | $\underline{2.192}$ | 0.915 | 1.972 | $\underline{0.920}$ | 2.487 | 1.040 | 2.646 | 1.088 | 6.631 | 1.902 | 2.615 | 1.007 | 3.103 | 1.148 | 4.755 | 1.467 |
|  | 48 | 2.164 | 0.938 | 2.238 | 0.940 | 2.406 | 1.024 | 2.614 | 1.086 | 7.299 | 1.982 | 2.359 | 0.972 | 2.669 | 1.085 | 4.763 | 1.469 |
|  | 60 | $\underline{2.287}$ | 0.946 | 2.027 | 0.928 | 2.475 | 1.037 | 2.804 | 1.146 | 7.283 | 1.985 | 2.487 | 1.016 | 2.770 | 1.125 | 5.264 | 1.564 |

Other Settings Our experiments are conducted on a NVIDIA GeForce GTX 1080 Ti. Moreover, experiments are implemented in PyTorch [Paszke et al., 2019] 1.10.1. Our method is trained with Huber Loss [Huber, 1964], using the Adam optimizer [Kingma and Ba, 2015].

### 4.2 Main Results

To show the effectiveness of our model on benchmark datasets, we conduct both multivariate and univariate longterm forecasting. The results are summarized as follows.

## Multivariate Long-term Forecasting

In order to present the results more clearly, we present the results of in Table 2, and full benchmark on ETT datasets are showned in supplementary materials. Our model achieves the best results in $80 \%$ of prediction results. Compared to the best CNN-based model, TimesNet, our model exhibits a reduction of $2.4 \%$ in the MSE metric and a reduction of $2.2 \%$ in the MAE metric. These results indicate that our design of shallow multi-scale 1D dilated convolutional modules outperforms deeper 2D ordinary convolution. Our method uses dilated convolution to downsample time series data at different scales, which can better capture multi-scale information.

When compared to the best transformer-based model, ETSformer, our model demonstrates a decrease of $9.4 \%$ in the MSE metric and a decrease of $11.0 \%$ in the MAE metric. After the TimesNet model, our model once again demonstrates that in long-term time series forecasting, the use of convolution networks can be more effective than transformer architectures. Furthermore, when compared to the best MLPbased model, NLinear, our model showcases a decrease of
$9.4 \%$ in the MSE metric and a decrease of $4.2 \%$ in the MAE metric. The NLinear model adopts an autoregressive structure with simple normalization operations before input and after output. Our model can be viewed as an extension of NLinear, incorporating our specifically designed convolutional modules. The results indicate that those convolutional modules enhance the autoregressive module expressive capabilities.

## Univariate Long-term Forecasting

In the convolutional modules, we utilize channel independence to model each variable separately. However, in the feed forward layer, multiple variables share the same model parameters. To validate the effectiveness of our model in singlevariable prediction, We select the Oil Temperature column in the ETT datasets as the univariate target. and the results are shown in Table 3. Following past studies, we set the lookback window to 336 . Compared to the NLinear model, our proposed method reduces the MSE by $4.8 \%$. Against the PatchTST/64 model, it decreases by 6.8\%, and compared to the TimesNet model, it drops by $12.9 \%$. These results shows that MSDCN achieves superior performance on univariate long sequences forecasting.

### 4.3 Ablation Study

## Convolutional Modules

We conduct experiments to analyze the effects of convolutional modules, as shown in Table 4. Without long and short convolutional modules denotes that we remove the entire convolutional modules and keep only the autoregressive module.

Table 3: Univariate long-term forecasting on the ETT datasets.

| Models |  | MSDCN |  | TimesNet |  | NLinear |  | DLinear |  | LightTS |  | ETSformer |  | Autoformer |  | Informer |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Metric |  | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE |
| ETTh1 | 96 | 0.052 | 0.177 | 0.059 | 0.189 | 0.055 | 0.179 | 0.056 | 0.183 | $\underline{0.053}$ | 0.177 | 0.056 | 0.180 | 0.071 | 0.206 | 0.193 | 0.377 |
|  | 192 | 0.067 | 0.202 | 0.074 | 0.215 | 0.071 | 0.205 | 0.072 | 0.208 | $\underline{0.069}$ | $\underline{0.204}$ | 0.071 | 0.204 | 0.114 | 0.262 | 0.217 | 0.395 |
|  | 336 | $\underline{0.078}$ | $\underline{0.225}$ | 0.076 | 0.220 | 0.081 | 0.225 | 0.083 | 0.225 | 0.081 | 0.226 | 0.098 | 0.244 | 0.107 | 0.258 | 0.202 | 0.381 |
|  | 720 | 0.077 | 0.220 | 0.087 | 0.236 | 0.087 | 0.232 | 0.089 | 0.236 | $\underline{0.080}$ | $\underline{0.226}$ | 0.189 | 0.359 | 0.126 | 0.283 | 0.183 | 0.355 |
| ETTh2 | 96 | 0.118 | 0.269 | 0.131 | 0.284 | 0.129 | 0.282 | 0.136 | 0.286 | $\underline{0.129}$ | $\underline{0.278}$ | 0.131 | 0.279 | 0.153 | 0.306 | 0.213 | 0.373 |
|  | 192 | 0.155 | 0.312 | 0.171 | 0.329 | $\underline{0.168}$ | 0.328 | 0.182 | 0.336 | 0.169 | $\underline{0.324}$ | 0.176 | 0.329 | 0.204 | 0.351 | 0.227 | 0.387 |
|  | 336 | $\underline{0.174}$ | 0.335 | 0.171 | $\underline{0.336}$ | 0.185 | 0.351 | 0.216 | 0.369 | 0.194 | 0.355 | 0.209 | 0.367 | 0.246 | 0.389 | 0.242 | 0.401 |
|  | 720 | 0.196 | 0.355 | $\underline{0.223}$ | $\underline{0.380}$ | 0.224 | 0.383 | 0.245 | 0.396 | 0.225 | 0.381 | 0.276 | 0.426 | 0.268 | 0.409 | 0.291 | 0.439 |
| ![](https://cdn.mathpix.com/cropped/2025_06_08_98537bb7d76efdb4eb8cg-6.jpg?height=89&width=34&top_left_y=638&top_left_x=285) | 96 | 0.026 | $\underline{0.122}$ | $\underline{0.026}$ | 0.123 | 0.026 | 0.121 | 0.029 | 0.127 | 0.026 | 0.122 | 0.028 | 0.123 | 0.056 | 0.183 | 0.109 | 0.277 |
|  | 192 | $\underline{0.040}$ | 0.151 | 0.040 | 0.151 | 0.039 | $\underline{0.150}$ | 0.046 | 0.162 | 0.039 | 0.149 | 0.045 | 0.156 | 0.081 | 0.216 | 0.151 | 0.310 |
|  | 336 | 0.052 | 0.175 | 0.053 | 0.174 | 0.053 | $\underline{0.173}$ | 0.060 | 0.188 | 0.052 | 0.172 | 0.061 | 0.182 | 0.076 | 0.218 | 0.427 | 0.591 |
|  | 720 | 0.072 | 0.208 | $\underline{0.073}$ | 0.206 | 0.074 | $\underline{0.207}$ | 0.081 | 0.219 | 0.073 | 0.207 | 0.080 | 0.210 | 0.110 | 0.267 | 0.438 | 0.586 |
| ETTm2 | 96 | 0.062 | 0.182 | 0.065 | 0.187 | 0.065 | 0.186 | 0.066 | 0.186 | $\underline{0.063}$ | $\underline{0.182}$ | 0.063 | 0.183 | 0.065 | 0.189 | 0.088 | 0.225 |
|  | 192 | 0.091 | 0.226 | 0.093 | 0.231 | 0.094 | 0.231 | 0.102 | 0.240 | 0.090 | 0.223 | 0.092 | 0.227 | 0.118 | 0.256 | 0.132 | 0.283 |
|  | 336 | 0.091 | 0.226 | 0.121 | 0.266 | 0.120 | 0.265 | 0.132 | 0.277 | $\underline{0.117}$ | $\underline{0.259}$ | 0.119 | 0.261 | 0.154 | 0.305 | 0.180 | 0.336 |
|  | 720 | 0.166 | $\underline{0.319}$ | 0.172 | 0.322 | 0.171 | 0.322 | 0.185 | 0.336 | $\underline{0.170}$ | 0.318 | 0.175 | 0.320 | 0.182 | 0.335 | 0.300 | 0.435 |

Table 4: Multivariate long-term forecasting with or without convolutional module.

| Convolution Module |  | Metric | ETTm2 |  |  |  |  | ECL |  |  |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| long | short |  | 96 | 192 | 336 | 720 | Avg | 96 | 192 | 336 | 720 | Avg |
| $\times$ |  | MSE MAE | \|0.182 | 0.246 | 0.306 | 0.408 | 0.286 | 0.198 | 0.198 | 0.212 | 0.254 | 0.216 |
|  | $\times$ |  | 0.265 | 0.304 | 0.341 | 0.398 | 0.327 | 0.275 | 0.278 | 0.293 | 0.326 | 0.293 |
| $\checkmark$ |  | MSE | 0.176 | 0.239 | 0.300 | 0.398 | 0.278 | 0.176 | 0.184 | 0.200 | 0.241 | 0.200 |
|  | $\times$ |  | 0.257 | 0.299 | 0.335 | 0.392 | 0.321 | 0.265 | 0.272 | 0.287 | 0.321 | 0.286 |
| $\times$ |  | MSE | 0.177 | 0.242 | 0.302 | 0.407 | 0.282 | 0.194 | 0.193 | 0.208 | 0.250 | 0.211 |
|  | $\checkmark$ |  | 0.256 | 0.300 | 0.336 | 0.399 | 0.323 | 0.271 | 0.274 | 0.289 | 0.322 | 0.289 |
| $\checkmark$ |  | MSE | 0.174 | 0.239 | 0.296 | 0.395 | 0.276 | 0.175 | 0.183 | 0.199 | 0.239 | 0.199 |
|  | $\checkmark$ | MAE | 0.256 | 0.298 | 0.332 | 0.392 | 0.320 | 0.265 | 0.271 | 0.287 | 0.320 | 0.286 |

The table demonstrates that incorporating both long and short convolution modules enhances prediction outcomes. For the ETTm2 dataset with a 15 -minute sampling interval, the impact on predictions is less noticeable due to the dataset's sensitivity to short-term factors. Conversely, for the Electricity dataset with a 1-hour sampling interval, solely adding the long convolution module yields notably better predictions than solely adding the short convolution module. This is attributed to the Electricity dataset being significantly influenced by long-term factors. In conclusion, introducing both modules simultaneously achieves optimal prediction performance.

## Autoregressive

To validate the impact of autoregressive module, we compare different architectures: MSDCN without convolution modules (only autoregressive module), MSDCN without autoregressive module (only convolution module), and the complete MSDCN, which integrates both modules. The results are shown in Figure 3.

The results show that using only the autoregressive module gives reasonable predictions. However, when we add the convolution module, the overall network performance improves. Specifically, we see a 7\% improvement in the Electricity dataset and a 3\% enhancement in the Traffic dataset. This highlights the autoregressive module's ability to catch linear
![](https://cdn.mathpix.com/cropped/2025_06_08_98537bb7d76efdb4eb8cg-6.jpg?height=695&width=873&top_left_y=902&top_left_x=1084)

Figure 3: Ablation results for the autoregressive module in MSDCN.
relationship, making predictions more accurate. The autoregressive is only a linear layer and cannot deal with complex temporal patterns. The introduced convolution module in MSDCN tackles this issue by capturing intricate patterns and trends in time series data, leading to better forecast accuracy. This confirms the effectiveness of our proposed method.

## 5 Analysis

### 5.1 Multi-Scale Representation

We visualize the outputs from various convolution blocks, as shown in Figure 4. The horizontal axis in the figure represents the time steps of the input time series, while the vertical axis represents the values after convolution. Each curve in the subfigures represents a different scale of convolution block representation, arranged from top to bottom. We use ReLU function in convolution blocks, so some negative values are suppressed. Blocks with missing IDs indicate that all values within those blocks are suppressed.
![](https://cdn.mathpix.com/cropped/2025_06_08_98537bb7d76efdb4eb8cg-7.jpg?height=494&width=871&top_left_y=173&top_left_x=172)

Figure 4: Visualization of different convolutional block output representation. The prediction length is 336 .
![](https://cdn.mathpix.com/cropped/2025_06_08_98537bb7d76efdb4eb8cg-7.jpg?height=708&width=863&top_left_y=806&top_left_x=176)

Figure 5: The MSE performance when input length is increasing. The prediction sequence length is 24 or 720 .

From Figure 4, we can observe that the different scales of convolution blocks can extract various curves with different shapes, indicating that the periods and trends of these curves are distinct, which demonstrates the effectiveness of our method. Therefore, our method exhibits effectiveness and robustness in handling diverse features of different scales in time series data.

### 5.2 Different Input Lengths On Prediction Performance

With a fixed output length, as the length of the input sequence increases, the model can access more information, extract additional temporal features, and make more effective predictions. A good model should maintain consistent prediction performance without a decline as the input sequence length increases.

As shown in Figure 5, as the input sequence length increases, models like TimesNet exhibit a decline in performance and suffer from overfitting, as they tend to learn more noise from the data rather than extracting meaningful tem-

Table 5: Comparison of practical efficiency of Models under input_len=96 and output_len=720 on the Traffic. Memory is one forward/backward pass size. MACs are the number of multiplyaccumulate operations. Time is the inference time.

| Model | Total Params | Memory | MACs | Time |
| :---: | :---: | :---: | :---: | :---: |
| MSDCN | $\underline{239.67 \mathrm{~K}}$ | $\underline{19.16 \mathrm{MB}}$ | $\underline{8.75 \mathrm{M}}$ | $\underline{3.68 \mathrm{~s}}$ |
| DLinear | $\mathbf{1 3 9 . 6 8 \mathrm { K }}$ | $\mathbf{1 3 . 4 8 M B}$ | $\mathbf{0 . 1 4 \mathrm { M }}$ | $\mathbf{1 . 5 0 s}$ |
| Autoformer | 14.91 M | 126.33 MB | 4.18 G | 62.39 s |
| PatchTST | 7.59 M | 800.96 MB | 12.69 G | 80.42 s |
| TimesNet | 301.75 M | 1628.31 MB | 1.23 T | 1491.58 s |

poral patterns from longer inputs. Our method outperforms models like DLinear when the input sequence length is relatively short and maintains its performance even with longer input sequences. Moreover, DLinear and NLinear demonstrate remarkably similar performance under most conditions. Our model exhibits superior performance when the input sequence is relatively short. As the length of input sequence increases, our model outperforms DLinear and NLinear by a smaller margin.

### 5.3 Efficiency Analysis

We compare our model with benchmark models based on the average practical efficiencies over 5 runs, as shown in Table 5. DLinear has only two linear layers, making it the fastest in terms of speed. Our model is second only to DLinear and is faster than models based on Transformer architecture.

More importantly, our model has significantly improved the inference speed compared to the CNN-based TimesNet model. TimesNet first transforms the data through an embedding layer and obtain representations through parameterefficient Inception blocks, resulting in a significant increase in its network parameters and computational complexity. Compared with TimesNet, our model does not have an embedding layer and use shallow convolution layer to extract representation, therefore, our model's inference time is faster than TimesNet model.

## 6 Conclusions

This paper presents a novel CNN-based neural network on long-term time series forecasting. To extract complex temporal dependencies in long-term time series, MSDCN can extract multi-scale temporal features by using different dilated convolution blocks. Extensive experiments showcase the effectiveness of MSDCN on prediction accuracy and time efficiency.

## References

[Bai et al., 2018] Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2018.
[Box et al., 2015] George EP Box, Gwilym M Jenkins, Gregory C Reinsel, and Greta M Ljung. Time series analysis: forecasting and control. 2015.
[California, 2017] California. Traffic, 2017.
[CDC, 2021] CDC. Illness, 2021.
[Challu et al., 2023] Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max MergenthalerCanseco, and Artur Dubrawski. N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, 2023.
[Chung et al., 2014] Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. In NIPS 2014 Workshop on Deep Learning, December 2014, 2014.
[Elman, 1990] Jeffrey L Elman. Finding structure in time. Cognitive science, 14(2):179-211, 1990.
[Han et al., 2023] Lu Han, Han-Jia Ye, and De-Chuan Zhan. The capacity and robustness trade-off: Revisiting the channel independent strategy for multivariate time series forecasting. arXiv preprint arXiv:2304.05206, 2023.
[He et al., 2016] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770-778, 2016.
[Huber, 1964] Peter J Huber. Robust estimation of a location parameter. The Annals of Mathematical Statistics, pages 73-101, 1964.
[Karevan and Suykens, 2020] Zahra Karevan and Johan AK Suykens. Transductive lstm for time-series prediction: An application to weather forecasting. Neural Networks, 125:1-9, 2020.
[Kingma and Ba, 2015] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations, 2015.
[Kitaev et al., 2019] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In International Conference on Learning Representations, 2019.
[Lai et al., 2018] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. In The 41st International ACM SIGIR Conference on Research \& Development in Information Retrieval, 2018.
[LeCun et al., 1998] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.
[Li et al., 2019] Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. In Advances in Neural Information Processing Systems, pages 52435253, 2019.
[Liu et al., 2022a] Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. Scinet:

Time series modeling and forecasting with sample convolution and interaction. In Advances in Neural Information Processing Systems, pages 5816-5828, 2022.
[Liu et al., 2022b] Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International Conference on Learning Representations, 2022.
[Liu et al., 2022c] Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Exploring the stationarity in time series forecasting. In Advances in Neural Information Processing Systems, pages 98819893, 2022.
[Memory, 2010] Long Short-Term Memory. Long shortterm memory. Neural computation, 9(8):1735-1780, 2010.
[Nie et al., 2023] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023.
[Oreshkin et al., 2020] Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-beats: Neural basis expansion analysis for interpretable time series forecasting. In International Conference on Learning Representations, 2020.
[Pang et al., 2018] Yue Pang, Bo Yao, Xiangdong Zhou, Yong Zhang, Yiming Xu, and Zijing Tan. Hierarchical electricity time series forecasting for integrating consumption patterns analysis and aggregation consistency. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, pages 3506-3512, 2018.
[Paszke et al., 2019] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. In Advances in neural information processing systems, 2019.
[Qin et al., 2017] Yao Qin, Dongjin Song, Haifeng Cheng, Wei Cheng, Guofei Jiang, and Garrison W Cottrell. A dual-stage attention-based recurrent neural network for time series prediction. In Proceedings of the 26th International Joint Conference on Artificial Intelligence, pages 2627-2633, 2017.
[Salinas et al., 2020] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36:1181-1191, 2020.
[Shih et al., 2019] Shun-Yao Shih, Fan-Keng Sun, and Hung-yi Lee. Temporal pattern attention for multivariate time series forecasting. Machine Learning, 108:14211441, 2019.
[Taylor and Letham, 2018] Sean J Taylor and Benjamin Letham. Forecasting at scale. The American Statistician, 72(1):37-45, 2018.
[Trindade, 2015] Artur Trindade. Electricity, 2015.
[van den Oord et al., 2016] Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. In 9th ISCA Speech Synthesis Workshop, pages 125-125, 2016.
[Vaswani et al., 2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 6000-6010, 2017.
[Wang et al., 2022] Huiqiang Wang, Jian Peng, Feihu Huang, Jince Wang, Junhui Chen, and Yifei Xiao. Micn: Multi-scale local and global context modeling for longterm series forecasting. In International Conference on Learning Representations, 2022.
[Wen et al., 2023] Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. Transformers in time series: A survey. In Proceedings of the 32th International Joint Conference on Artificial Intelligence, 2023.
[Wetterstation, 2021] Wetterstation. Weather, 2021.
[Woo et al., 2022] Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven Hoi. Etsformer: Exponential smoothing transformers for time-series forecasting. arXiv preprint arXiv:2202.01381, 2022.
[Wu et al., 2020] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, and Chengqi Zhang. Connecting the dots: Multivariate time series forecasting with graph neural networks. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery \& data mining, pages 753-763, 2020.
[Wu et al., 2021] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In Advances in Neural Information Processing Systems, pages 22419-22430, 2021.
[Wu et al., 2023] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. In International Conference on Learning Representations, 2023.
[Yu et al., 2017] Fisher Yu, Vladlen Koltun, and Thomas Funkhouser. Dilated residual networks. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 636-644. IEEE Computer Society, 2017.
[Zeng et al., 2022] Pengyu Zeng, Guoliang Hu, Xiaofeng Zhou, Shuai Li, Pengjie Liu, and Shurui Liu. Muformer: A long sequence time-series forecasting model based on modified multi-head attention. Knowledge-Based Systems, 254:109584, 2022.
[Zeng et al., 2023] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series fore-
casting? In Proceedings of the AAAI Conference on Artificial Intelligence, 2023.
[Zhang and Yan, 2023] Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In International Conference on Learning Representations, 2023.
[Zhang et al., 2022] Tianping Zhang, Yizhuo Zhang, Wei Cao, Jiang Bian, Xiaohan Yi, Shun Zheng, and Jian Li. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlp structures. arXiv preprint arXiv:2207.01186, 2022.
[Zhou et al., 2021] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1110611115, 2021.

