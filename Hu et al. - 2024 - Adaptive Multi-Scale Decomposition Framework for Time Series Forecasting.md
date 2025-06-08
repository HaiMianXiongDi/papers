# Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting 

Yifan Hu ${ }^{1, *}$ Peiyuan Liu ${ }^{3, *}$ Peng Zhu ${ }^{1}$ Dawei Cheng ${ }^{1, \otimes}$ Tao Dai ${ }^{2}$<br>${ }^{1}$ Tongji University ${ }^{2}$ Shenzhen University<br>${ }^{3}$ Tsinghua Shenzhen International Graduate School<br>\{pengzhu, dcheng\}@tongji.edu.cn<br>\{huyf0122, peiyuanliu.edu, daitao.edu\}@gmail.com


#### Abstract

Transformer-based and MLP-based methods have emerged as leading approaches in time series forecasting (TSF). While Transformer-based methods excel in capturing long-range dependencies, they suffer from high computational complexities and tend to overfit. Conversely, MLP-based methods offer computational efficiency and adeptness in modeling temporal dynamics, but they struggle with capturing complex temporal patterns effectively. To address these challenges, we propose a novel MLP-based Adaptive Multi-Scale Decomposition (AMD) framework for TSF. Our framework decomposes time series into distinct temporal patterns at multiple scales, leveraging the Multi-Scale Decomposable Mixing (MDM) block to dissect and aggregate these patterns in a residual manner. Complemented by the Dual Dependency Interaction (DDI) block and the Adaptive Multi-predictor Synthesis (AMS) block, our approach effectively models both temporal and channel dependencies and utilizes autocorrelation to refine multi-scale data integration. Comprehensive experiments demonstrate that our AMD framework not only overcomes the limitations of existing methods but also consistently achieves state-of-the-art performance in both long-term and short-term forecasting tasks across various datasets, showcasing superior efficiency. Code is available at https://github.com/TROUBADOUROOO/AMD.


## 1 Introduction

Time series forecasting (TSF) aims to use historical data to predict future values across various domains, such as finance [1, 2], energy [3], traffic management [4], and weather forecasting [5]. Recently, deep learning has made substantial and reliable advancements in TSF, with the most state-of-the-art performances achieved by Transformer-based methods [6-11] and MLP-based methods[1216].

Transformer-based methods excel at modeling long-range dependencies due to the self-attention mechanisms [17]. Although effective, they come with computational complexity that scales quadratically with the length of the sequence. Additionally, self-attention can diminish the temporal relationships when extracting semantic correlations between pairs in long sequences [13], leading to an overemphasis on mutation points and resulting in overfitting (see Fig. 1a). In contrast, MLP-based methods boast significantly lower computational complexity compared to Transformer-based methods. Moreover, MLP-based methods can chronologically model the temporal dynamics in consecutive points, which is crucial for time series analysis [13, 12, 16]. However, the simplicity of linear mappings in existing MLP-based methods presents an information bottleneck [18, 19], hindering their ability to capture diverse temporal patterns and limiting their predictive accuracy.

[^0]![](https://cdn.mathpix.com/cropped/2025_05_28_9ee9a9cc6faa08a54792g-02.jpg?height=548&width=1397&top_left_y=241&top_left_x=369)

Figure 1: (a) Illustration of multi-scale temporal patterns in time series and the impact of selector weights. Transformer-based methods often overfit by overemphasizing mutation points, weakening temporal relationships. Efficiently modeling and integrating distinct temporal patterns at various scales is crucial for accurate predictions. (b) Memory usage (MB), training time (ms/iter), and MSE comparisons on the Weather dataset. The input and predicted lengths are set to 512 and 96, respectively. Our proposed AMD achieves a low MSE of 0.145 with 17 ms training time and 1349 MB memory usage, demonstrating high efficiency and effectiveness.

It is worth noting that time series exhibit distinctly different temporal patterns at various sampling scales [15]. Moreover, the weight of these time scales in predicting future variations is not uniform, as future variations are jointly determined by the entangle of multiple scales (see Fig. 1a). For example, weather data sampled hourly reflects fine-grained, sudden changes, while monthly sampled data captures coarse-grained climate variations. Similarly, while short-term hourly data might highlight immediate weather shifts, long-term monthly data provides a broader view of climatic trends. Therefore, efficiently modeling the multi-scale changes in time series and adeptly integrating information across different scales remains a critical challenge.

Motivated by the above observations, we decompose the time series at multiple scales to precisely discern the intertwined temporal patterns within the complex series, rather than merely breaking it down into seasonal and trend components [7, 13]. Subsequently, we model the correlations across different scales in both time and channel dimensions. To account for the varying impacts of different temporal patterns on the future, we employ an autocorrelation approach to model their contributions and adaptively integrate these multi-scale temporal patterns based on their respective influences.
Technically, we propose an MLP-based Adaptive Multi-Scale Decomposition (AMD) Framework to better disentangle and model the diverse temporal patterns within time series. In concrete, the AMD initiates by employing the Multi-Scale Decomposable Mixing (MDM) block, which first decomposes the original time series into multiple temporal patterns through average downsampling and then aggregates these scales to provide aggregate information in a residual way. Subsequently, the Dual Dependency Interaction (DDI) block simultaneously models both temporal and channel dependencies within the aggregated information. Finally, the Adaptive Multi-predictor Synthesis (AMS) block uses the aggregated information to generate specific weights and then employs these weights to adaptively integrate the multiple temporal patterns produced by the DDI. Through comprehensive experimentation, our AMD consistently achieves state-of-the-art performance in both long-term and short-term forecasting tasks, with superior efficiency (see Fig. 1b) across multiple datasets.
Our contributions are summarized as follows: (i) We decompose time series across multiple scales to precisely identify intertwined temporal patterns within complex sequences and adaptively aggregate predictions of temporal patterns at different scales, addressing their varied impacts on future forecasts. We also demonstrate the feasibility through theoretical analysis. (ii) We propose a simple but effective MLP-based Adaptive Multi-Scale Decomposition (AMD) framework that initially decomposes time series into diverse temporal patterns, models both temporal and channel dependencies of these patterns, and finally synthesizes the outputs using a weighted aggregation approach to focus on the changes of dominant temporal patterns, thereby enhancing prediction accuracy across scales. (iii) Comprehensive experiments demonstrate that our AMD consistently delivers state-of-the-art
performance in both long-term and short-term forecasting across various datasets, with superior efficiency.

## 2 Related Works

### 2.1 Time Series Forecasting

Time series forecasting (TSF) involves predicting future values based on past observations in sequential order. In recent years, deep learning methods have gained prominence for their ability to automatically extract intricate patterns and dependencies from data, such as CNN [20-22], RNN [23, 24], GNN [25-27], Transformer [7, 6, 10, 28] and MLP [12, 13, 29]. Transformerbased models, renowned for superior performance in handling long and intricate sequential data, have gained popularity in TSF. Autoformer [7] introduces a decomposition architecture and autocorrelation mechanism. PatchTST [6] divides the input time series into patches to enhance the locality of temporal data. In addition to capturing cross-time dependencies, Crossformer [9] also mines cross-variable dependencies to leverage information from associated series in other dimensions. However, Transformer-based models always suffer from efficiency problems due to high computational complexity. In contrast, MLP-based models are more efficient and have a smaller memory footprint. For example, DLinear [13] utilizes the series decomposition as a pre-processing before linear regression and outperforms all of the previous Transformer-based models. FITS [30] proposes a new linear mapping for transformation of complex inputs, with only 10k parameters. However, due to the inherent simplicity and information bottleneck, MLP-based models struggle to effectively capture diverse temporal patterns [19]. In this work, we decompose time series across multiple scales and model the information at each scale with separate linear models, effectively addressing the representational limitations of MLP-based methods.

### 2.2 Series Decomposition in TSF

Recently, with high sampling rates leading to high-frequency data (such as daily, hourly, or minutely data), real-world time series data often contains multiple underlying temporal patterns. To competently harness different temporal patterns at various scales, several series decomposition designs are proposed [31-33, 15]. Seasonal Extraction in ARIMA Time Series [34] offers theoretical assurances but is limited to a monthly scale. Seasonal-Trend decomposition [35] is based on moving average and aims to disentangle the seasonal and trend components. FEDformer [8] incorporate frequency information to enhance the series decomposition block. TimesNet [21] decomposes time series data into multiple periods by Fast Fourier Transform, thus several multiple dominant frequencies. SCINet [36] utilizes a hierarchical downsampling tree to iteratively extract and exchange multi-scale information. Following the aforementioned designs, this paper proposes a new Multi-Scale Decomposition method to decompose the time series at multiple scales to discern the intertwined temporal patterns within the complex series precisely.

## 3 Preliminary: Linear Models with Multi-Scale Information for TSF

We consider the following problem: given a collection of time series samples with historical observations $\mathbf{X} \in \mathbb{R}^{C \times L}$, where $C$ denotes the number of variables and $L$ represents the length of the look-back sequence. The objective is to predict $\mathbf{Y} \in \mathbb{R}^{M \times T}$, where $M$ is the number of target variables to be predicted ( $M \leq C$ ) and $T$ is the length of the future time steps to be predicted. A linear model learns parameters $\mathbf{A} \in \mathbb{R}^{L \times T}$ and $\mathbf{b} \in \mathbb{R}^{T}$ to predict the values of next $T$ time steps as:

$$
\begin{equation*}
\hat{\mathbf{Y}}=\mathbf{X} \mathbf{A} \oplus \mathbf{b} \in \mathbb{R}^{C \times L} \tag{1}
\end{equation*}
$$

where $\oplus$ means column-wise addition. The corresponding $M$ rows in $\hat{\mathbf{Y}}$ can be used to predict $\mathbf{Y}$.
After that, we introduce the multi-scale information. For time series forecasting, the most influential real-world applications typically exhibit either smooth or periodicity. Without these characteristics, predictability tends to be low, rendering predictive models unreliable. If the time series only exhibits periodicity, linear models can easily model it [37]. We define the original sequence as $f(x)=f_{0}(x)=\left[x_{1}, x_{2}, \ldots, x_{L}\right]$ and assume that $f(x)$ possesses smoothness. After $k$ downsampling operations with a downsampling rate of $d$, we obtain $n$ sequences
$f_{i}(x), \forall i=1,2, \ldots, n$, where $f_{i}(x)=\frac{1}{d} \sum_{j=x d-d+1}^{x d} f_{i-1}(j)$ and $x=1,2, \ldots,\left[\frac{L}{d^{2}}\right]$. It is noteworthy that $f_{i}(x) \in \mathbb{R}^{C \times\left[\frac{L}{d^{i}}\right]}, \forall i=0,1, \ldots, n$. Then, the sequence $f_{i}(x), \forall i=0,1, \ldots, n$ is transformed into $g_{i}(x)$ through linear mapping and residual calculation. Specifically, $g_{n}(x)=f_{n}(x)$, then through top-down recursion for $i=n-1, \ldots, 0$, the operation

$$
\begin{equation*}
g_{i}(x)=f_{i}(x)+g_{i+1}(x) W_{i} \tag{2}
\end{equation*}
$$

is performed recursively, where $W_{i} \in \mathbb{R}^{\left[\frac{L}{d^{i+1}}\right] \times\left[\frac{L}{d^{i}}\right]}$. In this case, we derive the Theorem 1 (the proof is available in Appendix A):
Theorem 1. Let multi-scale mixing representation $g(x)$, where $g(x) \in \mathbb{R}^{1 \times L}$ (for simplicity, we consider univariate sequences) and the original sequence $f(x)$ is Lipschitz smooth with constant $\mathcal{K}$ (i.e. $\left|\frac{f(a)-f(b)}{a-b}\right| \leq \mathcal{K}$ ), then there exists a linear model such that $\left|y_{t}-\hat{y_{t}}\right|$ is bounded, $\forall t=1, \ldots, T$.

This derivation demonstrates that linear models are well-suited to utilize multi-scale temporal patterns in history. For nonperiodic patterns provided they exhibit smoothness, which is often observed in practice, the error remains bounded.

## 4 Method

As shown in Fig. 2, our proposed AMD mainly consists of three components: Multi-Scale Decomposable Mixing (MDM) Block, Dual Dependency Interaction (DDI) Block, and Adaptive Multi-predictor Synthesis (AMS) Block. Specifically, for the length- $L$ history inputs with $C$ recorded channels, MDM initially decomposes the $\mathbf{X} \in \mathbb{R}^{C \times L}$ into multi-scale temporal patterns through average downsampling and subsequently integrates these patterns residually to yield aggregate information $\mathbf{U} \in \mathbb{R}^{C \times L}$. Next, $\mathbf{U}$ is transformed into patches $\hat{\mathbf{U}} \in \mathbb{R}^{C \times N \times P}$, where the length of each patch is $P$ and the number of patches is $N$. The DDI block then takes $\hat{\mathbf{U}}$ as input, concurrently modeling both temporal and channel dependencies, and outputs $\mathbf{V} \in \mathbb{R}^{C \times L}$. Following this, AMS decomposes each channel $\mathbf{u} \in \mathbb{R}^{1 \times L}$ of $\mathbf{U}$ from MDM, deriving scores for different temporal patterns and calculating the corresponding weights $\mathbf{S} \in \mathbb{R}^{T \times m}$, where $T$ is the length of the prediction series and $m$ is the number of predictors. It also uses the split outputs $\mathbf{v} \in \mathbb{R}^{1 \times L}$ from each channel of $\mathbf{V}$ as inputs for the predictors. These weights, along with the predictor outputs, are then merged through a mixture-of-expert structure to generate the final prediction results. The details of each essential module are explained in the following subsections and the pseudocode is shown in Appendix B.

### 4.1 Multi-Scale Decomposable Mixing Block

Time series exhibit both coarse-grained temporal patterns and fine-grained patterns. These two types of information influence each other by dynamically interacting: coarse-grained patterns reflect a macroscopic context that shapes fine-grained patterns, while fine-grained patterns offer microscopic feedback that refines and adjusts the coarse-grained patterns [38]. Together, these complementary scales of information provide a comprehensive view of the time series. Therefore, we first decompose the time series into individual temporal patterns, and then mix them to enhance the time series data for a more nuanced analysis and interpretation.
Specifically, the raw input information $\mathbf{X}$ already contains fine-grained details, while coarse-grained information is extracted through average pooling. First-level temporal pattern $\boldsymbol{\tau}_{1}$ is the input of one channel $\mathbf{x}$. Next, distinct coarse-grained temporal patterns $\boldsymbol{\tau}_{i} \in \mathbb{R}^{1 \times\left\lfloor\frac{L}{d^{i-1}}\right\rfloor}(\forall i \in 2, \ldots, h)$ are extracted by applying average pooling over the previous layer of temporal patterns, where $h$ denotes the number of downsampling operations and $d$ denotes the downsampling rate. The decomposition of the $i^{t h}$ layer of temporal patterns can be represented as:

$$
\begin{equation*}
\boldsymbol{\tau}_{i}=\operatorname{AvgPooling}\left(\boldsymbol{\tau}_{i-1}\right) \tag{3}
\end{equation*}
$$

Then, distinct temporal patterns are mixed from the coarse-grained $\boldsymbol{\tau}_{k}$ to the fine-grained $\boldsymbol{\tau}_{1}$ through a feedforward residual network. The mixing of the $i^{\text {th }}$ layer of temporal patterns can be represented by the following formula:

$$
\begin{equation*}
\boldsymbol{\tau}_{i}=\boldsymbol{\tau}_{i}+\operatorname{MLP}\left(\boldsymbol{\tau}_{i+1}\right) \tag{4}
\end{equation*}
$$

Finally, after completing the mixing of temporal patterns across $h$ scales, we obtain mixed-scale information $\boldsymbol{\tau}_{1}$, with the output of one channel being $\mathbf{u}=\boldsymbol{\tau}_{1} \in \mathbb{R}^{1 \times L}$.
![](https://cdn.mathpix.com/cropped/2025_05_28_9ee9a9cc6faa08a54792g-05.jpg?height=503&width=1373&top_left_y=250&top_left_x=376)

Figure 2: AMD is an MLP-based model, consisting of the Multi-Scale Decomposable Mixing (MDM) Block, the Dual Dependency Interaction (DDI) Block, and the Adaptive Multi-predictor Synthesis (AMS) Block.

### 4.2 Dual Dependency Interaction Block

Modeling each scale of information separately could ignore the relationships between different scales. In reality, different scales interact with each other, for example in a stock price series where monthly economic trends influence daily fluctuations, and these monthly trends are influenced by annual market cycles.

To model both temporal and channel dependencies between different scales, we propose the DDI block, which first stacks the aggregated information $\mathbf{u}$ from various channels of the MDM into the matrix $\mathbf{U} \in \mathbb{R}^{C \times L}$ and then performs patch operations to transform $\mathbf{U}$ into $\hat{\mathbf{U}} \in \mathbb{R}^{C \times N \times P}$. For each patch, $\hat{\mathbf{V}}_{t}^{t+P}$ represents the embedding output of the residual network, and $\hat{\mathbf{U}}_{t}^{t+P}$ represents a patch of aggregated information from MDM. We adopt a residual block to aggregate historical information to obtain temporal dependency $\mathbf{Z}_{t}^{t+P}$. Next, we perform the transpose operation to fuse cross-channel information through another residual block. Finally, we perform the unpatch operation and split the output information into individual channels to obtain $\mathbf{v} \in \mathbb{R}^{1 \times L}$. The interaction of the patch $\hat{\mathbf{U}}_{t}^{t+P}$ can be represented by the following formula:

$$
\begin{gather*}
\mathbf{Z}_{t}^{t+P}=\hat{\mathbf{U}}_{t}^{t+P}+\operatorname{MLP}\left(\hat{\mathbf{V}}_{t-P}^{t}\right)  \tag{5}\\
\hat{\mathbf{V}}_{t}^{t+P}=\mathbf{Z}_{t}^{t+P}+\beta \cdot \operatorname{MLP}\left(\left(\mathbf{Z}_{t}^{t+P}\right)^{T}\right)^{T} \tag{6}
\end{gather*}
$$

where $A^{T}$ is the transpose of matrix A. Finally, by performing the unpatch operation, we obtain the output $\mathbf{V} \in \mathbb{R}^{C \times L}$.
In DDI, dual dependencies are captured under the mixed-scale information U. Temporal dependencies model the interactions across different periods, while cross-channel dependencies model the relationships between different variables, enhancing the representation of the time series. However, through experiments, we find that cross-channel dependencies are not always effective; instead, they often introduce unwanted interference. Therefore, we introduce the scaling rate $\beta$ to suppress the noise.

### 4.3 Adaptive Multi-predictor Synthesis Block

We utilize Mixture-of-Experts (MoE) in AWS due to its adaptive properties, allowing us to design different predictors for different temporal patterns to improve both accuracy and generalization capability. The AMS is partitioned into two components: the temporal pattern selector (TP-Selector) and the temporal pattern projection (TP-Projection). The TP-Selector decomposes different temporal patterns, scores them, and generates the selector weights $\mathbf{S}$. Unlike the downsampling in MDM, which decomposes individual scales to enhance temporal information, the TP-Selector adaptively untangles highly correlated, intertwined mixed scales through feedforward processing. Meanwhile, the TP-Projection synthesizes the multi-predictions and adaptively aggregates the outputs based on the specific weights.

TP-Selector takes a single channel input u from MDM, decomposes it through feedforward layers, and then applies a noisy gating design [39]:

$$
\begin{gather*}
\mathbf{S}=\operatorname{Softmax}(\operatorname{TopK}(\operatorname{Softmax}(Q(\mathbf{u})), k))  \tag{7}\\
Q(\mathbf{u})=\operatorname{Decompose}(\mathbf{u})+\psi \cdot \operatorname{Softplus}\left(\operatorname{Decompose}(\mathbf{u}) \cdot \mathbf{W}_{\text {noise }}\right) \tag{8}
\end{gather*}
$$

where $k$ is the number of dominant temporal patterns, $\psi \in \mathbb{N}(0,1)$ is standard Gaussian noise, and $\mathbf{W}_{\text {noise }} \in \mathbb{R}^{m \times m}$ is a learnable weight controlling the noisy values. The sum of the selector weights $\mathbf{S}$ for each channel is 1 .

TP-Projection takes the embedding $\mathbf{v}$ from the output $\mathbf{V}$ of DDI as input. It utilizes a two-layer feedforward network as a predictor. These predictions are then multiplied by the selector weights $\mathbf{S}$ and summed to yield the prediction result $\hat{\mathbf{y}}$ for one channel. The final results $\hat{\mathbf{Y}}$ are composed of the outputs $\hat{\mathbf{y}}$ from each channel:

$$
\begin{equation*}
\hat{\mathbf{y}}=\sum_{j=0}^{m} \mathbf{S}_{j} \cdot \operatorname{Predictor}_{j}(\mathbf{v}) \tag{9}
\end{equation*}
$$

Compared to sparse MoE [40], we adopt dense MoE in TP-Projection with three considerations. Firstly, each temporal pattern contributes to the prediction result. Secondly, we take the k-th largest value instead of the k -th largest value and avoid the sorting operation by employing the divide-andconquer method, thus reducing the time complexity from $O(n \log (n))$ to $O(n)$. Thirdly, this approach can help mitigate issues such as load unbalancing and embedding omissions that are prevalent in sparse MoE architectures. Our TopK method can be formalized with:

$$
\operatorname{TopK}(\mathbf{u}, k)= \begin{cases}\alpha \cdot \log (\mathbf{u}+1), & \text { if } \mathbf{u}<v_{k}  \tag{10}\\ \alpha \cdot \exp (\mathbf{u})-1, & \text { if } \mathbf{u} \geq v_{k}\end{cases}
$$

where $v_{k}$ is the k-th largest value among $\mathbf{u}$ and $\alpha$ is a constant used to adjust the selector weights. The scaling operation within our TopK requires the input values to be restricted to the interval $[0,1]$. Consequently, we perform an additional Softmax operation according to Eq. (7).

### 4.4 Loss Function

The loss function of AMS consists of two components. (1) For the predictors, the Mean Squared Error (MSE) loss $\left(\mathcal{L}_{\text {pred }}=\sum_{i=0}^{T}\left\|\mathbf{y}_{\mathbf{i}}-\hat{\mathbf{y}}_{\mathbf{i}}\right\|_{2}^{2}\right)$ is used to measure the variance between predicted values and ground truth. (2) For the gating network loss, we apply the coefficient of variation loss function $\left(\mathcal{L}_{\text {selector }}=\frac{\operatorname{Var}(S)}{\operatorname{Mean}(S)^{2}+\epsilon}\right.$, where $\epsilon$ is a small positive constant to prevent numerical instability), which optimizes the gating mechanism by promoting a balanced assignment of experts to inputs, thereby enhancing the overall performance of the MoE model. The total loss function is defined as:

$$
\begin{equation*}
\mathcal{L}=\mathcal{L}_{\text {pred }}+\lambda_{1} \mathcal{L}_{\text {selector }}+\lambda_{2}\|\Theta\|_{2} \tag{11}
\end{equation*}
$$

where $\|\boldsymbol{\Theta}\|_{2}$ is the L2-norm of all model parameters, $\lambda_{1}$ and $\lambda_{2}$ are hyper-parameters.

## 5 Experiments

### 5.1 Main Results

We thoroughly evaluate the proposed model in various time series forecasting tasks and confirm the generality of the proposed framework. Additionally, we offer insights into the effectiveness of integrating MoE components and the low computational complexity.

Datasets. We conduct experiments on seven real-world datasets, including Weather, ETT (ETTh1, ETTh2, ETTm1, ETTm2), ECL, Exchange, Traffic and Solar Energy for long-term forecasting and PEMS (PEMS03, PEMS04, PEMS07, PEMS08) for short-term forecasting. A detailed description of each dataset is provided in Appendix C.1.

Baselines. We carefully select some representative models to serve as baselines in the field of time series forecasting, including (1) MLP-based models: TiDE [12], MTS-Mixers [16], and DLinear [13]; (2) Transformer-based models: PatchTST [6], Crossformer [9], and FEDformer [8]; (3) CNN-based models: TimesNet [21], and MICN [20]. See Appendix C. 2 for a detailed description of the baseline.

Experimental Settings. To ensure fair comparisons, for long-term forecasting, we re-run all the baselines with different input lengths $L$ and choose the best results to avoid underestimating the baselines and provide a fairer comparison. For short-term forecasting, we choose an input length of 96. To trade off the memory footprint and accuracy, the number of predictors $n$ is set to 8 . We select two common metrics in time series forecasting: Mean Absolute Error (MAE) and Mean Squared Error (MSE). All experiments are conducted using PyTorch on an NVIDIA V100 32GB GPU and repeat five times for consistency. See Appendix C. 4 for detailed parameter settings.

Results. Comprehensive forecasting results are shown in Tab. 1 and Tab. 2, which present longterm and short-term forecasting results respectively. The best results are highlighted in red and the second-best are underlined. The lower the MSE/MAE, the more accurate the forecast. AMD stands out with the best performance in 50 cases and the second best in 27 cases out of the overall 80 cases. Compared with other baselines, AMD performs well on both high-dimensional and low-dimensional datasets. It is worth noting that PatchTST does not perform well on the PEMS dataset, possibly because the patching design leads to the neglect of highly fluctuating temporal patterns. In contrast, AMD leverages information from multi-scale temporal patterns. Furthermore, the performance of Crossformer is unsatisfactory, possibly because it introduces unnecessary noise by exploring cross-variable dependencies. AMD skillfully reveals the intricate dependencies existing among time steps across various variables. Additionally, DLinear and MTS-mixers perform poorly on high-dimensional datasets, whereas AMD can handle them. Moreover, as shown in Fig. 3a, in most cases, the forecasting performance benefits from the increase of input length, which is also observed in the majority of other datasets.

Table 1: Long-term forecasting task. All the results are averaged from 4 different prediction lengths $T \in\{96,192,336,720\}$. To the best for a fairer comparison for all baselines, the input sequence length $L$ is searched among $\{96,192,336,512,672,720\}$. See Appendix D. 1 for the full results.

| Models | AMD (Ours) |  | PatchTST [6] |  | Crossformer [9] |  | FEDformer [8] |  | TimesNet [21] |  | MICN [20] |  | DLinear [13] |  | MTS-Mixers [16] |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Metric | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE |
| Weather | 0.223 | 0.262 | $\underline{0.226}$ | $\underline{0.264}$ | 0.230 | 0.290 | 0.310 | 0.357 | 0.259 | 0.287 | 0.242 | 0.298 | 0.240 | 0.300 | 0.235 | 0.272 |
| ETTh1 | 0.407 | 0.424 | 0.413 | 0.431 | 0.441 | 0.465 | 0.428 | 0.454 | 0.458 | 0.450 | 0.433 | 0.462 | 0.423 | 0.437 | 0.430 | 0.436 |
| ETTh2 | 0.350 | 0.392 | 0.330 | 0.379 | 0.835 | 0.676 | 0.388 | 0.434 | 0.414 | 0.427 | 0.385 | 0.430 | 0.431 | 0.447 | 0.386 | 0.413 |
| ETTm1 | 0.347 | 0.375 | 0.351 | 0.381 | 0.431 | 0.443 | 0.382 | 0.422 | 0.400 | 0.406 | 0.383 | 0.406 | 0.357 | $\underline{0.379}$ | 0.370 | 0.395 |
| ETTm2 | 0.253 | 0.315 | $\underline{0.255}$ | 0.315 | 0.632 | 0.578 | 0.292 | 0.343 | 0.291 | 0.333 | 0.277 | 0.336 | 0.267 | 0.332 | 0.277 | $\underline{0.325}$ |
| ECL | 0.159 | $\underline{0.254}$ | 0.159 | 0.253 | 0.293 | 0.351 | 0.207 | 0.321 | 0.192 | 0.295 | 0.182 | 0.297 | 0.177 | 0.274 | $\underline{0.173}$ | 0.272 |
| Exchange | 0.328 | 0.387 | 0.387 | 0.419 | 0.701 | 0.633 | 0.478 | 0.478 | 0.416 | 0.443 | $\underline{0.315}$ | 0.404 | 0.297 | 0.378 | 0.373 | 0.407 |
| Traffic | $\underline{0.393}$ | $\underline{0.272}$ | 0.391 | 0.264 | 0.535 | 0.300 | 0.604 | 0.372 | 0.620 | 0.336 | 0.535 | 0.312 | 0.434 | 0.295 | 0.494 | 0.354 |
| Solar-Energy | 0.192 | 0.240 | 0.256 | 0.298 | $\underline{0.204}$ | $\underline{0.248}$ | 0.243 | 0.350 | 0.244 | 0.334 | 0.213 | 0.266 | 0.329 | 0.400 | 0.315 | 0.363 |

Table 2: Short-term forecasting task. The input sequence length $L$ is 96, while the prediction length $T$ is 12 for all baselines.

| Models Metric | AMD (Ours) |  | PatchTST [6] |  | Crossformer [9] |  | FEDformer [8] |  | TimesNet [21] |  | TiDE [12] |  | DLinear [13] |  | MTS-Mixers [16] |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE |
| PEMS03 | 0.084 | $\underline{0.198}$ | 0.099 | 0.216 | 0.090 | 0.203 | 0.126 | 0.251 | 0.085 | 0.192 | 0.178 | 0.305 | 0.122 | 0.243 | 0.117 | 0.232 |
| PEMS04 | 0.083 | $\underline{0.198}$ | 0.105 | 0.224 | 0.098 | 0.218 | 0.138 | 0.262 | 0.087 | 0.195 | 0.219 | 0.340 | 0.148 | 0.272 | 0.129 | 0.267 |
| PEMS07 | 0.074 | 0.180 | 0.095 | 0.150 | 0.094 | 0.200 | 0.109 | 0.225 | $\underline{0.082}$ | $\underline{0.181}$ | 0.173 | 0.304 | 0.115 | 0.242 | 0.134 | 0.278 |
| PEMS08 | 0.093 | 0.206 | 0.168 | 0.232 | 0.165 | 0.214 | 0.173 | 0.273 | $\underline{0.112}$ | $\underline{0.212}$ | 0.227 | 0.343 | 0.154 | 0.276 | 0.186 | 0.286 |

### 5.2 Ablation Study and Analysis

Observation 1: AMS learn from different temporal patterns. As shown in Tab. 3, we demonstrate the performance improvement attributed to AMD is not solely due to the enlarged model size,
![](https://cdn.mathpix.com/cropped/2025_05_28_9ee9a9cc6faa08a54792g-08.jpg?height=419&width=1394&top_left_y=243&top_left_x=368)

Figure 3: (a) The performance on Weather dataset of AMD improves as the input length increases, indicating our model can effectively extract useful information from longer history and capture long-term multi-scale dependency. (b) Cross-channel dependencies may lead to deviations from the original distribution.
but rather the integration of temporal pattern information, we devised the following experiments: (1) The permutation-invariant nature of the self-attention mechanism leads to temporal information loss, whereas the MLP-based model inherently maintains the sequential order of data. To demonstrate the utilization of sequential information, we replaced the temporal pattern embedding with random orders as RandomOrder. (2) To demonstrate the effectiveness of AMS aggregation, we replaced the TP-Selector with the average weighting as AverageWeight, treating different temporal patterns equally.
Consistent with our observation, RandomOrder and AverageWeight produce losses higher than AMD in most cases. Compared to self-attention, whose permutation-invariant nature leads to the loss of sequential information, AMS makes better use of temporal relations. Compared to simple averaging, AMS adaptively assigns corresponding weights to different temporal patterns, resulting in more accurate predictions.
On top of that, AMD benefits from improved interpretability. We plotted selector weights as shown in Fig. 4a. Before and after the time step T, the temporal variations are respectively dominated by TP16 and TP5. Before T, the predicted data resembles the trend of TP16, both exhibiting a downward fluctuation. However, after T, the predicted data resembles TP5, which suddenly shows a significant increase. AMD recognizes this changing dominant role over time and adaptively assigns them higher weights.

Observation 2: Cross-channel dependencies are not always effective. To prove cross-channel dependencies often introduce unwanted noise, we conducted another ablation study on the Weather dataset as shown in Tab. 3. We introduce cross-channel dependencies by adjusting the scaling rate. Specifically, we set the parameter $\beta$ in Eq. (7) to 0.5 and 1.0, respectively. In addition, we conduct experiments by removing components (w/o) DDI. From the results, it can be seen that the introduction of cross-channel dependencies do not enhance the prediction accuracy, and this was consistent across other datasets as well.
To explain this phenomenon, we visualize the learned dependencies as shown in Fig. 3b. Compared to the temporal dependency, especially when the target variable is not correlated with other covariates, cross-channel dependencies tend to smooth out the variability in the target variable, causing its distribution to deviate from what would be expected based solely on its own past values.

Table 3: Component ablation of AMD on Weather and ECL.

| Models Metric |  | AMD(Ours) |  | RandomOrder |  | AverageWeight |  | w/o DDI |  | $\beta=0.5$ |  | $\beta=1.0$ |  | w/o $\boldsymbol{L}_{\text {selector }}$ |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE |
| ![](https://cdn.mathpix.com/cropped/2025_05_28_9ee9a9cc6faa08a54792g-08.jpg?height=85&width=27&top_left_y=2281&top_left_x=401) | 96 | 0.145 | 0.198 | 0.152 | 0.209 | 0.149 | 0.205 | 0.154 | 0.208 | 0.146 | 0.198 | 0.147 | 0.200 | 0.160 | 0.211 |
|  | 192 | 0.190 | 0.240 | 0.194 | 0.245 | 0.194 | 0.246 | 0.197 | 0.248 | 0.194 | 0.245 | 0.195 | 0.246 | 0.209 | 0.258 |
|  | 336 | 0.242 | 0.280 | 0.245 | 0.286 | 0.249 | 0.288 | 0.249 | 0.289 | 0.249 | 0.287 | 0.248 | 0.286 | 0.278 | 0.309 |
|  | 720 | 0.315 | 0.332 | 0.321 | 0.338 | 0.325 | 0.340 | 0.323 | 0.340 | 0.318 | 0.335 | 0.320 | 0.336 | 0.364 | 0.366 |
| Uㅓ | 96 | 0.129 | 0.225 | 0.133 | 0.230 | 0.135 | 0.234 | 0.132 | 0.231 | 0.140 | 0.242 | 0.147 | 0.248 | 0.139 | 0.238 |
|  | 192 | 0.148 | 0.242 | 0.155 | 0.248 | 0.155 | 0.251 | 0.151 | 0.247 | 0.169 | 0.268 | 0.175 | 0.273 | 0.156 | 0.456 |
|  | 336 | 0.164 | 0.258 | 0.171 | 0.264 | 0.169 | 0.263 | 0.167 | 0.266 | 0.174 | 0.274 | 0.183 | 0.285 | 0.177 | 0.271 |
|  | 720 | 0.195 | 0.289 | 0.202 | 0.294 | 0.202 | 0.295 | 0.200 | 0.294 | 0.207 | 0.298 | 0.215 | 0.308 | 0.208 | 0.299 |

![](https://cdn.mathpix.com/cropped/2025_05_28_9ee9a9cc6faa08a54792g-09.jpg?height=462&width=1407&top_left_y=241&top_left_x=359)

Figure 4: (a) AMD guides the prediction by assigning greater weight to the dominant time pattern. (b) Memory usage (MB), training time (ms/iter), and MSE comparisons on the ETTh1 dataset. The input and predicted lengths are set to 512 and 96, respectively. $n$ denotes the number of predictors.

Observation 3: TP-Selector helps not only generalization but also efficiency. We thoroughly compare the training time and memory usage of various baselines on the ETTh1 dataset, using the official model configurations and the same batch size. The results, shown in Fig. 4b, indicate that AMD demonstrates superior efficiency with a relatively small number of parameters.

Observation 4: The balance of the TP-Selector is essential. We conduct experiments on the scaling rate of the load balancing loss, denoted by $\lambda_{1}$ in Eq. (11). As shown in Tab. 3, utilizing $\mathcal{L}_{\text {selector }}$ results in significantly improved performance, exceeding $11.2 \%$ in terms of MSE compared to when $\lambda_{1}=0.0$. This underscores the crucial role of implementing load-balancing losses. Furthermore, the selector weights in the Fig. 4a do not tend to favor specific temporal patterns, addressing the load imbalance issue in sparse MoE structures.

Observation 5: AMD improves other TSF methods as a plugin. Finally, we explore whether our proposed MoE-based method can yield an improvement in the performance of other TSF methods. We selected DLinear [13] and MTS-Mixers [16] as the baselines. After integrating the MDM and AMS modules, the predictive capabilities of all these models are enhanced as shown in Tab. 4, while maintaining the same computational resource requirements.

Table 4: Comparative impact of MDM \& AMS on different baselines. Imp. represents the average percentage improvement of MDM \& AMS compared to the original methods.

| Models Metric |  | DLinear [13] |  | + MDM \& AMS |  | MTS-Mixers [16] |  | + MDM \& AMS |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE |
| Weather | 96 | 0.152 | 0.237 | 0.146 | 0.212 | 0.156 | 0.206 | 0.155 | 0.203 |
|  | 192 | 0.220 | 0.282 | 0.194 | 0.261 | 0.199 | 0.248 | 0.202 | 0.251 |
|  | 336 | 0.265 | 0.319 | 0.245 | 0.305 | 0.249 | 0.291 | 0.244 | 0.286 |
|  | 720 | 0.323 | 0.362 | 0.313 | 0.356 | 0.336 | 0.343 | 0.326 | 0.337 |
| Imp. |  | - | - | 6.46\% | 5.50\% | - | - | 1.38\% | 1.01\% |
| ECL | 96 | 0.153 | 0.237 | 0.150 | 0.244 | 0.141 | 0.243 | 0.137 | 0.239 |
|  | 192 | 0.152 | 0.249 | 0.159 | 0.256 | 0.163 | 0.261 | 0.160 | 0.258 |
|  | 336 | 0.169 | 0.267 | 0.167 | 0.265 | 0.176 | 0.277 | 0.170 | 0.271 |
|  | 720 | 0.233 | 0.344 | 0.221 | 0.313 | 0.212 | 0.308 | 0.203 | 0.303 |
| Imp. |  | - | - | 1.41\% | 1.73\% | - | - | 3.18\% | 1.65\% |

## 6 Conclusion

In this paper, we propose the Adaptive Multi-Scale Decomposition (AMD) framework for time series forecasting to address the inherent complexity of time series data by decomposing it into multiple temporal patterns at various scales and adaptively aggregating these patterns. Comprehensive experiments demonstrate that AMD consistently achieves state-of-the-art performance in both longterm and short-term forecasting tasks across various datasets, showcasing superior efficiency and effectiveness. Looking ahead, we plan to further explore the integration of multi-scale information and expand the application of AMD as a backbone or plugin in other mainstream time series analysis tasks, such as imputation, classification, and anomaly detection.

## References

[1] Hongjie Xia, Huijie Ao, Long Li, Yu Liu, Sen Liu, Guangnan Ye, and Hongfeng Chai. CI-STHPAN: Pre-trained attention network for stock selection with channel-independent spatio-temporal hypergraph. Proceedings of the AAAI Conference on Artificial Intelligence, 38(8):9187-9195, Mar. 2024. 1
[2] Dawei Cheng, Fangzhou Yang, Sheng Xiang, and Jin Liu. Financial time series forecasting with multimodality graph neural network. Pattern Recognition, 121:108218, 2022. 1
[3] Hao Xue and Flora D. Salim. Utilizing language models for energy load forecasting. In Proceedings of the ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation, BuildSys '23, page 224-227, New York, NY, USA, 2023. Association for Computing Machinery. 1
[4] Daejin Kim, Youngin Cho, Dongmin Kim, Cheonbok Park, and Jaegul Choo. Residual correction in realtime traffic forecasting. In Proceedings of the ACM International Conference on Information \& Knowledge Management, page 962-971, New York, NY, USA, 2022. Association for Computing Machinery. 1
[5] Kristofers Volkovs, Evalds Urtans, and Vairis Caune. Primed unet-lstm for weather forecasting. In Proceedings of the International Conference on Advances in Artificial Intelligence, ICAAI '23, page 13-17, New York, NY, USA, 2024. Association for Computing Machinery. 1
[6] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023. 1, 3, 7, 15, 17, 19, 20
[7] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 22419-22430. Curran Associates, Inc., 2021. 2, 3
[8] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International Conference on Machine Learning, pages 27268-27286. PMLR, 2022. 3, 7, 15, 17, 20
[9] Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In International Conference on Learning Representations, 2023. 3, 7, 15, 17
[10] Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. iTransformer: Inverted transformers are effective for time series forecasting. In International Conference on Learning Representations, 2024. 3
[11] Tao Dai, Beiliang Wu, Peiyuan Liu, Naiqi Li, Jigang Bao, Yong Jiang, and Shu-Tao Xia. Periodicity decoupling framework for long-term series forecasting. In International Conference on Learning Representations, 2024. 1
[12] Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan K Mathur, Rajat Sen, and Rose Yu. Long-term forecasting with TiDE: Time-series dense encoder. Transactions on Machine Learning Research, 2023. 1, 3, 7, 15
[13] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? Proceedings of the AAAI Conference on Artificial Intelligence, 37(9):11121-11128, Jun. 2023. 1, 2, 3, 7, 9, 15, 17
[14] Ilya O Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, and Zhai. MLP-Mixer: An all-MLP architecture for vision. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 24261-24272. Curran Associates, Inc., 2021.
[15] Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y. Zhang, and JUN ZHOU. TimeMixer: Decomposable multiscale mixing for time series forecasting. In International Conference on Learning Representations, 2024. 2, 3
[16] Zhe Li, Zhongwen Rao, Lujia Pan, and Zenglin Xu. MTS-Mixers: Multivariate time series forecasting via factorized temporal and channel mixing. arXiv preprint arXiv:2302.04501, 2023. 1, 7, 9, 15, 17
[17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017. 1
[18] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, and Max Tegmark. Kan: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756, 2024. 1
[19] Ronghao Ni, Zinan Lin, Shuaiqi Wang, and Giulia Fanti. Mixture-of-linear-experts for long-term time series forecasting. arXiv preprint arXiv:2312.06786, 2024. 1, 3
[20] Huiqiang Wang, Jian Peng, Feihu Huang, Jince Wang, Junhui Chen, and Yifei Xiao. MICN: Multi-scale local and global context modeling for long-term series forecasting. In International Conference on Learning Representations, 2023. 3, 7, 15, 17
[21] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. TimesNet: Temporal 2d-variation modeling for general time series analysis. In International Conference on Learning Representations, 2023. 3, 7, 15, 17
[22] Jiezhu Cheng, Kaizhu Huang, and Zibin Zheng. Towards better forecasting by fusing near and distant future visions. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04):3593-3600, Apr. 2020. 3
[23] Shengsheng Lin, Weiwei Lin, Wentai Wu, Feiyu Zhao, Ruichao Mo, and Haotong Zhang. SegRNN: Segment recurrent neural network for long-term time series forecasting. arXiv preprint arXiv:2308.11200, 2023. 3
[24] Yuxin Jia, Youfang Lin, Xinyan Hao, Yan Lin, S. Guo, and Huaiyu Wan. WITRAN: Water-wave information transmission and recurrent acceleration network for long-range time series forecasting. In Advances in Neural Information Processing Systems, 2023. 3
[25] Yijing Liu, Qinxian Liu, Jianwei Zhang, H. Feng, Zhongwei Wang, Zihan Zhou, and Wei Chen. Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks. In Advances in Neural Information Processing Systems, 2022. 3
[26] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, and Chengqi Zhang. Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, KDD '20, page 753-763, New York, NY, USA, 2020. Association for Computing Machinery.
[27] Kun Yi, Qi Zhang, Wei Fan, Hui He, Liang Hu, Pengyang Wang, Ning An, Longbing Cao, and Zhendong Niu. FourierGNN: Rethinking multivariate time series forecasting from a pure graph perspective. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 69638-69660. Curran Associates, Inc., 2023. 3
[28] Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary Transformers: Exploring the stationarity in time series forecasting. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 9881-9893. Curran Associates, Inc., 2022. 3
[29] Tianping Zhang, Yizhuo Zhang, Wei Cao, Jiang Bian, Xiaohan Yi, Shun Zheng, and Jian Li. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlp structures. arXiv preprint arXiv:2207.01186, 2022. 3

[30] Zhijian Xu, Ailing Zeng, and Qiang Xu. FITS: Modeling time series with $\$ 10 \mathrm{k} \$$ parameters. In International Conference on Learning Representations, 2024. 3, 20
[31] Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X. Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International Conference on Learning Representations, 2022. 3
[32] Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In International Conference on Learning Representations, 2020.
[33] Tian Zhou, Ziqing MA, Xue Wang, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, and Rong Jin. FiLM: Frequency improved legendre memory model for long-term time series forecasting. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 12677-12690. Curran Associates, Inc., 2022. 3
[34] G. E. P. Box, Gwilym M. Jenkins, and John F. Macgregor. Some recent advances in forecasting and control. Journal of The Royal Statistical Society Series C-applied Statistics, 17:158-179, 1968. 3
[35] Gabriel Dalforno Silvestre, Moisés Rocha dos Santos, and André C.P.L.F. de Carvalho. Seasonal-Trend decomposition based on Loess + Machine Learning: Hybrid Forecasting for Monthly Univariate Time Series. In 2021 International Joint Conference on Neural Networks (IJCNN), pages 1-7, 2021. 3
[36] Minhao LIU, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia LAI, Lingna Ma, and Qiang Xu. Scinet: Time series modeling and forecasting with sample convolution and interaction. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 5816-5828. Curran Associates, Inc., 2022. 3
[37] Si-An Chen, Chun-Liang Li, Sercan O Arik, Nathanael Christian Yoder, and Tomas Pfister. TSMixer: An all-MLP architecture for time series forecast-ing. Transactions on Machine Learning Research, 2023. 3
[38] Wanlin Cai, Yuxuan Liang, Xianggen Liu, Jianshuai Feng, and Yuankai Wu. MSGNet: Learning multiscale inter-series correlations for multivariate time series forecasting. Proceedings of the AAAI Conference on Artificial Intelligence, 38(10):11141-11149, Mar. 2024. 4
[39] Noam Shazeer, *Azalia Mirhoseini, *Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017. 6
[40] Bo Li, Yifei Shen, Jingkang Yang, Yezhen Wang, Jiawei Ren, Tong Che, Jun Zhang, and Ziwei Liu. Sparse mixture-of-experts are domain generalizable learners. In International Conference on Learning Representations, 2023. 6
[41] Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jangho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2022. 19
[42] Lu Han, Han-Jia Ye, and De-Chuan Zhan. The capacity and robustness trade-off: Revisiting the channel independent strategy for multivariate time series forecasting. arXiv preprint arXiv:2304.05206, 2023. 20
[43] Charles C. Holt. Forecasting seasonals and trends by exponentially weighted moving averages. International Journal of Forecasting, 20(1):5-10, 2004. 20

## A Proof of Theorem 1

Theorem 1. Let multi-scale mixing representation $g(x)$, where $g(x) \in \mathbb{R}^{1 \times L}$ (for simplicity, we consider univariate sequences) and the original sequence $f(x)$ is Lipschitz smooth with constant $\mathcal{K}$ (i.e. $\left|\frac{f(a)-f(b)}{a-b}\right| \leq \mathcal{K}$ ), then there exists a linear model such that $\left|y_{t}-\hat{y}_{t}\right|$ is bounded, $\forall t=1, \ldots, T$.

Proof. We first prove that the downsampled sequence $f_{i}(x), \forall i=1, \ldots, n$ possesses Lipschitz smooth. $\forall t \in \mathcal{D}\left(f_{i}(x)\right), t \in \mathcal{D}(f(x))$, where $\mathcal{D}$ means the domain of the sequence. Therefore, we can conclude that,

$$
\begin{align*}
\left|\frac{f_{1}(a)-f_{1}(b)}{a-b}\right| & =\frac{1}{d}\left|\frac{\sum_{j=a d-d+1}^{a d} f(j)-\sum_{j=b d-d+1}^{b d} f(j)}{a-b}\right|  \tag{12}\\
& =\frac{1}{d}\left|\frac{\sum_{j=0}^{d-1}[f(a d+j-d+1)-f(b d+j-d+1)]}{a d-b d}\right|  \tag{13}\\
& \leq \frac{1}{d} \cdot d \mathcal{K}  \tag{14}\\
& \leq \mathcal{K} \tag{15}
\end{align*}
$$

Similarly, by mathematical induction in a bottom-up manner, we can prove that $f_{i}(x), \forall i=1, \ldots, n$ possesses Lipschitz smooth.
Then, we prove multi-scale mixing representation $g_{i}(x), \forall i=0, \ldots, n$ possesses Lipschitz smooth. According to the property of linear combination, we have $g_{i}(t)=f_{i}(t)+\sum_{j=0}^{\left[\frac{n}{d^{i+1}}\right]} g_{i+1}(t) W_{i}(j, t)$, where $W_{i}(j, t)$ represents the $j$-th row and $t$-th column. So we have,

$$
\begin{align*}
& \left|\frac{g_{n-1}(a)-g_{n-1}(b)}{a-b}\right|  \tag{16}\\
& =\left|\frac{f_{n-1}(a)-f_{n-1}(b)+\sum_{j=0}^{\left[\frac{L}{d^{n}}\right]} g_{n}(a) W_{n-1}(j, a)-\sum_{j=0}^{\left[\frac{L}{\left.d^{i+1}\right]}\right]} g_{n}(b) W_{n-1}(j, b)}{a-b}\right|  \tag{17}\\
& \leq\left|\frac{f_{n-1}(a)-f_{n-1}(b)}{a-b}\right|+\left|\frac{\sum_{j=0}^{\left[\frac{L}{d^{n}}\right]} g_{n}(a) W_{n-1}(j, a)-\sum_{j=0}^{\left[\frac{L}{d^{n}}\right]} g_{n}(b) W_{n-1}(j, b)}{a-b}\right|  \tag{18}\\
& \leq \mathcal{K}+\left|\sum_{j=0}^{\left[\frac{L}{d^{n}}\right]} \frac{g_{n}(a)-g_{n}(b)}{a-b} \cdot \max \left\{W_{n-1}(j, a), W_{n-1}(j, b)\right\}\right|  \tag{19}\\
& =\mathcal{K}+\left|\sum_{j=0}^{\left[\frac{L}{d^{n}}\right]} \frac{f_{n}(a)-f_{n}(b)}{a-b} \cdot \max \left\{W_{n-1}(j, a), W_{n-1}(j, b)\right\}\right|  \tag{20}\\
& \leq \mathcal{K} \cdot\left(1+\left|\sum_{j=0}^{\left[\frac{L}{d^{n}}\right]} \max \left\{W_{n-1}(j, a), W_{n-1}(j, b)\right\}\right|\right) \tag{21}
\end{align*}
$$

Therefore, $g_{n-1}$ possesses smooth due to $W_{i}$ being a constant matrix. Similarly, by mathematical induction in a top-down manner, we can prove that $g_{i}(x), \forall i=0, \ldots, n$ possesses Lipschitz smooth.
Subsequently, we prove $\left|y_{t}^{m}-y_{t}^{\hat{m}}\right|$ is bounded, Where $y_{t}^{m}$ means the TP mixed observed data, and $y_{t}^{\hat{m}}$ represents the predicted TP mixed data. We set the period of the finest granularity time pattern as P. If there is no periodicity, then P tends to positive infinity ( $P \rightarrow+\infty$ ).

$$
\begin{align*}
& y_{t}^{m}=g(L+t)=g(P+1+t)  \tag{22}\\
& y_{t}^{\hat{m}}=g(L+t) A \oplus b \tag{23}
\end{align*}
$$

Let $A \in \mathbb{R}^{L \times T}$, and

$$
A_{t j}=\left\{\begin{array}{ll}
1, & \text { if } j=P+1 \text { or } j=(t \% P)+1  \tag{24}\\
-1, & \text { if } j=1 \\
0, & \text { otherwise }
\end{array} \quad, \quad b_{t}=0\right.
$$

Then,

$$
\begin{equation*}
y_{t}^{\hat{m}}=g(t \% P+1)-g(1)+g(P+1) \tag{25}
\end{equation*}
$$

So,

$$
\begin{align*}
& \left|y_{t}^{m}-y_{t}^{\hat{m}}\right|  \tag{26}\\
= & |[g(P+1+t)-g(P+1)]-[g(t \% P+1)-g(1)]|  \tag{27}\\
\leq & |g(P+1+t)-g(P+1)|+|g(t \% P+1)-g(1)|  \tag{28}\\
\leq & \mathcal{K}(t+t \% P) \tag{29}
\end{align*}
$$

Finally, we employ a weighted pattern predictor on the TP mixed data. Since we solely apply internal averaging operations during this process, the resulting remains bounded. This is due to the inherent property of internal averaging to smooth out fluctuations within the data without introducing significant variations beyond certain bounds. Therefore, $\left|y_{t}-\hat{y_{t}}\right|$ is bounded.

## B Detailed Algorithm Description

The pseudocode of the AMD algorithm is shown in Algorithm 1. The algorithm initializes input data and parameters and perform normalization. The data is then iterated through MDM block to extract multi-scale information. DDI blocks subsequently performs aggregation operations. Following this, for each feature, TPSelector determines predictor weights, with outputs concatenated and weighted for prediction. The algorithm transposes and de-normalizes predictions before returning them.

```
Algorithm 1 The Overall Architecture of AMD
    Input: look-back sequence $X \in \mathbb{R}^{L \times C}$.
    Parameter: DDI block number $n$.
    Output: Predictions $\hat{\mathrm{Y}}$.
    $\mathbf{X}=\operatorname{Rev} \operatorname{In}(\mathbf{X}, \quad$ 'norm')
    $\mathbf{X}=\mathbf{X}^{\mathbf{T}} \quad \triangleright \mathbf{X} \in \mathbb{R}^{C \times L}$
    $\mathbf{U}=\operatorname{MDM}(\mathbf{X}) \quad \quad \triangleright \mathbf{U} \in \mathbb{R}^{C \times L}$
    for $i$ in $\{1, \ldots, n\}$ do
        $\mathbf{U}=$ LayerNorm $(\mathbf{U})$
        $\mathbf{V}_{0}^{l}=\mathbf{U}_{0}^{P}$
        while $j \leq L$ do
            $\mathbf{Z}_{j}^{j+\bar{P}}=\mathbf{U}_{j}^{j+P}+$ FeedForward $\left(\mathbf{V}_{j-P}^{j}\right)$
            $\mathbf{V}_{j}^{j+P}=\mathbf{Z}_{j}^{j+P}+\beta \cdot$ FeedForward $\left(\left(\mathbf{Z}_{j}^{j+P}\right)^{T}\right)^{T}$
            $j=j+P$
        end while
    end for
    $\mathbf{v}=\operatorname{Split}(\mathbf{V})$
    for $i$ in $\{1, \ldots, C\}$ do
        $\mathbf{S}=$ TP-Selector $(\mathbf{u}) \quad \triangleright \mathbf{S} \in \mathbb{R}^{m \times T}$
        $\hat{\mathbf{y}}=\operatorname{SUM}(\mathbf{S} \odot \operatorname{Predictor}(\mathbf{v}), \operatorname{dim}=0) \quad \triangleright \hat{\mathbf{y}} \in \mathbb{R}^{1 \times T}$
    end for
    $\hat{\mathbf{Y}}=\hat{\mathbf{Y}}^{\mathbf{T}} \quad \triangleright \hat{\mathbf{Y}} \in \mathbb{R}^{L \times C}$
    $\hat{\mathbf{Y}}=\operatorname{Rev} \operatorname{In}(\hat{\mathbf{Y}}$, 'denorm')
    Return $\hat{\mathbf{Y}} \quad \triangleright$ Prediction Results.
```


## C Details of Experiments

## C. 1 Detailed Dataset Descriptions

Detailed dataset descriptions are shown in Tab. 5. Dim denotes the number of channels in each dataset. Dataset Size denotes the total number of time points in (Train, Validation, Test) split respectively. Prediction Length denotes the future time points to be predicted and four prediction settings are included in each dataset. Frequency denotes the sampling interval of time points. Information refers to the meaning of the data.

Table 5: Detailed dataset descriptions.

| Dataset | Dim | Prediction Length | Dataset Size | Frequency | Information |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ETTh1, ETTh2 | 7 | \{96,192,336,720\} | $(8545,2881,2881)$ | Hourly | Electricity |
| ETTm1, ETTm2 | 7 | \{96,192,336,720\} | (34465,11521,11521) | 15 min | Electricity |
| Exchange | 8 | \{96,192,336,720\} | $(5120,665,1422)$ | Daily | Economy |
| Weather | 21 | \{96,192,336,720\} | (36792,5271,10540) | 10 min | Weather |
| ECL | 321 | \{96,192,336,720\} | (18317,2633,5261) | Hourly | Electricity |
| Traffic | 862 | \{96,192,336,720\} | (12185,1757,3509) | Hourly | Transportation |
| Solar-Energy | 137 | \{96,192,336,720\} | $(36601,5161,10417)$ | 10 min | Energy |
| PEMS03 | 358 | 12 | (15617,5135,5135) | 5 min | Transportation |
| PEMS04 | 307 | 12 | (10172,3375,3375) | 5 min | Transportation |
| PEMS07 | 883 | 12 | (16911,5622,5622) | 5 min | Transportation |
| PEMS08 | 170 | 12 | (10690,3548,3548) | 5 min | Transportation |

## C. 2 Baseline Models

We briefly describe the selected baselines:
(1) MTS-Mixers [16] is an MLP-based model utilizing two factorized modules to model the mapping between the input and the prediction sequence. The source code is available at https://github. com/plumprc/MTS-Mixers.
(2) DLinear [13] is an MLP-based model with just one linear layer, which unexpectedly outperforms Transformer-based models in long-term TSF. The source code is available at https://github.com/ cure-lab/LTSF-Linear.
(3) TiDE [12] is a simple and effective MLP-based encoder-decoder model. The source code is available at https://github.com/thuml/Time-Series-Library.
(4) PatchTST [6] is a Transformer-based model utilizing patching and CI technique. It also enable effective pre-training and transfer learning across datasets. The source code is available at https: //github.com/yuqinie98/PatchTST.
(5) Crossformer [9] is a Transformer-based model introducing the Dimension-Segment-Wise (DSW) embedding and Two-Stage Attention (TSA) layer to effectively capture cross-time and crossdimension dependencies. The source code is available at https://github.com/Thinklab-SJTU/ Crossformer.
(6) FEDformer [8] is a Transformer-based model proposing seasonal-trend decomposition and exploiting the sparsity of time series in the frequency domain. The source code is available at https://github.com/DAMO-DI-ML/ICML2022-FEDformer.
(7) TimesNet [21] is a CNN-based model with TimesBlock as a task-general backbone. It transforms 1D time series into 2D tensors to capture intraperiod and interperiod variations. The source code is available at https://github.com/thuml/TimesNet.
(8) MICN [20] is a CNN-based model combining local features and global correlations to capture the overall view of time series. The source code is available at https://github.com/wanghq21/MICN.

## C. 3 Metric Details

Regarding metrics, we utilize the mean square error (MSE) and mean absolute error (MAE) for long-term forecasting. The calculations of these metrics are:

$$
\begin{aligned}
& M S E=\frac{1}{T} \sum_{0}^{T}\left(\hat{y}_{i}-y_{i}\right)^{2} \\
& M A E=\frac{1}{T} \sum_{0}^{T}\left|\hat{y}_{i}-y_{i}\right|
\end{aligned}
$$

## C. 4 Hyper-Parameter Selection and Implementation Details

For the main experiments, we have the following hyper-parameters. The patch length $P$, the number of DDI blocks $n$. The dimension of hidden state of DDI $d_{\text {model }}=\max \left\{32,2^{[\log (\text { feature_num })]}\right\}$. The number of predictor is set to 8 , while the topK is set to 2 . The dimension of hidden state in AMS is set to 2048. The weight decay is set to $1 e^{-7}$. Adam optimizer is used for training and the initial learning rate is shown in Tab. 6. For all datasets, to leverage more distinct temporal patterns, we set the number of MDM layers $h$ to 3 and the downsampling rate $c$ to 2 . We report the specific hyper-parameters chosen for each dataset in Tab. 6.
For all of the baseline models, we replicated the implementation using configurations outlined in the original paper or official code.

Table 6: The hyper-parameters for different experimental settings.

| Dataset | $P$ | $\alpha$ | Batch Size | Epochs | $n$ | Learning Rate | Layer Norm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ETTh1 | 16 | 0.0 | 128 | 10 | 1 | $5 \times e^{-5}$ | True |
| ETTh2 | 4 | 1.0 | 128 | 10 | 1 | $5 \times e^{-5}$ | False |
| ETTm1 | 16 | 0.0 | 128 | 10 | 1 | $3 \times e^{-5}$ | True |
| ETTm2 | 8 | 0.0 | 128 | 10 | 1 | $1 \times e^{-5}$ | True |
| Exchange | 4 | 0.0 | 512 | 10 | 1 | $3 \times e^{-4}$ | True |
| Weather | 16 | 0.0 | 128 | 10 | 1 | $5 \times e^{-5}$ | True |
| ECL | 16 | 0.0 | 128 | 20 | 1 | $3 \times e^{-4}$ | False |
| Traffic | 16 | 0.0 | 32 | 20 | 1 | $8 \times e^{-5}$ | False |
| Solar-Energy | 8 | 1.0 | 128 | 10 | 1 | $2 \times e^{-5}$ | True |
| PEMS03 | 4 | 1.0 | 32 | 10 | 1 | $5 \times e^{-5}$ | False |
| PEMS04 | 4 | 1.0 | 32 | 5 | 1 | $5 \times e^{-5}$ | False |
| PEMS07 | 16 | 1.0 | 32 | 10 | 1 | $5 \times e^{-5}$ | False |
| PEMS08 | 16 | 1.0 | 32 | 10 | 1 | $5 \times e^{-5}$ | False |

## D Extra Experimental Results

## D. 1 Full Results

Due to the space limitation of the main text, we place the full results of long-term forecasting in Tab. 7. To evaluate the generality of our proposed AMS, we conduct long-term forecasting on existing realworld multivariate benchmarks under different prediction lengths $S \in\{96,192,336,720\}$. The input sequence length is searched among $\{96,192,336,512,672,720\}$ to the best for a fairer comparison for all baselines.

## D. 2 Robustness Evaluation

The results showed in Tab. 8 are obtained from five random seeds on the ETTh1, ETTm1, Weather, Solar-Energy, ECL and Traffic datasets, exhibiting that the performance of AMD is stable.

Table 7: Full results of the long-term forecasting task. All the results are averaged from 4 different prediction lengths $T \in\{96,192,336,720\}$. To the best for a fairer comparison for all baselines, the input sequence length $L$ is searched among $\{96,192,336,512,672,720\}$.

| Models Metric |  | AMD (Ours) |  | PatchTST [6] |  | Crossformer [9] |  | FEDformer [8] |  | TimesNet [21] |  | MICN [20] |  | DLinear [13] |  | MTS-Mixers [16] |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE |
| Weather | 96 | 0.145 | 0.198 | $\underline{0.149}$ | 0.198 | 0.153 | 0.217 | 0.238 | 0.314 | 0.172 | 0.220 | 0.161 | 0.226 | 0.152 | 0.237 | 0.156 | $\underline{0.206}$ |
|  | 192 | 0.190 | 0.240 | $\underline{0.194}$ | $\underline{0.241}$ | 0.197 | 0.269 | 0.275 | 0.329 | 0.219 | 0.261 | 0.220 | 0.283 | 0.220 | 0.282 | 0.199 | 0.248 |
|  | 336 | 0.242 | 0.280 | $\underline{0.245}$ | $\underline{0.282}$ | 0.252 | 0.311 | 0.339 | 0.377 | 0.280 | 0.306 | 0.275 | 0.328 | 0.265 | 0.319 | 0.249 | 0.291 |
|  | 720 | 0.315 | 0.332 | 0.314 | $\underline{0.334}$ | 0.318 | 0.363 | 0.389 | 0.409 | 0.365 | 0.359 | 0.311 | 0.356 | 0.323 | 0.362 | 0.336 | 0.343 |
|  | avg | 0.223 | 0.262 | $\underline{0.226}$ | $\underline{0.264}$ | 0.230 | 0.290 | 0.310 | 0.357 | 0.259 | 0.287 | 0.242 | 0.298 | 0.240 | 0.300 | 0.235 | 0.272 |
| ETTh1 | 96 | 0.369 | 0.397 | $\underline{0.370}$ | $\underline{0.399}$ | 0.386 | 0.429 | 0.376 | 0.415 | 0.384 | 0.402 | 0.396 | 0.427 | 0.375 | 0.399 | 0.372 | 0.395 |
|  | 192 | 0.401 | $\underline{0.417}$ | 0.413 | 0.421 | 0.416 | 0.444 | 0.423 | 0.446 | 0.457 | 0.436 | 0.430 | 0.453 | 0.405 | 0.416 | 0.416 | 0.426 |
|  | 336 | 0.418 | 0.427 | $\underline{0.422}$ | $\underline{0.436}$ | 0.440 | 0.461 | 0.444 | 0.462 | 0.491 | 0.469 | 0.433 | 0.458 | 0.439 | 0.443 | 0.455 | 0.449 |
|  | 720 | 0.439 | 0.454 | $\underline{0.447}$ | $\underline{0.466}$ | 0.519 | 0.524 | 0.469 | 0.492 | 0.521 | 0.500 | 0.474 | 0.508 | 0.472 | 0.490 | 0.475 | 0.472 |
|  | avg | 0.407 | 0.424 | $\underline{0.413}$ | $\underline{0.431}$ | 0.441 | 0.465 | 0.428 | 0.454 | 0.458 | 0.450 | 0.433 | 0.462 | 0.423 | 0.437 | 0.430 | 0.436 |
| ETTh2 | 96 | 0.274 | 0.337 | 0.274 | 0.337 | 0.628 | 0.563 | 0.332 | 0.374 | 0.340 | 0.374 | 0.289 | 0.357 | $\underline{0.289}$ | $\underline{0.353}$ | 0.307 | 0.354 |
|  | 192 | $\underline{0.351}$ | $\underline{0.383}$ | 0.339 | 0.379 | 0.703 | 0.624 | 0.407 | 0.446 | 0.402 | 0.414 | 0.409 | 0.438 | 0.383 | 0.418 | 0.374 | 0.399 |
|  | 336 | $\underline{0.375}$ | $\underline{0.411}$ | 0.329 | 0.380 | 0.827 | 0.675 | 0.400 | 0.447 | 0.452 | 0.452 | 0.417 | 0.452 | 0.448 | 0.465 | 0.398 | 0.432 |
|  | 720 | $\underline{0.402}$ | $\underline{0.438}$ | 0.379 | 0.422 | 1.181 | 0.840 | 0.412 | 0.469 | 0.462 | 0.468 | 0.426 | 0.473 | 0.605 | 0.551 | 0.463 | 0.465 |
|  | avg | $\underline{0.350}$ | $\underline{0.392}$ | 0.330 | 0.379 | 0.835 | 0.676 | 0.388 | 0.434 | 0.414 | 0.427 | 0.385 | 0.430 | 0.431 | 0.447 | 0.386 | 0.413 |
| ETTm1 | 96 | 0.284 | 0.339 | 0.290 | 0.342 | 0.316 | 0.373 | 0.326 | 0.390 | 0.338 | 0.375 | 0.314 | 0.360 | 0.299 | 0.343 | 0.314 | 0.358 |
|  | 192 | 0.322 | 0.362 | $\underline{0.332}$ | 0.369 | 0.377 | 0.411 | 0.365 | 0.415 | 0.371 | 0.387 | 0.359 | 0.387 | 0.335 | $\underline{0.365}$ | 0.354 | 0.386 |
|  | 336 | 0.361 | 0.383 | 0.366 | 0.392 | 0.431 | 0.442 | 0.391 | 0.425 | 0.410 | 0.411 | 0.398 | 0.413 | 0.369 | $\underline{0.386}$ | 0.384 | 0.405 |
|  | 720 | $\underline{0.421}$ | 0.418 | 0.416 | $\underline{0.420}$ | 0.600 | 0.547 | 0.446 | 0.458 | 0.478 | 0.450 | 0.459 | 0.464 | 0.425 | 0.421 | 0.427 | 0.432 |
|  | avg | 0.347 | 0.375 | 0.351 | 0.381 | 0.431 | 0.443 | 0.382 | 0.422 | 0.400 | 0.406 | 0.383 | 0.406 | 0.357 | $\underline{0.379}$ | 0.370 | 0.395 |
| ETTm2 | 96 | $\underline{0.167}$ | 0.258 | 0.166 | 0.256 | 0.421 | 0.461 | 0.180 | 0.271 | 0.187 | 0.267 | 0.178 | 0.273 | $\underline{0.167}$ | 0.260 | 0.177 | 0.259 |
|  | 192 | 0.221 | 0.295 | $\underline{0.223}$ | $\underline{0.296}$ | 0.503 | 0.519 | 0.252 | 0.318 | 0.249 | 0.309 | 0.245 | 0.316 | 0.224 | 0.303 | 0.241 | 0.303 |
|  | 336 | 0.270 | 0.327 | $\underline{0.274}$ | 0.329 | 0.611 | 0.580 | 0.324 | 0.364 | 0.321 | 0.351 | 0.295 | 0.350 | 0.281 | 0.342 | 0.297 | 0.338 |
|  | 720 | 0.356 | 0.382 | 0.362 | $\underline{0.385}$ | 0.996 | 0.750 | 0.410 | 0.420 | 0.497 | 0.403 | 0.389 | 0.409 | 0.397 | 0.421 | 0.396 | 0.398 |
|  | avg | 0.253 | 0.315 | 0.255 | 0.315 | 0.632 | 0.578 | 0.292 | 0.343 | 0.291 | 0.333 | 0.277 | 0.336 | 0.267 | 0.332 | 0.277 | $\underline{0.325}$ |
| ECL | 96 | 0.129 | 0.225 | 0.129 | 0.222 | 0.187 | 0.283 | 0.186 | 0.302 | 0.168 | 0.272 | 0.159 | 0.267 | 0.153 | 0.237 | 0.141 | 0.243 |
|  | 192 | $\underline{0.148}$ | 0.242 | 0.147 | 0.240 | 0.258 | 0.330 | 0.197 | 0.311 | 0.184 | 0.289 | 0.168 | 0.279 | 0.152 | 0.249 | 0.163 | 0.261 |
|  | 336 | $\underline{0.164}$ | 0.258 | 0.163 | 0.259 | 0.323 | 0.369 | 0.213 | 0.328 | 0.198 | 0.300 | 0.196 | 0.308 | 0.169 | 0.267 | 0.176 | 0.277 |
|  | 720 | 0.195 | 0.289 | $\underline{0.197}$ | $\underline{0.290}$ | 0.404 | 0.423 | 0.233 | 0.344 | 0.220 | 0.320 | 0.203 | 0.312 | 0.233 | 0.344 | 0.212 | 0.308 |
|  | avg | 0.159 | 0.254 | 0.159 | 0.253 | 0.293 | 0.351 | 0.207 | 0.321 | 0.192 | 0.295 | 0.182 | 0.297 | 0.177 | 0.274 | 0.173 | 0.272 |
| 윯 <br> Exchange | 96 | $\underline{0.083}$ | 0.201 | 0.093 | 0.214 | 0.186 | 0.346 | 0.136 | 0.276 | 0.107 | 0.234 | 0.102 | 0.235 | 0.081 | 0.203 | 0.083 | 0.201 |
|  | 192 | $\underline{0.171}$ | 0.293 | 0.192 | 0.312 | 0.467 | 0.522 | 0.256 | 0.369 | 0.226 | 0.344 | 0.172 | 0.316 | 0.157 | 0.293 | 0.174 | $\underline{0.296}$ |
|  | 336 | 0.309 | 0.402 | 0.350 | 0.432 | 0.783 | 0.721 | 0.426 | 0.464 | 0.367 | 0.448 | 0.272 | $\underline{0.407}$ | $\underline{0.305}$ | 0.414 | 0.336 | 0.417 |
|  | 720 | 0.750 | $\underline{0.652}$ | 0.911 | 0.716 | 1.367 | 0.943 | 1.090 | 0.800 | 0.964 | 0.746 | $\underline{0.714}$ | 0.658 | 0.643 | 0.601 | 0.900 | 0.715 |
|  | avg | 0.328 | $\underline{0.387}$ | 0.387 | 0.419 | 0.701 | 0.633 | 0.478 | 0.478 | 0.416 | 0.443 | $\underline{0.315}$ | 0.404 | 0.297 | 0.378 | 0.373 | 0.407 |
| Traffic | 96 | $\underline{0.366}$ | $\underline{0.259}$ | 0.360 | 0.249 | 0.512 | 0.290 | 0.576 | 0.359 | 0.593 | 0.321 | 0.508 | 0.301 | 0.410 | 0.282 | 0.462 | 0.332 |
|  | 192 | $\underline{0.381}$ | $\underline{0.265}$ | 0.379 | 0.256 | 0.523 | 0.297 | 0.610 | 0.380 | 0.617 | 0.336 | 0.536 | 0.315 | 0.423 | 0.287 | 0.488 | 0.354 |
|  | 336 | 0.397 | 0.273 | 0.392 | 0.264 | 0.530 | 0.300 | 0.608 | 0.375 | 0.629 | 0.336 | 0.525 | 0.310 | 0.436 | 0.296 | 0.498 | 0.360 |
|  | 720 | 0.430 | $\underline{0.292}$ | $\underline{0.432}$ | 0.268 | 0.573 | 0.313 | 0.621 | 0.375 | 0.640 | 0.350 | 0.571 | 0.323 | 0.466 | 0.315 | 0.529 | 0.370 |
|  | avg | 0.393 | 0.272 | 0.391 | 0.264 | 0.535 | 0.300 | 0.604 | 0.372 | 0.620 | 0.336 | 0.535 | 0.312 | 0.434 | 0.295 | 0.494 | 0.354 |
| Solar | 96 | 0.175 | 0.228 | 0.224 | 0.278 | 0.181 | 0.240 | 0.201 | 0.304 | 0.219 | 0.314 | 0.188 | 0.252 | 0.289 | 0.337 | 0.284 | 0.325 |
|  | 192 | 0.188 | 0.235 | 0.253 | 0.298 | $\underline{0.196}$ | $\underline{0.252}$ | 0.237 | 0.337 | 0.231 | 0.322 | 0.215 | 0.280 | 0.319 | 0.397 | 0.307 | 0.362 |
|  | 336 | 0.201 | 0.246 | 0.273 | 0.306 | 0.216 | 0.243 | 0.254 | 0.362 | 0.246 | 0.337 | 0.222 | 0.267 | 0.352 | 0.415 | 0.333 | 0.384 |
|  | 720 | 0.203 | 0.249 | 0.272 | 0.308 | $\underline{0.220}$ | $\underline{0.256}$ | 0.280 | 0.397 | 0.280 | 0.363 | 0.226 | 0.264 | 0.356 | 0.412 | 0.335 | 0.383 |
|  | avg | 0.192 | 0.240 | 0.256 | 0.298 | 0.204 | $\underline{0.248}$ | 0.243 | 0.350 | 0.244 | 0.334 | 0.213 | 0.266 | 0.329 | 0.400 | 0.315 | 0.363 |
| $1{ }^{\text {st }}$ count |  | 44 |  | $\underline{24}$ |  |  | 1 |  | 0 | 0 |  |  | 2 |  | 5 |  | 0 |

## D. 3 Hyper-Parameter Sensitivity

Varying Input Length and Downsampling Parameters. Time patterns are obtained from the input sequence through downsampling. Therefore, the size of the input length $L$, the number of downsampling operations $h$, and downsampling rate $d$ all significantly affect the accuracy of prediction. To investigate the impact, we conduct the following experiments. On the ETTm1 dataset, we first choose $L$ among $\{96,192,336,512,672,720\}$. As shown in Fig. 5a, the forecasting performance benefits from the increase of input length. For the best input length, we choose $h$ among $\{1,2,3,4,5\}$ and $d$ among $\{1,2,3\}$. From the results shown in Tab. 9, it can be found that as the number of downsampling operations $h$ increases, we observe improvements in different prediction lengths. Therefore, we choose a setting of 3 layers to find a balance between efficiency and performance.

Table 8: Robustness of AMD performance. The results are obtained from five random seeds.

| Datasets Metric | ETTh1 |  | ETTm1 |  | Weather |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | MSE | MAE | MSE | MAE | MSE | MAE |
| 96 | $0.3691 \pm 0.0008$ | $0.3969 \pm 0.0001$ | $0.2838 \pm 0.0004$ | $0.3387 \pm 0.0003$ | $0.1451 \pm 0.0003$ | $0.1982 \pm 0.0002$ |
| 192 | $0.4008 \pm 0.0007$ | $0.4170 \pm 0.0002$ | $0.3218 \pm 0.0004$ | $0.3618 \pm 0.0002$ | $0.1898 \pm 0.0003$ | $0.2401 \pm 0.0003$ |
| 336 | $0.4177 \pm 0.0005$ | $0.4272 \pm 0.0002$ | $0.3607 \pm 0.0003$ | $0.3828 \pm 0.0002$ | $0.2419 \pm 0.0004$ | $0.2801 \pm 0.0003$ |
| 720 | $0.4389 \pm 0.0009$ | $0.4541 \pm 0.0002$ | $0.4209 \pm 0.0004$ | $0.4182 \pm 0.0003$ | $0.3151 \pm 0.0005$ | $0.3316 \pm 0.0002$ |
| Datasets | Solar-Energy |  | ECL |  | Traffic |  |
| Metric | MSE | MAE | MSE | MAE | MSE | MAE |
| 96 | $0.1751 \pm 0.0003$ | $0.2277 \pm 0.0003$ | $0.1293 \pm 0.0002$ | $0.2552 \pm 0.0002$ | $0.3659 \pm 0.0004$ | $0.2591 \pm 0.0003$ |
| 192 | $0.1875 \pm 0.0004$ | $0.2350 \pm 0.0003$ | $0.1481 \pm 0.0003$ | $0.2419 \pm 0.0001$ | $0.3806 \pm 0.0003$ | $0.2647 \pm 0.0003$ |
| 336 | $0.2009 \pm 0.0003$ | $0.2456 \pm 0.0002$ | $0.1642 \pm 0.0003$ | $0.2581 \pm 0.0002$ | $0.3965 \pm 0.0004$ | $0.2725 \pm 0.0003$ |
| 720 | $0.2032 \pm 0.0003$ | $0.2490 \pm 0.0003$ | $0.1948 \pm 0.0002$ | $0.2890 \pm 0.0002$ | $0.4302 \pm 0.0004$ | $0.2918 \pm 0.0002$ |

![](https://cdn.mathpix.com/cropped/2025_05_28_9ee9a9cc6faa08a54792g-18.jpg?height=524&width=1397&top_left_y=768&top_left_x=369)

Figure 5: Hyper-Parameter Sensitivity with respect to the input sequence length, the number of predictiors and the $\lambda_{1}$. The results are recorded with the all four predict length.

Varying Number of Predictors and TopK. The number of decomposed temporal patterns and the dominant temporal patterns are determined by the number of predictors and the topK value, respectively. We conduct hyperparameter sensitivity experiments about these two parameters on the Weather dataset and the results are shown in Fig. 5b. We observe an improvement in prediction results as the number of predictors increases. However, this also leads to an increase in the memory usage of selector weights. To strike a balance between memory and performance, we finally opt for 8 predictors.

Varying Scaling of $\mathcal{L}_{\text {selector }}$. We conduct an investigation into the scaling factor for the load balancing loss $\mathcal{L}_{\text {selector }}$, denoted by $\lambda_{1}$, on the Weather dataset. As shown in observation 4, we found that when $\lambda_{1}$ is set to 0 , the loss function does not guide the reasonable assignment of different temporal patterns to different predictors, resulting in poor prediction performance. However, when $\lambda_{1}$ is not 0 , as shown in the Fig. 5c, the model demonstrates strong robustness to $\lambda_{1}$, with prediction results independent of scaling. Upon observing the loss function, we noticed that $\mathcal{L}_{\text {selector }}$ has already decreased significantly in the early epochs. Therefore, we finally choose $\lambda_{1}$ to be 1.0 .

Table 9: The MSE resluts of different downsampling operations $h$ and downsampling rate $d$ on the ETTm1 dataset. '-' indicates no downsampling when $h=0$. In this case, $d$ has no effect on the result.

| Predict Length $\mathrm{d}=1$ h | 96 | 192 | 336 | 720 | Predict Length $\mathrm{d}=2$ h | 96 | 192 | 336 | 720 | Predict Length $\mathrm{d}=3$ h | 96 | 192 | 336 | 720 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 0.292 | 0.333 | 0.371 | 0.431 | 0 | - | - | - | - | 0 | - | - | - | - |
| 1 | 0.288 | 0.326 | 0.367 | 0.428 | 1 | 0.286 | 0.326 | 0.365 | 0.427 | 1 | 0.286 | 0.327 | 0.366 | 0.428 |
| 2 | 0.285 | 0.323 | 0.364 | 0.424 | 2 | 0.284 | 0.322 | 0.362 | 0.423 | 2 | 0.284 | 0.324 | 0.366 | 0.426 |
| 3 | 0.285 | 0.322 | 0.362 | 0.423 | 3 | 0.284 | 0.322 | 0.361 | 0.421 | 3 | 0.283 | 0.322 | 0.363 | 0.424 |
| 4 | 0.283 | 0.322 | 0.360 | 0.421 | 4 | 0.284 | 0.323 | 0.360 | 0.423 | 4 | 0.284 | 0.322 | 0.364 | 0.423 |
| 5 | 0.284 | 0.323 | 0.360 | 0.422 | 5 | 0.285 | 0.321 | 0.359 | 0.423 | 5 | 0.284 | 0.322 | 0.363 | 0.422 |

## D. 4 Extra Ablation Study

Ablation on MDM. MDM is used to extract different temporal patterns. We conduct an ablation experiment by removing MDM. The results are as shown in the first row of Tab. 9 when the number of downsampling operations $n$ equals 0 . It can be seen that removing the MDM module does not yield satisfactory prediction results.

Ablation on AMS. AMS is based on a MoE structure to exploit information from every temporal pattern. We conducted an ablation experiment by replacing the TPSelector and predictors in AMS with a single predictor. The results, as shown in Tab. 10, demonstrate a significant decrease in prediction accuracy.

Table 10: Ablation on AMS results of only a single predictor.

| Datasets | ETTh1 |  | ETTm1 |  | Weather |  | Solar-Energy |  | ECL |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Metric | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE |
| 96 | 0.379 | 0.406 | 0.291 | 0.349 | 0.150 | 0.206 | 0.179 | 0.235 | 0.137 | 0.234 |
| 192 | 0.413 | 0.428 | 0.330 | 0.373 | 0.199 | 0.248 | 0.194 | 0.246 | 0.157 | 0.251 |
| 336 | 0.434 | 0.440 | 0.373 | 0.398 | 0.248 | 0.287 | 0.209 | 0.257 | 0.171 | 0.267 |
| 720 | 0.458 | 0.466 | 0.443 | 0.436 | 0.320 | 0.339 | 0.213 | 0.263 | 0.203 | 0.298 |

Dense MoE Strategy and Sparse MoE Strategy for AMD. In theory, the information contained in each temporal pattern is useful and discarding any one of them would result in loss of information. Therefore, we adopt the dense MoE strategy, where dominant temporal patterns are given larger weights, while others are given smaller weights instead of being set to 0 . Here, we conduct ablation experiments that compare the dense MoE with the sparse MoE to demonstrate this assertion. As shown in Tab. 11, compared to Dense MoE, Sparse MoE shows increased prediction errors. This observation highlights the consequence of omitting non-dominant temporal pattern information, which invariably leads to a degradation in performance.

Table 11: Ablation on Dense MoE Strategy and Sparse MoE Strategy, where (d) represents Dense MoE, and (s) represents Sparse MoE.

| Datasets | ETTh1(d) |  | ETTh1(s) |  | Weather(d) |  | Weather(s) |  | ECL(d) |  | ECL(s) |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Metric | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE |
| 96 | 0.369 | 0.397 | 0.375 | 0.403 | 0.145 | 0.198 | 0.150 | 0.205 | 0.129 | 0.225 | 0.132 | 0.228 |
| 192 | 0.401 | 0.417 | 0.410 | 0.426 | 0.190 | 0.240 | 0.193 | 0.245 | 0.148 | 0.242 | 0.151 | 0.244 |
| 336 | 0.418 | 0.427 | 0.428 | 0.436 | 0.242 | 0.280 | 0.244 | 0.284 | 0.164 | 0.258 | 0.165 | 0.259 |
| 720 | 0.439 | 0.454 | 0.444 | 0.459 | 0.315 | 0.332 | 0.315 | 0.334 | 0.195 | 0.289 | 0.199 | 0.293 |

## E Discussions on Limitations and Future Improvement

Recently, several specific designs have been utilized to better capture complex sequential dynamics, such as normalization, patching, frequency domain representation, channel independence, sequence decomposition, and others, as shown in Fig. 6.
(1) Normalization : Real-world time series always exhibit non-stationary behavior, where the data distribution changes over time. RevIn [41] is a normalization-and-denormalization method for TSF to effectively constrain non-stationary information (mean and variance) in the input layer and restore it in the output layer. The work has managed to improve the delineation of temporal dependencies while minimizing the influence of noise. In AMD, we also adopt the method of Revin. However, it struggles to adequately resolve the intricate distributional variations among the layers within deep networks, so we need to make further improvements to address this distributional shift.
(2) Patching : Inspired by the utilization of local semantic context in computer vision (CV) and natural language processing (NLP), the technique of patching is introduced [6]. Since TS data exhibit locality, individual time steps lack the semantic meaning found in words within sentences. Therefore, extracting the local semantic context is crucial for understanding their connections. Additionally, this approach has the advantage of reducing the number of parameters. In AMD, we also develop a
![](https://cdn.mathpix.com/cropped/2025_05_28_9ee9a9cc6faa08a54792g-20.jpg?height=739&width=1385&top_left_y=243&top_left_x=370)

Figure 6: Five Designs for Sequential Modelling with respect to the RevIN, Patching, Frequency Domain Representation, Channel Independence and Sequence Decomposition.
patching mechanism to extract recent history information. However, how to better exploit locality remains an issue that requires further research.
(3) Frequency Domain Representation : TS data, characterized by their inherent complexity and dynamic nature, often contain information that is sparse and dispersed across the time domain. The frequency domain representation is proposed to promise a more compact and efficient representation of the inherent patterns. Related methods [8] still rely on feature engineering to detect the dominant period set. However, some overlooked periods or trend changes may represent significant events, resulting in information loss. In the future, we can explore adaptive temporal patterns mining in the frequency domain, thereby utilizing the Complex Frequency Linear module proposed by FITS [30] to mitigate the problem of large parameter sizes in MLP-based models when the look-back length is long.
(4) Channel Independence (CI) and Channel Dependence (CD) : CI and CD represent a trade-off between capacity and robustness [42], with the CD method offering greater capacity but often lacking robustness when it comes to accurately predicting distributionally drifted TS. In contrast, the CI method sacrifices capacity in favor of more reliable predictions. PatchTST [6] achieves SOTA results using the CI approach. However, neglecting correlations between channels may lead to incomplete modelling. In AMD, we leverage the CI approach while integrating dependencies across different variables over time, thus exploiting cross-channel relationships and enhancing robustness. However, the trade-off between capacity and robustness is also a balance between generalization and specificity, which still requires further research.
(5) Sequence Decomposition : The classical TS decomposition method [43] divides the complex temporal patterns into seasonal and trend components, thereby benefiting the forecasting process. In AMD, we go beyond the constraints of seasonal and trend-based time series decomposition. Instead, we develop an adaptive decomposition, mixing and forecasting module that fully exploits the information from different temporal patterns. However, the effectiveness of the adaptive decomposition module relies heavily on the availability and quality of historical data, which poses challenges in scenarios with limited or noisy data.

We believe that more effective sequence modelling designs will be proposed to adequately address issues such as distribution shift, multivariate sequence modelling, and so on. As a result, MLP-based models will also perform better in more areas of time series.


[^0]:    *Equal Contribution

