Welcome to this one-day workshop: **A crash course on using machine
learning methods effectively in practice**

-   Some materials and illustration are based on **Chapter 2** and 3 of
    [The Mathematical Engineering of Deep
    Learning](%7Bhttps://deeplearningmath.org/)

-   Data are available at
    <a href="https://github.com/benoit-liquet/SSA-MAPS-ML" class="uri">https://github.com/benoit-liquet/SSA-MAPS-ML</a>

COURSE OUTLINE
==============

-   9:00 to 10:45: Supervised Learning (part 1)
-   10:45 to 11:15: Break (might need a coffee)
-   11:15 to 12:45: Supervised Learning (part 2)
-   12:45 to 13:45: Lunch Break
-   13h45 to 15:15: Unsupervised Learning (part 1)
-   15:15 to 15:45: Break (might need a coffee)
-   15:45 to 17:00: Unsupervised Learning (part 2)

Supervised learning
===================

![](figure2_1_a_training_prediction.png)

Unsupervised learning
=====================

![](figure_2_1_b_clustering_reduction.png) - data is unlabelled and is
denoted via 𝒟 = {*x*<sup>(1)</sup>, …, *x*<sup>(*n*)</sup>}

Clustering using Kmeans
-----------------------

-   **Clustering** allows to identify meaningful groups, or clusters,
    among the data points and find representative centers of these
    clusters.

-   Samples within each cluster are more closely related to one another
    than samples from different clusters.

-   Clustering is the act of associating a cluster ℓ with each
    observation, where ℓ comes from a small finite set, {1, …, *K*}.

-   Clustering algorithm works on the data 𝒟 and outputs a function
    *c*( ⋅ ) which maps individual data points to the label values
    {1, …, *K*}.

-   **K-means** algorithm is one very basic, yet powerful heuristic
    algorithm.

-   With K-means, we pre-specify a number *K*, determining the number of
    clusters.

-   *K* may be treated as a hyper-parameter.

-   The algorithm seeks the function *c*( ⋅ ), or alternatively the
    partition *C*<sub>1</sub>, …, *C*<sub>*K*</sub>, it also seeks
    representative **centers** (also known as **centroids**), of the
    clusters, denoted by *J*<sub>1</sub>, …, *J*<sub>*K*</sub>, each an
    element of ℝ<sup>*p*</sup>.

-   Ideal aim of K-means is to minimization,

-   Generally computationally intractable since it requires considering
    all possible partitions of 𝒟 into clusters.

-   Can be approximately minimized via the K-means algorithm using a
    classic iterative approach.

-   The K-means algorithm: two sub-tasks called **mean computation**,
    and **labelling**.

-   **Mean computation:** Given *c*( ⋅ ), or a clustering
    *C*<sub>1</sub>, …, *C*<sub>*K*</sub>, find
    *J*<sub>1</sub>, …, *J*<sub>*K*</sub> that minimizes

-   **Labelling:** Given, *J*<sub>1</sub>, …, *J*<sub>*K*</sub>,
    *c*( ⋅ ) is defined as

-   The label of each element is determined by the closest center in
    Euclidean space.

Kmeans in action
----------------

![](workflowkmeans.png)

``` r
library(animation)
set.seed(101)
library(mvtnorm)
x = rbind(rmvnorm(40, mean=c(0,1),sigma = 0.05*diag(2)),rmvnorm(40, mean=c(0.5,0),sigma = 0.05*diag(2)),rmvnorm(40, mean=c(1,1),sigma = 0.05*diag(2)))
par(mfrow=c(3,2))
colnames(x) = c("x1", "x2")
kmeans.ani(x, centers = matrix(c(0.5,1,0.5,0,1,1),byrow=T,ncol=2))
```

Image Segmentation with K-means
-------------------------------

-   The goal of image segmentation is to label each pixel of an image
    with a unique class from a finite number of classes.

-   unsupervised image segmentation via K-means clustering

-   Each pixel of the image is considered a point in 𝒟 and the dimension
    of each point is typically *p* = 3 (red, green, and blue) for color
    images.

-   Can produce *impressive image segmentation* without any other
    information except for the image.

-   Example: color image is a *n* = 640 × 640 = 409, 600 pixel color
    image (*p* = 3).

-   Run K-means algorithm which groups similar pixels based on their
    *attributes* and *assigns* the attributes of the corresponding
    *cluster center* to the pixel in the image.

![](Segmentation_kmeans.png)

Your TURN
---------

``` r
library(ggplot2)
library(jpeg)
img <- readJPEG("Yoni-ben-pool-seg.jpg")

# Obtain the dimension
imgDm <- dim(img)

# Assign RGB channels to data frame
imgRGB <- data.frame(
  x = rep(1:imgDm[2], each = imgDm[1]),
  y = rep(imgDm[1]:1, imgDm[2]),
  R = as.vector(img[,,1]),
  G = as.vector(img[,,2]),
  B = as.vector(img[,,3])
)

par(mfrow=c(3,1))

# Plot the original image
p1 <- ggplot(data = imgRGB, aes(x = x, y = y)) + 
  geom_point(colour = rgb(imgRGB[c("R", "G", "B")])) +
  labs(title = "Original Image") +
  xlab("x") +
  ylab("y") 
p1
```

![](Machine_learning_Part2_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r
kClusters <- 2
kMeans <- kmeans(imgRGB[, c("R", "G", "B")], centers = kClusters)
kColours <- rgb(kMeans$centers[kMeans$cluster,])


p2 <- ggplot(data = imgRGB, aes(x = x, y = y)) + 
  geom_point(colour = kColours) +
  labs(title = paste("k-Means Clustering of", kClusters, "Colours")) +
  xlab("x") +
  ylab("y") 
p2
```

![](Machine_learning_Part2_files/figure-markdown_github/unnamed-chunk-2-2.png)

``` r
kClusters <- 6
kMeans <- kmeans(imgRGB[, c("R", "G", "B")], centers = kClusters)
kColours <- rgb(kMeans$centers[kMeans$cluster,])


p6 <- ggplot(data = imgRGB, aes(x = x, y = y)) + 
  geom_point(colour = kColours) +
  labs(title = paste("k-Means Clustering of", kClusters, "Colours")) +
  xlab("x") +
  ylab("y") 
p6
```

![](Machine_learning_Part2_files/figure-markdown_github/unnamed-chunk-2-3.png)

Principal Component Analysis
----------------------------

-   Not all *p* dimensions of the data are equally useful.

-   Especially the case in the presence of high dimensional data (large
    *p*).

-   Many features may be either completely redundant or uninformative.

-   These cases are referred to as **correlated features** or **noise
    features**

-   PCA is a well-known and widely used **dimensionality reduction
    technique** for a wide variety of applications such as *data
    compression*, *feature extraction*, and *visualization*.

-   Basic idea of PCA is to project each point of 𝒟 which has many
    correlated coordinates onto fewer coordinates called **principal
    components** which are uncorrelated.

![](PCA.jpg)

-   This is done while still retaining most of the variability present
    in the data.

-   PCA offers a low-dimensional representation of the features that
    attempts to capture the most important information from the data.

-   As input, PCA uses the de-meaned data from the centered data matrix
    *X*

-   PCA uses a linear combination of these columns to arrive at the
    vectors of the **new features**
    *x̃*<sub>(1)</sub>, …, *x̃*<sub>(*m*)</sub>.

$$
\\tilde{x}\_{(i)} = 
v\_{i,1}
\\begin{bmatrix}
\\vert  \\\\ 
 x\_{(1)}  \\\\ 
 \\vert 
\\end{bmatrix}
+
v\_{i,2}
\\begin{bmatrix}
\\vert  \\\\ 
 x\_{(2)}  \\\\ 
 \\vert 
\\end{bmatrix}
+ 
\~
\\ldots
\~
+
v\_{i,p}
\\begin{bmatrix}
\\vert  \\\\ 
 x\_{(p)}  \\\\ 
 \\vert 
\\end{bmatrix}
\\quad
\\text{for}
\\quad
i=1,\\ldots,m,
$$

-   each new *n* dimensional vector, *x̃*<sub>(*i*)</sub>, is a linear
    combination of the original features.

-   *x̃*<sub>(*i*)</sub> = *X**v*<sub>*i*</sub> where
    *v*<sub>*i*</sub> = (*v*<sub>*i*, 1</sub>, …, *v*<sub>*i*, *p*</sub>)
    is called the **loading vector** for *i*

PCA on Wisconsin breast cancer data
-----------------------------------

-   Wisconsin breast cancer data: *p* = 30 and *n* = 569.

-   Aim to visualize this data using PCA we set *m* = 2

``` r
load("Breast_cancer.RData")
head(Breast_cancer_data[,c(1:6)])
```

    ##         id diagnosis radius_mean texture_mean perimeter_mean area_mean
    ## 1   842302         M       17.99        10.38         122.80    1001.0
    ## 2   842517         M       20.57        17.77         132.90    1326.0
    ## 3 84300903         M       19.69        21.25         130.00    1203.0
    ## 4 84348301         M       11.42        20.38          77.58     386.1
    ## 5 84358402         M       20.29        14.34         135.10    1297.0
    ## 6   843786         M       12.45        15.70          82.57     477.1

``` r
data <- Breast_cancer_data
dim(data)
```

    ## [1] 569  32

``` r
library(FactoMineR)
library(ggplot2)
library(dplyr)

pca <- PCA(data[,-c(1,2)],ncp=2,graph=FALSE)
dat <- data.frame(data,pc1=pca$ind$coord[,1],pc2=pca$ind$coord[,2],diagnosis=as.factor(data[,2]))

#dat <- dat %>% filter(pc1<7 & pc2<10) 

p1 <- ggplot(data = dat, aes(x = pc1, y = pc2))+
  geom_hline(yintercept = 0, lty = 2) +
  geom_vline(xintercept = 0, lty = 2) +
  geom_point(alpha = 0.8,size=2.5) + theme_bw()
 
p1 + theme(axis.text = element_text(size = 20))+ theme(axis.title = element_text(size = 20))   
```

![](Machine_learning_Part2_files/figure-markdown_github/unnamed-chunk-4-1.png)

-   Variance explained by the first two components

``` r
pca$eig[1:2,]
```

    ##        eigenvalue percentage of variance cumulative percentage of variance
    ## comp 1  13.281608               44.27203                          44.27203
    ## comp 2   5.691355               18.97118                          63.24321

-   Color the points based on the labels benign vs. malignant, a useful
    pattern emerges ?

``` r
p2 <- ggplot(data = dat, aes(x = pc1, y = pc2, color = diagnosis))+
  geom_hline(yintercept = 0, lty = 2) +
  geom_vline(xintercept = 0, lty = 2) +
  geom_point(alpha = 0.8,size=2.5) + theme_bw()+
  theme(legend.position=c(0.15,0.85),legend.title=element_blank())
p3 <- p2 +  scale_color_discrete( labels = c("benign", "malignant"))

p3 + theme(legend.text=element_text(size=20),axis.text = element_text(size = 20))+ theme(axis.title = element_text(size = 20))
```

![](Machine_learning_Part2_files/figure-markdown_github/unnamed-chunk-6-1.png)

Derivation of PCA
-----------------

-   The PCA framework tries to project the data in the directions with
    maximum variance.

-   Since *x̃*<sub>(*i*)</sub> = *X**v*<sub>*i*</sub> we can formulate
    this by maximizing the sample variance of the components of
    *x̃*<sub>(*i*)</sub>.

-   *x̃*<sub>(*i*)</sub> is a 0 mean vector, its sample variance is
    *x̃*<sub>(*i*)</sub><sup>⊤</sup>*x̃*<sub>(*i*)</sub>/*n*.

-   We have,
    $$
    \\text{Sample variance of component}\~i = \\frac{1}{n} v\_i^\\top X^\\top X v\_i  = v\_i^\\top S v\_i,
    $$
    where *S* is the sample covariance of the data.

-   It turns out the a very useful way to represent the loading vectors
    *v*<sub>1</sub>, …, *v*<sub>*m*</sub> is by normed eigenvectors
    associated with eigenvalues of the sample covariance matrix *S*

-   *S* is symmetric and positive semi-definite, the eigenvalues of *S*
    are real and non-negative, a fact which allows us to order them via
    *λ*<sub>1</sub> ≥ *λ*<sub>2</sub> ≥ … ≥ *λ*<sub>*p*</sub> ≥ 0.

-   We then pick the loading vector *v*<sub>*i*</sub> to be a normed
    eigenvector associated with *λ*<sub>*i*</sub>, namely,

-   The first loading vector is associated with the highest eigenvalue;
    the second is associated with the second highest eigenvalue; and so
    fourth. The symmetry of *S* also means that its eigenvectors are
    orthogonal and hence *Ṽ* is a matrix with orthonormal columns.

PCA Through SVD
---------------

-   Any *n* × *p* dimensional matrix *X* of rank *r* can be represented
    as

-   *n* × *r* matrix *U* and the *p* × *r* matrix *V* are both with
    orthonormal columns denoted *u*<sub>*i*</sub> and *v*<sub>*i*</sub>
    respectively for *i* = 1, …, *r*.

-   Columns are called the left and right **singular vectors**
    respectively.

-   *δ*<sub>*i*</sub> in the *r* × *r* diagonal matrix *Δ* are called
    **singular values** and are ordered as
    *δ*<sub>1</sub> ≥ *δ*<sub>2</sub> ≥ ⋯ ≥ *δ*<sub>*r*</sub> \> 0.

-   SVD representation of the sample covariance:
    $$
    S = \\frac{1}{n} \\underbrace{V\\Delta^\\top U^\\top}\_{X^{\\top}}\\underbrace{U\\Delta V^\\top}\_{X} = \\frac{1}{n} V \\Delta^2 V^\\top,
    \\quad
    \\text{with}
    \\quad
    \\Delta^2=\\textrm{diag}(\\delta\_1^2,\\ldots,\\delta\_r^2).
    $$

Here the fact that *U* has orthonormal columns implies
*U*<sup>⊤</sup>*U* is the *r* × *r* identity matrix and hence it cancels
out:

-   Compare to the eigenvector based representation of PCA:

-   Using the *Spectral decomposition of *S**

-   Thus, *λ*<sub>*i*</sub> = *δ*<sub>*i*</sub><sup>2</sup>/*n* and the
    loading vectors in spectral decomposition are the right singular
    vectors in SVD: *Ṽ* = *V*.

-   Further, to obtain the data matrix of principal components, *X̃* we
    set *X̃* = *X**V*. Using the SVD, PCA can be represented:

-   Each column of the reduced data matrix *X̃* is a left singular vector
    *u*<sub>*i*</sub> stretched by the singular value *δ*<sub>*i*</sub>.

SVD for Compression
-------------------

-   The singular value decomposition can also be viewed as a means for
    compressing any matrix *X*.

-   A rank *m* \< *r* approximation of *X* is,

-   The rank of *X̂* is *m* and since one often uses *m* significantly
    smaller than *r*, this is called a **low rank approximation**.

-   For small enough *δ*<sub>*m* + 1</sub> the approximation error is
    negligible since the summation of rank one matrices
    *δ*<sub>*i*</sub> *u*<sub>*i*</sub> *v*<sub>*i*</sub><sup>⊤</sup>
    for *i* = *m* + 1, …, *r* is small.

-   The number of values used in this representation of *X̂* is
    *m* × (1 + *n* + *p*) and for small *m* this number is generally
    much smaller than *n* × *p* which is the number of values in *X*.

-   Hence this may viewed as a compression method.

SVD in action for compression
-----------------------------

-   We seek to have the best rank *m* approximation in terms of
    minimization of ∥*X* − *X̂*∥<sub>*F*</sub>.

-   Frobenious norm noted ∥*A*∥<sub>*F*</sub>: square root of the sum of
    the squared elements of the matrix *A*

-   Low rank approximations established by **Eckart–Young-Mirsky
    theorem**.

-   consider a simple visual example with a 353 × 469 monochrome
    (grayscale) image appearing this is *X*.

![](SVD_compress.png)

-   The original image uses 353 × 469 = 165, 557 values while the
    *m* = 50 approximation only uses 50 × (1 + 353 + 469) = 41, 150
    values. That is the approximation yields *X̂* which is compressed to
    about 25% of the size of *X* and looks very similar.

``` r
if (!"jpeg" %in% installed.packages()) install.packages("jpeg")
# Read image file into an array with three channels (Red-Green-Blue, RGB)
myImage <- jpeg::readJPEG("CODE_WORKSHOP/pool_graysacle.jpg")

r <- myImage[, , 1] 
# Performs full SVD 
myImage.r.svd <- svd(r)# ; lmyImage.g.svd <- svd(g) ; myImage.b.svd <- svd(b)
rgb.svds <- list(myImage.r.svd)#



plot.image <- function(pic, main = "") {
  h <- dim(pic)[1] ; w <- dim(pic)[2]
  plot(x = c(0, h), y = c(0, w), type = "n", xlab = "", ylab = "", main = main)
  rasterImage(pic, 0, 0, h, w)
}


compress.image <- function(rgb.svds, nb.comp) {
  # nb.comp (number of components) should be less than min(dim(img[,,1])), 
  # i.e., 170 here
  svd.lower.dim <- lapply(rgb.svds, function(i) list(d = i$d[1:nb.comp], 
                                                     u = i$u[, 1:nb.comp], 
                                                     v = i$v[, 1:nb.comp]))
  img <- sapply(svd.lower.dim, function(i) {
    img.compressed <- i$u %*% diag(i$d) %*% t(i$v)
  }, simplify = 'array')
  img[img < 0] <- 0
  img[img > 1] <- 1
  return(list(img = img, svd.reduced = svd.lower.dim))
}



par(mfrow = c(2, 2))
plot.image(r, "Original image")

p <- 10 ; plot.image(compress.image(rgb.svds, p)$img[,,1], 
                     paste("SVD with", p, "components"))

p <- 30 ; plot.image(compress.image(rgb.svds, p)$img[,,1], 
                     paste("SVD with", p, "components"))


p <- 50 ; plot.image(compress.image(rgb.svds, p)$img[,,1], 
                     paste("SVD with", p, "components"))
```

A taste of Shallow Autoencoders
-------------------------------

-   A schematic of an autoencoder with a single **hidden layer**.

![](autoencoder.png)

-   The input *x* ∈ ℜ<sup>*p*</sup> is transformed into a
    **bottleneck**, also called the **code** which is some
    *x̃* ∈ ℜ<sup>*m*</sup> and is the hidden layer of the model.

-   Then the bottleneck is further transformed into the output
    *x̂* ∈ ℜ<sup>*p*</sup>.

-   The part of the autoencoder that transforms the input into the
    bottleneck is called the **encoder** and the part of the autoencoder
    that transforms the bottleneck to the output is called the
    **decoder**. Both the encoder and the decoder have parameters that
    are to be learned.

-   Interestingly for input *x*, once parameters are trained, we
    generally expect the autoencoder to generate output *x̂* that is as
    similar to the input *x* as possible.

-   Consider the activity of data reduction where the dimension of the
    bottleneck *m* is significantly smaller than the input and output
    dimension *p*

-   For example, return to the case of MNIST digits where *p* = 784. For
    our example here, assume we have an auto encoder with *m* = 30.

![](recontrsuction_AE.png)

-   A trained autoencoder yields *x* ≈ *x̂* then it means that we have an
    immediate data reduction method.

-   With the trained encoder we are able to convert digit images, each
    of size 28 × 28 = 784, into much smaller vectors, each of size 30.

-   With the trained decoder we are able to convert back and get an
    approximation of the original image. This choice of *m* implies a
    rather remarkable compression factor of about 26.

Autoencoder loss
----------------

-   The most straightforward choice for the distance penalty in
    *C*<sub>*i*</sub>(*θ*) is the square of the Euclidean distance:

With this cost structure, learning the parameters, *θ*, of an
autoencoder based on data 𝒟 is the process of minimizing *C*(*θ* ; 𝒟.

-   Decompose *f*<sub>*θ*</sub>( ⋅ ) to be a composition of the encoder
    function denoted via
    *f*<sub>*θ*<sup>\[1\]</sup></sub><sup>\[1\]</sup>( ⋅ ) and the
    decoder function denoted via
    *f*<sub>*θ*<sup>\[2\]</sup></sub><sup>\[2\]</sup>( ⋅ ):

*x̂* = *f*<sub>*θ*</sub>(*x*) = (*f*<sub>*θ*<sup>\[2\]</sup></sub><sup>\[2\]</sup> ∘ *f*<sub>*θ*<sup>\[1\]</sup></sub><sup>\[1\]</sup>)(*x*) = *f*<sub>*θ*<sup>\[2\]</sup></sub><sup>\[2\]</sup>(*f*<sub>*θ*<sup>\[1\]</sup></sub><sup>\[1\]</sup>(*x*)),

-   We define,

-   The encoder parameters *θ*<sup>\[1\]</sup> are composed of the bias
    *b*<sup>\[1\]</sup> ∈ ℜ<sup>*m*</sup> and weight matrix
    *W*<sup>\[1\]</sup> ∈ ℜ<sup>*m* × *p*</sup>

-   The decoder parameters *θ*<sup>\[2\]</sup> are composed of the bias
    *b*<sup>\[2\]</sup> ∈ ℜ<sup>*p*</sup> and weight matrix
    *W*<sup>\[2\]</sup> ∈ ℜ<sup>*p* × *m*</sup>.

-   Vector activation functions *S*<sup>\[1\]</sup>( ⋅ ) and
    *S*<sup>\[2\]</sup>( ⋅ ): we construct these based on scalar
    activation functions *σ*<sup>\[ℓ\]</sup> : ℜ → ℜ for ℓ = 1, 2.

-   Specifically, we set *S*<sup>\[ℓ\]</sup>(*z*) to be the element wise
    application of *σ*<sup>\[ℓ\]</sup>( ⋅ ) on each of the coordinates
    of *z*

-   The loss function representation as,

-   With this loss function, for given data 𝒟, the learned autoencoder
    parameters *θ̂* are given by a solution to the optimization problem
    min<sub>*θ*</sub>*C*(*θ* ; 𝒟).

PCA is an Autoencoder
---------------------

-   Autoencoders generalize principal component analysis (PCA)

-   PCA is essentially a shallow auto-encoder with identity activation
    functions *σ*<sup>\[ℓ\]</sup>(*u*) = *u* for ℓ = 1, 2, also known as
    a **linear autoencoder**.

-   PCA yields one possible solution to the learning optimization
    problem for linear autoencoders.

![](PCA_autoencoder.png)

Autoencoder on MNIST
--------------------

-   consider using an autoencoder on MNIST where *p* = 28 × 28 = 784 and
    we use *m* = 2.

-   We encode this via PCA, a shallow non-linear autoencoder, and a deep
    autoendoer that has hidden layers.

-   The autoencoders are trained on the training set andcthe codes
    presented are both for the training set, and for the testing set
    data.

-   We color the code points based on the labels. This allows us to see
    how different labels are generally encoded onto different regions of
    the code space.

![](MNIST_autoencoder.png)

-   One application of such data reduction is to help separate the data

-   It is evident that as model complexity increases better separation
    occurs in the data.

-   In terms of reconstruction, it is also evident in this case that
    more complex models exhibit better reconstruction ability.

A denoising autoencoder
-----------------------

-   This model learns to remove noise during the reconstruction step for
    noisy input data.

-   It takes in partially corrupted input and learns to recover the
    original denoised input.

-   It relies on the hypothesis that high-level representations are
    relatively stable and robust to entry corruption and that the model
    is able to extract characteristics that are useful for the
    representation of the input distribution.

![](reconstructing_input.png)

Interpolation on the latent space
---------------------------------

-   Take *x*<sup>(*i*)</sup> and *x*<sup>(*j*)</sup> from
    𝒟 = {*x*<sup>(1)</sup>, …, *x*<sup>(*n*)</sup>} and consider the
    convex combination
    *x*<sub>*λ*</sub><sup>naive</sup> = *λ**x*<sup>(*i*)</sup> + (1 − *λ*)*x*<sup>(*j*)</sup>,
    for some *λ* ∈ \[0, 1\].

-   *x*<sub>*λ*</sub><sup>naive</sup> is a weighted average between the
    two observations.

-   *λ* captures which of the observations has more weight.

-   Such arithmetic on the associated feature vectors is too naive and
    often meaningless

-   When considering the latent space representation of the images it is
    often possible to create a much more meaningful interpolation
    between the images.

![](interpol.png)

-   Train an autoencoder and then encode *x*<sup>(*i*)</sup> and
    *x*<sup>(*j*)</sup> to obtain *x̃*<sup>(*i*)</sup> and
    *x̃*<sup>(*j*)</sup>.

-   Then interpolate on the codes, and finally decode *x̃*<sub>*λ*</sub>
    to obtain an interpolated image.

*x*<sub>*λ*</sub><sup>encoder</sup> = *f*<sup>\[2\]</sup>(*λ**f*<sup>\[1\]</sup>(*x*<sup>(*i*)</sup>) + (1 − *λ*)*f*<sup>\[1\]</sup>(*x*<sup>(*j*)</sup>)).

-   one potential application of such interpolation is for design
    purposes, say in art or architecture, where one chooses two samples
    as a starting point and then uses interpolation to see other samples
    lying \`\`in between’’.

``` r
library(ruta)
library(rARPACK)
library(ggplot2)
###############
### Function plot 
###############
plot_digit <- function(digit, ...) {
  image(keras::array_reshape(digit, c(28, 28), "F")[, 28:1], xaxt = "n", yaxt = "n", col=gray(1:256 / 256), ...)
}

plot_sample <- function(digits_test, model1,model2,model3, sample) {
  sample_size <- length(sample)
  layout(
    matrix(c(1:sample_size, (sample_size + 1):(4 * sample_size)), byrow = F, nrow = 4)
  )
  
  
  for (i in sample) {
    par(mar = c(0,0,0,0) + 1)
    plot_digit(digits_test[i, ])
    plot_digit(model1[i, ])
    plot_digit(model2[i, ])
    plot_digit(model3[i, ])
  }
}


#######################
#### Load MNIST DATA
#######################

mnist = keras::dataset_mnist()
```

    ## Loaded Tensorflow version 2.3.1

``` r
# Normalization to the [0, 1] interval
x_train <- keras::array_reshape(
  mnist$train$x, c(dim(mnist$train$x)[1], 784)
)
x_train <- x_train / 255.0
x_test <- keras::array_reshape(
  mnist$test$x, c(dim(mnist$test$x)[1], 784)
)
x_test <- x_test / 255.0

if(T){
network <- input() + dense(30, "tanh") + output("sigmoid")
network1 <- input() + dense(50, "tanh") +dense(10, "linear")+dense(50, "tanh") +output("sigmoid")
}

### model simple
network.simple <- autoencoder(network)#, loss = "binary_crossentropy")
model = train(network.simple, x_train, epochs = 10)
decoded.simple <- reconstruct(model, x_test)


### model deep
my_ae2 <- autoencoder(network1)#, loss = "binary_crossentropy")
model2 = train(my_ae2, x_train, epochs = 10)
decoded2 <- reconstruct(model2, x_test)

#### Linear interpolation between two digits
digit_A = x_train[which(mnist$train$y==3)[1],]#MNIST digit with 3 (This is the first digit in the train set that has 3)
digit_B = x_train[which(mnist$train$y==3)[10],]#another MNIST digit with 3 (This is the 10[th] digit in the train set that has 3)
latent_A = encode(model2,matrix(digit_A,nrow=1))
latent_B = encode(model2,matrix(digit_B,nrow=1))
lambda = 0.5
latent_interpolation = lambda*latent_A + (1-lambda)*latent_B
rought_interpolation = lambda*digit_A + (1-lambda)*digit_B

output_interpolation = decode(model2,latent_interpolation)


par(mar = c(0,0,0,0) + 1,mfrow=c(1,3))
plot_digit(digit_A)
plot_digit(as.vector(output_interpolation))
plot_digit(digit_B)
```

![](Machine_learning_Part2_files/figure-markdown_github/unnamed-chunk-8-1.png)

``` r
par(mar = c(0,0,0,0) + 1,mfrow=c(1,3))
plot_digit(digit_A)
plot_digit(as.vector(rought_interpolation))
plot_digit(digit_B)
```

![](Machine_learning_Part2_files/figure-markdown_github/unnamed-chunk-8-2.png)
