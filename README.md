![alt text](https://miro.medium.com/max/3840/1*GkWTK4BydcVe3d334SKsjg.jpeg)

# Support Vector Machine Hot Dog Classifier
A python implementation of *Support Vector Machine(SVM)* from scratch using the *Stochastic Gradient Descent(SGD)*.
A Hot Dog classifiier made using SVM.


# SUPPORT VECTOR MACHINE (SVM)
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:
1. Effective in high dimensional spaces.
2. Still effective in cases where number of dimensions is greater than the number of samples.
3. Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
4. Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to    specify custom kernels.

The disadvantages of support vector machines include:
1. If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term    is   crucial.
2. SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation .

The SVM (Support Vector Machine) is a supervised machine learning algorithm typically used for binary classification problems. It’s trained by feeding a dataset with labeled examples (xᵢ, yᵢ). For instance, if your examples are email messages and your problem is spam detection, then:
An example email message xᵢ is defined as an n-dimensional feature vector that can be plotted on n-dimensional space.
The feature vector, as the name explains, contains features (eg. word count, link count, etc.) of your email message in numerical form
Each feature vector is labeled with a class yᵢ
The class yᵢ can either be a +ve or -ve (eg. spam=1, not-spam=-1)

![alt text](https://miro.medium.com/max/625/1*ala8WX2z47WYpn932hUkhA.jpeg)


# Feature Engineering
Machine learning algorithms operate on a dataset that is a collection of labeled examples which consist of features and a label i.e. in our case diagnosis is a label, [radius_mean, structure_mean, texture_mean…] features, and each row is an example.
In most of the cases, the data you collect at first might be raw; its either incompatible with your model or hinders its performance. That’s when feature engineering comes to rescue. It encompasses preprocessing techniques to compile a dataset by extracting features from raw data. These techniques have two characteristics in common:
1. Preparing the data which is compatible with the model
2. Improving the performance of the machine learning algorithm

## Normalization
It is one of the many feature engineering techniques that we are going to use. Normalization is the process of converting a range of values, into a standard range of values, typically in the interval [−1, 1] or [0, 1]. It’s not a strict requirement but it improves the speed of learning (e.g. faster convergence in gradient descent) and prevents numerical overflow.


# Cost Function
![Cost Function](https://miro.medium.com/max/500/1*vn2HDrdqBsKN5rYw7rjO5w.png)
Also known as the Objective Function. One of the building blocks of every machine learning algorithm, it’s the function we try to minimize or maximize to achieve our objective.
*What’s our objective in SVM?* Our objective is to find a hyperplane that separates +ve and -ve examples with the largest margin while keeping the misclassification as low as possible (see Figure 3).
*How do we achieve this objective?* We will minimize the cost/objective function shown below:
![Cost Function Equation](https://miro.medium.com/max/688/1*JAS6rUTO7TDlrv4XZSMbsA.png)
In the training phase, Larger C results in the narrow margin (for infinitely large C the SVM becomes hard margin) and smaller C results in the wider margin.
You might have seen another version of a cost function that looks like this:
![Cost Function Equation](https://miro.medium.com/max/678/1*6w_B_DjhGvaqCnvhzhhkDg.png)
Larger λ gives a wider margin and smaller λ results in the narrow margin (for infinitely small λ the SVM becomes hard margin).
In this cost function, λ is essentially equal to 1/C and has the opposite effect i.e larger λ gives a wider margin and vice versa. We can use any of the above cost functions keeping in mind what each regularization parameter (C and λ) does and then tuning them accordingly. Let’s see how can we calculate the total cost as given in (1) and then we will move on to its gradient which will be used in the training phase to minimize it.


### Stopping Critieria for SGD
We will stop the training when the current cost hasn’t decreased much as compared to the previous cost.


# RESULTS WITH SCRATCH IMPLEMENTATIONS

## Without using Feature Selection
Accuracy on test dataset: 95.36%
Recall on test dataset: 92.34%
Precision on test dataset: 92.34%

## With using Feature Selection
Accuracy on test dataset: 96.74%
Recall on test dataset: 95.34%
Precision on test dataset: 95.34%


# RESULTS using Scki