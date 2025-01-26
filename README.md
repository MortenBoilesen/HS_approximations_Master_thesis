# Github repository containing relevant code and data for master thesis: Hilbert space approximations of Gaussian processes in Bayesian modelling 

Master thesis project by:
Johanne Hvidberg Conradsen and Morten Rahbæk Boilesen

The repository is divided into folders corresponding to the chapters in the thesis. 


**Abstract**

In many fields such as medicine and econometrics, we encounter data where domain knowledge specifies a particular shape in the data, such as monotonicity or concavity. Incorporating these constraints into regression models can potentially improve prediction accuracy, especially in the presence of limited data. This thesis explores the development of shape-constrained models, enforcing monotonic and u-shaped behaviour by leveraging a reduced rank approximation of Gaussian processes, known as the Hilbert space approximation. In the first part of the thesis, we derive this approximation and investigate how it affects both predictive precision and computational complexity compared to a full Gaussian process. Through theoretical analysis, we illuminate how the trade-off between accuracy and complexity depends on the number of basis functions and the domain size of the approximation. In the second part of the thesis, we examine the advantages and limitations of using the Hilbert space approximation for constructing shape-constrained functions. We find that while it offers flexible and analytically tractable models, challenges remain in selecting appropriate priors and hyperparameters when data is scarce. In terms of prediction accuracy, the incorporation of shape constraints generally improves performance in interpolation tasks. We find that extrapolation tasks are more difficult, particularly for u-shaped models, due to an inherent exponentially growing prior variance. Finally, we explore whether shape-constrained models enhance data efficiency, observing some improvements on small datasets, although the results remain inconclusive and further investigation is warranted. Overall, this work provides useful insights into the potential and challenges of using the Hilbert space approximation for shape-constrained modelling.
