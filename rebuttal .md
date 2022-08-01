


## Reviewer tLbS 
We sincerely appreciate your detailed comments and positive ranking. We provide point-wise responses to your concerns below.

**Comment1:  Concerns about the overemphasis on robust generalization. Abstract Line 2, Line 40, Line 52 This work mainly targets the robust overfitting, not a robust generalization problem which does not show any results on robust generalization on the unseen dataset. The test set is not an unseen dataset. Moreover, the validation set which is sampled from the training set is not an unseen dataset. It is seen dataset but unseen samples or validation samples. To claim the robust generalization on the unseen dataset, the author should test on a different dataset that has different data distribution to the training set. For example, use CIFAR-10 for training and test on CIFAR-100. However, since the author didn't target this work, I hope the author could clarify the term robust generalization on the unseen dataset.**
- Thank you very much for sharing your concerns. Following previous work [1,2,3], we use robust generalization to denote the generalization gap between the adversarial accuracy of training set and test set. Indeed, we agree that the "robust generalization" used in our paper can be misleading. We have clarified it as "the generalization gap between the adversarial accuracy of training set and test set" in the revised version.

[1] [Robust Overfitting may be mitigated by properly learned smoothening. ICLR 2021.](https://openreview.net/pdf?id=qZzy5urZw9)
[2] [Adversarial weight perturbation helps robust generalization. NeurIPS 2020.](https://arxiv.org/pdf/2004.05884.pdf)
[3] [Relating Adversarially Robust Generalization to Flat Minima. ICCV 2021.](https://arxiv.org/abs/2104.04448)


**Comment2:  The reported performances in [1] and the performances of current paper are different. [1] shows 49.44, and 25.35 against AutoAttack in CIFAR-10, and CIFAR-100, respectively. However, in Table 1, the author claim SWA shows 48.61 against AutoAttack in CIFAR-10 which is worse than reported performance. Moreover, WOT-B which is the best method of authors' shows 48.96, and 25.26 against AutoAttack in CIFAR-10, and CIFAR-100, respectively, which are worse than the performance of [1]'s paper.**
- Thank you for your concerns. The result difference between the original SWA and our paper is because the results of SWA in [1] were achieved by combining SWA with knowledge distillation (KD). Although WOT-W/B is orthogonal to KD and we can also combine KD with our method, the authors of [1] did not provide the pretrained teacher model. Therefore, we feel our way to compare will clearly breakdown the effect of various components (SWA and WOT) on robust performance.  
- Besides, **we highlight that** even without the combination of KD technique, WOT-B can already achieve a matching performance with the combination of SWA and KD as shown below in Table S1 . We hope our clarification will address your concern.

**Table S1** Robust Accuacy (\%) Under AA$_{\infty}$ attack
|Methods | CIFAR-10 | CIFAR-100 |
| -------------  | :-----------: |:-----------:|
|  WOT-B(Gaps:m=400, Number of Gaps:K=4)  |   48.96   |  25.26  |
|  WOT-B(Gpas:m=600, Number of Gaps:K=4)  | **49.51** |  25.33   |
|  KD\&SWA[1]                             |   49.44   |  **25.35**   |

[1] [Robust Overfitting may be mitigated by properly learned smoothening. ICLR 2021.](https://openreview.net/pdf?id=qZzy5urZw9)

**Comment3: Moreover, I hope the author could compare the baselines in a different dataset in Table3.**
- Thank you for your suggestion. Following your suggestion, we added a set of experiments on a more complex dataset - Tiny-ImageNet. The results have been updated in the revised paper and also reported below in Table S2 which are fully consistent with our previous conclusion: our method consistently improves the robust accuracy over the previous methods. 

**Table S2** Robust Accuacy (\%) Under AA$_{\infty}$ attack 
|Methods | Clean Accuracy | Robust Accuracy (L$_2$) |Robust Accuracy (L$_\infty$) |
| -------------  | :-----------: |:-----------:|:-----------:|
|  AT+early stop                          |   42.76   |  36.61  |  14.39  |
|  AT+SWA                                 |   49.19   |  42.40  |  17.94  | 
|  WOT-W(Gpas:m=400, Number of Gaps:K=4)  | **49.31** |  42.43  |  17.10  |
|  WOT-B(Gpas:m=400, Number of Gaps:K=4)  |   48.83   | **42.54** |   **18.77** |

**Comment4:  As shown in figure 4, alpha becomes 0 after the best robust performances. From the results, WOT seems to work as a flag of early stopping. When authors could provide more explanation of how WOT can lead to a weighted sum that is not overfitted could make the paper more strong. It would be better if the authors also explain why WOT can find a better smooth weight than SWA.**
-  Thank you for your concerns. The fundamental difference between WOT and SWA is that, different from SWA which heuristically averages all the models during training, WOT ensembles different trajectories in a way that maximizes the robust performance on a small **hold-out** set. This process involves a learning step on the unseen data that can naturally mitigates the robust overfitting issue, improving the generalization performance of our model.
- Besides, also due to the learning step, our method automatically serves as a flag of early stopping (still, the alpha values gradually decrease to zero instead of suddenly becoming zero as early stopping does shown in Figure 4 and Figure 8 (in our submission)).
- On the other hand, while SWA demonstrates promising results  in mitigating robust overfitting, it can not fully eliminate robust overfitting [1] (also shown in the below Table S3). In sharp contrast, WOT can completely address the robust overfitting issue (shown in our submission: Table 4 and Figure 4), also indicating that WOT works better than SWA.

**Table S3** Robust Accuacy (\%) Under PGD-10 attack
|Methods | Best CheckPoint | Last Checkpoint | GAP |
| -------------  | :-----------: |:-----------:|:-----------:|
|   WOT-B(Gaps:m=400, Number of Gaps:K=4)  |   55.87   |  55.83  |  **0.04**  |
|    SWA                                   |   55.22   |  53.14  |  2.09  |

[1] [Fixing Data Augmentation to Improve Adversarial Robustness. NeurIPS 2021.](https://arxiv.org/abs/2103.01946)

**Comment5:  Is there any visualization or dynamics of the alpha of blocks? Is there a large difference in the dynamic of alpha for each block? I wonder what the characteristics of each block are which could explain why WOT-B is better than WOT-W.**
-  Thank you for your insightful suggestion, which helps to improve our paper. We report alpha values below for WOT-B (Number of Gaps:K=4,Gaps: m=400) on CIFAR-10 dataset with PreActResNet-18. Table S4 shows the k-averaged alpha values during training process.  Table S5 shows the mean alpha values averaging along the training process.
-  Both Table S4 and Table S5 show that WOT-B assigns large weights for middle blocks (Block-2,3,4,5) and small weights for bottom and top blocks (block-1,6). 
-  Moreover, Table S4 shows that the alpha gradually decreases during training rather than directly decreasing to zero, indicating that WOT-B does not simple work as early stop.
-  We have added plots of the dynamic of alpha for each block during training in Appendix H, where the trend is highly consistent with our findings.

**Table S4** The K-averaged alpha values during training process.
|  Epochs |  Block-1  |  Block-2  |  Block-3  |  Block-4  |  Block-5  |  Block-6  |
| -------------  | :-----------: |:-----------:|:-----------:| :-----------: |:-----------:|:-----------:|
|  Epoch:103     |   0.488   |  0.883  |   0.998  |  0.969  |  0.875  |  0.096  |
|  Epoch:124     |   0.194   |  0.526  |   0.852  |  0.716  |  0.544  |  0.000  |
|  Epoch:144     |   0.296   |  0.322  |   0.310  |  0.260  |  0.311  |  0.000  |
|  Epoch:161     |   0.005   |  0.0144 |   0.043  |  0.012  |  0.006  |  0.000  |
|  Epoch:178     |   0.003   |  0.004  |   0.034  |  0.000  |  0.000  |  0.000  |
|  Epoch:195     |   0.004   |  0.000  |   0.014  |  0.001  |  0.000  |  0.000  |

**Table S5** The mean alpha values averaging along the training process.
|  Epochs |  Block-1  |  Block-2  |  Block-3  |  Block-4  |  Block-5  |  Block-6  |
| -------------  | :-----------: |:-----------:|:-----------:| :-----------: |:-----------:|:-----------:|
|   K=1   |   0.488   |  0.883  |   0.998  |  0.969  |  0.875  |  0.096  |
|   K=2   |   0.194   |  0.526  |   0.852  |  0.716  |  0.544  |  0.000  |
|   K=3   |   0.296   |  0.322  |   0.310  |  0.260  |  0.311  |  0.000  |
|   K=4   |   0.005   |  0.0144 |   0.043  |  0.012  |  0.006  |  0.000  |

**Comment6: Line 184: WOT starts at 100 epoch while Figure 4 shows WOT starts at 50 epoch. What is the correct experimental setting?.**
- Thank you for your question. For all experiments in our paper, WOT starts at 100-th epoch except for Figure 4. The reason we did this is to show that WOT also works well when starting at early epochs. To avoid misleading, we have changed the setting of Figure 4 to 100 epoch as well.



## Reviewer 7d7c 
We sincerely appreciate your constructive suggestions and detailed comments. Your suggested baselines indeed helps us clarify the unique contribution of our method compared with the previous work! We provide point-wise responses to your concerns.  If there is anything unclear or you wish for additional clarification, please let us know. If you believe that our response has successfully addressed your concerns, please kindly consider increasing your score.

**Comment1: In practice the three baselines missing in order to answer those two questions are: Keep the same hold-out dataset just for pure evolution and use it to choose the early stopping point. The results in Fig 4 suggest that the vanilla version would have been as good as the full WOT solution if stopped at the right moment. Can you show that the stopping point identify by the hold-out dataset is not as good? Keep the same hold-out dataset but instead of learning the coefficients try to simply continue the training on it for the same amount of steps. The data in the hold-out will be seen less frequently than the rest of the training data. This experiment would show if the hold-out datasets is sufficient to gain such robustness or if the specific idea of averaging trajectories is the key factor. The last baseline would show if it is important to learn the coefficients the way it is proposed, rather than using a simpler approach. For example one could use the Polyak exponential moving average (EMA) that would not require any training. In this scenario the hold-out set could be used to tune the hyper parameter of the Polyak EMA or it could simply be merged to the training set and the hyperparemter of the EMA could be treated like all other hyper parameters. Given that the proposed method is a more complicated version of this averaging this baseline seems crucial to justify the additional complexity of the proposed method.**

- Many thanks for your constructive suggestions, which indeed helps us clarify the unique contribution of our method compared with the previous work. We included the four baselines as you suggested with PreActResNet-18 model on CIFAR-10 and CIFAR-100. The  experimental settings of these four baselines are as follows:
  - **AT+early stop:** Take the same hold-out set to choose the early stopping point.
  - **AT+optimizing on validation set:** Keep the same hold-out set and train model on the hold-out set for the same steps. 
  - **AT+EMA:** Combine EMA with AT. The hypyparameter for EMA $\beta$ is set to 0.999 following the official introduction website given by the authors: {https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage}
  - **AT+tuned EMA:** Combine AT with the tuned EMA (the hyperparameter of EMA is tuned by the hold-out set).
- The results are reported in Table S1 and Table S2 and demonstrate:
  - WOT outperforms **AT+early stop** by up to 3.38\% on CIFAR-10 and to 2.13\% on CIFAR-100, indicating that the stopping point identify by the hold-out dataset is not as good as our method.
  - WOT outperforms **AT+optimizing on validation set** by a large margin, confirming that the operation of averaging on historical trajectories plays a key role for our method.
  - WOT-B outperforms either **AT+EMA** or **AT+tuned EMA**, indicating that our learning-based re-weighting is superior than the simple averaging technique used by EMA.

**Table S1** Robust Accuracy (\%) of various methods under various attacks on CIFAR-10.
|  Methods |  PGD-20  |  PGD-100  |  C\&W$_\infty$  |  AA-L$_\infty$  |
| -------------  | :-----------: |:-----------:|:-----------:| :-----------: |
|   AT+early stop                         |   52.90   |   51.90  |  50.90    |  47.43  |
|   AT+optimizing on validation set       |   49.68   |   47.44  |  49.04    |  45.26  |
|   AT+EMA                                |   52.88   |   52.24  |  50.75    |  47.94  |
|   AT+ tuned EMA                         |   52.74   |   51.28  |  51.31    |  48.22  |
|   WOT-B (Gaps:m=400,Number of Gaps:K=4)  |   54.85   |   53.77  |  52.56    |  48.96  |
|   WOT-B (Gaps:m=600,Number of Gaps:K=4)  |   **56.13**   |   **55.28**  |  **52.95**    |  **49.51**  |

**Table S2** Robust Accuracy (\%) of various methods under various attacks on CIFAR-100.
|  Methods |  PGD-20  |  PGD-100  |  C\&W$_\infty$  |  AA-L$_\infty$  |
| -------------  | :-----------: |:-----------:|:-----------:| :-----------: |
|   AT+early stop                         |   28.11   |   27.54  |  26.06    |  23.69  |
|   AT+optimizing on validation set       |   24.45   |   23.45  |  23.75    |  20.98  |
|   AT+EMA                                |   29.83   |   29.22  |  27.60    |  24.88  |
|   AT+ tuned EMA                         |   29.66   |   28.60  |  27.34    |   24.32 |  
|   WOT-B (Gaps:m=400,Number of Gaps:K=4)  |   29.92   |   29.50  |  28.10    |  25.26  |
|   WOT-B (Gaps:m=600,Number of Gaps:K=4)  |   **30.10**   |   **29.93**  |  **28.19**    |  **25.33**  |

**Comment2: A study on the effect of the size of the held-out set could also be interesting.**
- Thank you for your constructive suggestion. We added the requested results of WOT-B (Gaps: m=400, Number of Gaps: K=4) with various sizes of hold-out data in Table S3 and the experiments are conducted with PreActResNet-18 on CIFAR-10.  We see that the robust accuracy decreases as the the size of hold-out data increases, specifically decreasing from 55.79\% to 54.58\% with the size increasing from 1000 to 8000.

**Table S3** Robust and clean accuracy (\%) under PGD-10 on CIFAR-10 with various sizes of hold-out set. 
|Size | Clean Accuracy | Robust Accuracy |
| -------------  | :-----------: |:-----------:|
|  1000  |   83.97   |  55.83  | 
|  2000  |   84.30   |  55.56  | 
|  4000  |   83.90   |  55.10  |
|  8000  |   84.17   |  54.58  |

**Comment3: Could you elaborate on why using $\alpha^i \in [0,1]$ is preventing the final weight from being too far from their initial value? If $\alpha^i \forall i$ is set to 1 couldn’t the $\Delta \omega$ become big enough to push the weights far away from their original value? Could it be that the real reason why the weights don’t move too far is not because of the above constraint but because of the momentum update?.**
- Thank you for your concerns. We constrain $\alpha$ to [0, 1] such that the refined optimization trajectories will not go too far away from the optimization trajectories. In contrast, we conducted experiments without such constraint on CIFAR-10/CIFAR-100 with PreActResNet-18. The results are reported in Table 4 and show that removing such constraint leads to $NaN$ loss, verifying our claim.
-  On the other hand, setting all $\alpha$ to 1 means that the refined optimization trajectories is exactly the same as the original optimization trajectories before refining (can be observed in our submission: Eq. (2)).
-  We clarify that the momentum update is not the reason for the success of our method, since the momentum buffer is reset after each time of weight refining.  To address your concerns, we also report the robust accuracy for WOT-B without resetting momentum buffer in Table 5. The performance difference is marginal (with momentum resetting achieves slightly higher accuracy than without). Hence, the momentum likely does not have a significant influence on our method.

**Table S4** Robust Accuracy (\%) under PGD-10 attack for WOT,WOT without constraints for $\alpha$. Number of Gaps: K=4 for WOT-B. NaN denotes that the refining process leads to NaN loss.
|Methods | CIFAR-10 | CIFAR-100 |
| -------------  | :-----------: |:-----------:|
|  WOT-B (Gaps:m=400)                 |  55.83  | 30.22  |
|  WOT-B (Gaps:m=800)                 |  **56.22**  | **30.47**  |
|  WOT-B (NO constraints,Gaps:m=400)  |  55.68  |  NaN   |
|  WOT-B (NO constraints,Gaps:m=800)  |  NaN    |  NaN   |

**Table S5** Robust Accuracy under PGD-10 attack for WOT-B With momentum resetting and without momentum resetting. Gaps: m=400 and Number of Gaps: K=4.
|Methods | CIFAR-10 | CIFAR-100 |
| -------------  | :-----------: |:-----------:|
|  WOT-B (w/ momentum resetting)  |  **55.83**  |  **30.22**  |
|  WOT-B(w/o momentum resetting)  |  55.03  |  29.89  |

**Comment4: I find the explanation of WOT-W vs WOT-B difficult to parse.**
- The main difference between WOT-W and WOT-B is that WOT-W learns one $\alpha$ for each cached $\Delta w$ (the difference of parameters in two checkpoints) whereas WOT-B further breakdowns each cached $\Delta w$ into several blocks and learns individual $\alpha$ value for each block. Concretely, we naturally divide ResNet kind of  architectures according to their original block design. For VGG-16, we group the layers with the same width as a block. We have shared the details in Appendix B. Therefore, the learning space of WOT-B is larger than WOT-W, leading to better performance in practice. We hope this provides a clearer explanation.

**Comment5: While the paper is mostly easy to read there are a few imprecisions that should be corrected. E.g., the formulation of the AT-PGD algorithm is presented as min-max but the minimization is missing in front of the formula. Some sentences might need rephrasing: E.g., "All the empirical evidences sufficient suggest that".**
- Thank you! We have added the minimization in the formulation of AT-PGD. Besides, we have proofread the paper and fixed typos and other small language mistakes in the revision.

## Reviewer FB7D
Thank you for reviewing our paper. We provide pointwise responses to your concerns.

**Comment1: Appendix is incomplete, and does not have sufficient exposition.**
- Thanks for pointing out this. We have polished our appendix and added more elaborate experiments  to help better understand our method. The results can be found in the appendix of our revised paper. We also provide a brief summary here for convenience.
  - We visualized the learned weights: $\alpha$ for WOT-B in Appendix H. It shows that WOT-B tends to assign larger weights for middle blocks (Block-2,3,4,5) and smaller weights for the bottom and top blocks (block-1,6). These different weights magnitudes for different blocks shed insights on why WOT-B outperforms WOT-W. 
  - We added multiple baselines to validate the effectiveness of WOT (In Appendix I). It shows that re-weighting historical trajectories plays a key role for our method and our learning-based re-weighting is superior to simple averaging techniques (such as EMA).
  - We conducted experiments to show the rationality for constraining $\alpha$ to $[0,1]$ in Appendix J. It shows that without the constraint for $\alpha$ could lead to worse performance or even cause training collapse.
  - We added experiments to show the effect of the size of hold-out set (validation set) in Appendix K. It shows that a large hold-out set size could lead to a worse result.
  - We added experiments to show the influence of the choice of the validation set in Appendix L. It shows that WOT is not sensitive to the choice of the validation set.

**Comment2: I'm confused by Algorithm 1: do you mean to add $\Delta w$  to $\Delta Ws$  in line 9, not just overwrite delta Ws? Also, some mismatched fonts in if conditions. What is $\Delta s$  and how is it chosen.**
- Thank you for pointing to these issues. Yes, line 9 in Algorithm 1 indeed means adding new $\Delta w$ to $\Delta Ws$. Our previous format can be a bit misleading. We have  corrected the mismatched fonts in the revised version and changed line 9 as $\Delta Ws = \Delta Ws \cup \{ \Delta w\}$. And the $\Delta s$ is a typo which should be $\Delta x$ denoting the adversarial perturbations on x. We have fixed it in the revised version. 

**Comment3: Novelty over simple SWA is not that large. As such, should include SWA comparisons in more of the experiments, such as in Table 2, 3, and 4. Especially since simple SWA does not require separate unseen data, it is currently unclear how much the proposed method actually helps.**
- Our method has several critical design factors compared with SWA, which helps to unleash the superior performance of WOT over SWA.
  - The strategy of re-weighting optimization trajectories is novel, which is well recognized by Reviewer **tLbS** and Reviewer **MrMH**. The fundamental difference between WOT and SWA is that, different from SWA which heuristically averages all the models during training, WOT ensembles different trajectories in a way that maximizes the robust performance on a small \textit{hold-out} validation set. This process involves a learning step on the unseen data that can naturally mitigates the robust overfitting issue, improving the generalization performance of our model.
  
  - As you suggested, we directly compared our method with SWA in Table 2, 3, 4 in the revised paper as you suggested. The new results (also shown in Table S1, Table S2 and Table S3) demonstrate again that WOT outperforms SWA and can better eliminate the robust overfitting issue.
**Table S1** Test robustness under multiple adversarial attacks based on  VGG-16/WRN-34-10 architectures for SWA. The experiments are conducted on CIFAR-10 with AT and Trades. The bold denotes the best performance.
|  Architecture  |  Method      |  C\&W$_\infty$  |   PGD-20  |  PGD-100  |  AA-L$_\infty$  |
|     VGG16      |   AT+SWA      |   47.01         |   49.58   |   49.13   |     43.89       |
|     VGG16      | AT+WOT-B      |   **47.52**     | **50.28** | **49.58** |    **44.10**    |
|     VGG16      | TRADES+SWA    |   45.92         |   48.64   | **47.86** |     44.12       |
|     VGG16      | TRADES+WOT-B  |   **46.21**     | **48.81** |  47.85    |    **44.17**    |
|   WRN-34-10    |   AT+SWA      |   56.04         |   55.34   |   53.61   |     52.25       |
|   WRN-34-10    | AT+WOT-B      | **57.13**       | **60.15** | **59.38** |   **53.89**     |
|   WRN-34-10    | TRADES+SWA    |   54.55         |   54.95   |   53.08   |     51.43       |
|   WRN-34-10    | TRADES+WOT-B  | **56.62**       | **57.92** | **56.80** |   **54.33**     |

**Table S2**  Test robustness under AA-$L_{2}$ and AA-$L_{\infty}$ attacks across SVHN/CIFAR-10/CIFAR-100/Tiny-ImageNet datasets for SWA. The experiments are based on PreActResNet-18 and AT. The bold denotes the best performance.
|   Attack  |   Method      |  SVHN     |  CIFAR-10 |  CIFAR-100 |  Tiny-ImageNet |
| AA-$L_{\infty}$ | AT+SWA   |  40.24    |   48.61   |   23.90    |      17.94     |
| AA-$L_{\infty}$ | AT+WOT-B | **51.83** | **48.96** | **25.26**  |     **18.77**  |
| AA-$L_{2}$      | AT+SWA   |  67.76     |   73.28  |   43.10    |      42.40     |
| AA-$L_{2}$      | AT+WOT-B | **72.80** | **73.39** | **43.32**  |      42.54     |

**Table S3** Test robustness under AA-$L_{\infty}$ attack to show the robust overfitting issue in AT and the effectiveness of WOT in overcoming it. The difference between the best and final checkpoints indicates performance degradation during training and the best checkpoint is chosen by PGD-10 attack on the validation set. The experiments are conducted on CIFAR-10 with PreRN-18/WRN-34-10 architectures.
|   Architectures  |   Method      |  Best Checkpoint  | Final Checkpoint |  Difference |
|    PreRN18       |   AT+SWA      |     48.93         |   48.61          |   -0.32     | 
|    PreRN18       |   AT+WOT-B    |     48.90         |   48.96          |   +0.0      | 
|   WRN-34-10      |   AT+SWA      |     53.38         |   52.25          |   -1.13     |    
|   WRN-34-10      |   AT+WOT-B    |     52.23         |   53.89          |   +1.66     | 

  - Moreover, we conducted the three baselines to show the unique contribution of our method, including 1) Keeping the same validation set and training model on the validation set for the same steps. 2) Combining AT with the simple averaging strategy (i.e., vanilla EMA). 3) Combining AT with fintuned EMA where the hyperparameter is learned by the same validation set. The results in Table S4 show that (1) WOT outperforms **AT+optimizing on validation set** by a large margin, confirming that the operation of re-weighting historical trajectories plays a key role for our appealing performance; (2) WOT outperforms either **AT+EMA** or **AT + tuned EMA**, indicating that our learning-based re-weighting is superior than simple averaging techniques.
**Table S4** Robust Accuracy (\%) of various methods under various attacks on CIFAR-10.
|  Methods |  PGD-20  |  PGD-100  |  C\&W$_\infty$  |  AA-L$_\infty$  |
| -------------  | :-----------: |:-----------:|:-----------:| :-----------: |
|   AT+optimizing on validation set       |   49.68   |   47.44  |  49.04    |  45.26  |
|   AT+EMA                                |   52.88   |   52.24  |  50.75    |  47.94  |
|   AT+ tuned EMA                         |   52.74   |   51.28  |  51.31    |  48.22  |
|   WOT-B (Gaps:m=400,Number of Gaps:K=4)  |   54.85   |   53.77  |  52.56    |  48.96  |
|   WOT-B (Gaps:m=600,Number of Gaps:K=4)  |   **56.13**   |   **55.28**  |  **52.95**    |  **49.51**  |

**Comment4: Why did you constrain $\alpha^i$ to [0,1]? Have you tried constraining alpha further, such as summing to 1? Or taking values in $[0, a] for a < 1?$**
- Thank you for your concerns. We constrain $\alpha$ to [0,1] such that the refined optimization trajectories will not go too far away from the original optimization trajectories. In contrast, if we do not constrain $\alpha$, refining optimization trajectories could lead to worse performance or even cause collapse, i.e., the training loss turning to NaN.
- To alleviate your concerns, we conducted experiments with following settings on CIFAR-10 and CIFAR-100:
    - WOT-B without constraining $\alpha$. (Abbreviated as WOT-B(No constraints)).
    - WOT-B with sum($\alpha$) = 1.
    - WOT-B with max($\alpha$) = 0.5,0.8.
- The results are reported in Table S5 where we can see that without our constraint, the loss of WOT-B is **NaN**. We empirically find that neither constraining  nor constraining max($\alpha$)<1 can match the performance of our default setting. Hence, We think it is reasonable to stick to our default option.
**Table S5** Robust Accuracy under PGD-10 attack for WOT,WOT without constraints for $\alpha$, WOT with the constraint by setting sum($\alpha$) =1 or max($\alpha$)<1. NaN denotes the refining process leads to a NaN loss.
|Methods | CIFAR-10 | CIFAR-100 |
| -------------  | :-----------: |:-----------:|
|  WOT-B (Gaps:m=400)                 |  55.83  | 30.22  |
|  WOT-B (Gaps:m=800)                 |  **56.22**  | **30.47**  |
|  WOT-B (sum($\alpha$)=1             |   53.12  |      28.43    |
|  WOT-B (max($\alpha$)=0.8)          |   55.18  |      29.83    |
|  WOT-B (max($\alpha$)=0.5)          |   55.34  |      29.66    |
|  WOT-B (NO constraints,Gaps:m=400)  |  55.68   |       NaN   |
|  WOT-B (NO constraints,Gaps:m=800)  |  NaN     |       NaN   |


**Comment5: Please put number of gaps=1 as a data point in Figure 5 as well, would be interesting to compare, especially since increasing number of gaps does not seem to improve performance much in this case.**
- Thank you for your suggestion. We included the number of gaps K=1 as data point in Figure 5 (in our paper). We also report it here in Table S6 for convenience. The results still support our claim in the  paper that WOT is not sensitive to the number of Gaps:K.
**Table S6** Robust Accuracy under AA$_\infty$ attack with varying number of gaps:K.
|Methods | K=1 | K=2 | K=4 | K=6 | K=8 |
| -------------  | :-----------: |:-----------:| -------------  | :-----------: |:-----------:|
|  WOT-B | 48.88 | 48.96 | 49.08 | 48.88 | 49.03 |
|  WOT-W | 48.25 | 48.47 | 48.36 | 48.75 | 48.29 |


## Reviewer MrMH 
Thank you for your positive and encouraging review! We agree that it is important to discuss how expensive our method is memory-wise. In fact, the extra memory cost for WOT is marginal. We provide detailed answers to your comments below.  We hope that we have addressed them successfully and you can kindly consider increasing your scores and champion for our paper to be accepted.

**Comment1: The proposed method is expensive memory wise (see the limitations below).So, it would be nice to have some numbers about the memory usage to quantify it.**
- Thank you for the suggestion. As you suggested, we included the memory cost of WOT in our revised paper and also reported it here. we report the memory cost of WOT (Gaps:m=400 Number of K=4) with VGG-16 and PreActResNet-18 in Table S1. Even though we need extra memory to cache $\Delta w$, the memory overhead in practice is quite small as the majority of memory usage is concentrated on the intermediate feature maps and Gradient maps, accounting for 81\% of memory usage on AlexNet and 96\% on VGG-16 for an example [1].
**Table S1** Momory cost of WOT (Gaps:m=400,Number of Gaps: K=4)
|Architectures | WOT-B | Vanilla AT |
| -------------  | :-----------: |:-----------:|
|    VGG16       |     3771M     |    3409M    |
| PreAcResNet-18 |     4635M     |    4213M    |


[1] [vDNN: Virtualized deep neural networks for scalable, memory-efficient neural network design. 2016.](https://ieeexplore.ieee.org/document/7783721)

**Comment2: In table 4, the final robust accuracy is sometimes higher than the "best" robust accuracy. How is the "best" checkpoint chosen? With another attack than AutoAttack?**
- Thank you for your review. The 'best' checkpoint is chosen by a weaker attack: PGD-10 with random initialization. While AA attack used in Table 4 is a much stronger attack. It has been shown that a higher accuracy under weaker attack ,e.g. PGD-10,  does not mean there must have a higher accuracy under AA attack [1]. " The the final robust accuracy is sometimes higher than the 'best' robust accuracy" also occurs in previous study, e.g. in Table 4 of the paper [2].
  
[1] [Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks. ICML2020.](https://arxiv.org/abs/2003.01690)
[2] [Robust Overfitting may be mitigated by properly learned smoothening. ICLR2021.](https://openreview.net/pdf?id=qZzy5urZw9)

**Comment3: It would be interesting to check the influence of the choice of the validation set (used for the trajectory refining). Maybe by reporting some standard deviations on the robust accuracy with runs using different samples in the validation set.**
- Thank you for your insightful suggestion. We report the standard deviation of robust accuracy among the three repeated runs based on different validation set on CIFAR-10 and CIFAR-100 respectively with WOT-B. The robust accuracy is calculated under CW$_\infty$ attack. The different validation set are randomly sampled from CIFAR-10/100 with different seed.
- The results (In Table~\ref{tab:diffval}) show that the standard deviations are still less than 0.3\%, in line with the statement in our paper.
**Table S2** The standard deviations on the robust accuracy with three runs using different samples in the validation set.
| Validation set index | Robust Accuracy(CIFAR-10) | Robust Accuracy (CIFAR-100)|
| -------------  | :-----------: |:-----------:|
|      set 1   |   52.01   |   27.04   |
|      set 2   |   52.36   |   27.14   |
|      set 3   |   52.11   |    27.3   |
|       mean   |   52.16   |  27.16    |
|       std    |    0.18   |  0.13     |
 




