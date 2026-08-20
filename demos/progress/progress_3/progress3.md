To narrow the gap between the training loss and the validation loss, I conducted several sets of comparative experiments and also cross-referenced the logic of the official codebase:



1. First, I added ==cropping== to the val_data.

- While the **train_data** utilized <u>random cropping</u>, I experimented with two distinct methods for the **val_data**: <u>random cropping</u> and <u>center cropping</u>. In both cases, I standardized the image dimensions to 256×256, which are identical to those of the training set. 

- The results were clear: after applying this standardized cropping, the validation loss dropped directly from approximately 2000 to around 700. However, there was no significant difference in performance between the random cropping and center cropping methods.

  

2.  I also conducted an additional test involving the application of ==data augmentation== to the val_data.

- After adding augmentation, the validation loss showed almost no noticeable improvement, remaining consistently within the 600–700 range. 

- Furthermore, from a logical standpoint, I think we shouldnt apple data augmentation to our val_data. This will obscure the model's true generalization capabilities and renders the validation metrics uninformative for assessing performance on unseen datasets later on. So, I think: the val_data can only size-alignment cropping, with no data augmentation applied.

  

3.  I also tried to adjust the parameters :

- I adjusted including tuning the learning rate, adjusting Batch Normalization settings, increasing data augmentation intensity for the training set, and incorporating regularization techniques such as weight decay, but none have proven effective in significantly narrowing the gap between the two loss values.

- Anyways: its always the training loss continues to decline steadily, whereas the validation loss has remained stubbornly stuck around 700 (with only cropping) since the very first epoch, showing no downward trend throughout the entire training process. 

  

4. I think these might be the problems (I m not sure, Whats your opinon??):

- First, the **CpnU22** model architecture is inherently complex and possesses a relatively high model capacity.

- Second, our current training dataset is relatively **small** in scale. So, the model can easily memorizes the specific details and noise present in the training data rather than learning generalized features of the cells. 

  

5. Also, in their code: `demos/Cell Detection with Contour Proposal Networks.ipynb`.

- They didnt use any cropping and data augmentation to val_data, they also dont have code for plotting or visualizing loss curves. I guess the developers themselves are not overly concerned with the absolute disparity between the training and validation loss values.
- And, their dataset is also small. When I ran experiments using their dataset, I observed the exact same phenomenon: a significant discrepancy between the training and validation losses. It is evident that the developers do not place much emphasis on the numerical loss values, instead, they rely entirely on F1 scores, precision, and recall rates across various IoU thresholds to evaluate performance.



6. What do you think?