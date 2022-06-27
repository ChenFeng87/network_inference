# Inferring structural and dynamical properties of gene networks from data with deep learning

We have annotated the code in detail to help the reader understand the code.
## Take the MISA task as an example
There are 8 files under "MISA" file, including 

5 python code files: `data_generation.py`, `train.py`, `Edge_removal.py`, `predict_stable_state.py` and `monotonicity_f.py`;

2 image files: `The_role_X1.png` and `f1_f2_X2.png`;

1 `Parameters_saved.pickle`, save the DNN parameters after training

PS: 1 `data.pickle` after we run `data_generation.py`, which has been divided into three parts, training set (80%), validation set (10%) and test set (10%);

> We should first run `data_generation.py` to Prepare the data for training, we can get a `data.pickle` after about 60 seconds. 

> Then we can run `train.py` for training, of note, we include "epochs" and "sub_epochs" in our parameters, each epoch contains "sub_epochs" training. We apply the DNN model to the validation set after each epoch, i.e., we perform a validation after training "sub_epochs" times (Default "sub_epochs" = 10). 
> 
> After we run `train.py`, we can get the `Parameters_saved.pickle`.
> 
> Then, we can use this DNN model do everything we want, including inferring the structural of gene networks by `Edge_removal.py` (corresponding `The_role_X1.png` can be obtained), getting the monotonicity of the synthesis rate f with respect to the variable x by `monotonicity_f.py` (corresponding `f1_f2_X2.png` can be obtained), and predicting the steady states by  `predict_stable_state.py`.

Other files "Four_dimensional" and "Oscillation" focus on the inference of gene regulatory networks. And in "Four_dimensional" file, We additionally test the performance of DNN on network inference when it has 3 hidden layers.
