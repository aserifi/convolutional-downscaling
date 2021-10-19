# Spatio-Temporal Downscaling of Climate Data using Convolutional and Error-Predicting Neural Networks 


This folder contains a tensorflow implementation for the DCN and RPN architectures of the paper:
  
  * Agon Serifi, Tobias Günther, Nikolina Ban <br/>
  [Spatio-Temporal Downscaling of Climate Data using Convolutional and Error-Predicting Neural Networks](https://doi.org/10.3389/fclim.2021.656479)
  
The file model.py contains functions to construct the proposed architectures. <br/>
Use 'get_model(residual = False)' for the DCN and 'get_model(residual = True)' for the RPN architecture.

### Dependencies
- Tensorflow

### Citation
If you find our work useful to your research, please consider citing:
```
@article{serifi2021spatio,
  title={Spatio-Temporal Downscaling of Climate Data using Convolutional and Error-Predicting Neural Networks},
  author={Serifi, Agon and G{\"u}nther, Tobias and Ban, Nikolina},
  journal={Frontiers in Climate},
  volume={3},
  pages={26},
  year={2021},
  publisher={Frontiers}
}
```

### License
By downloading and using the code you agree to the terms in the [LICENSE](https://github.com/aserifi/convolutional-downscaling/blob/main/LICENSE). 
