# Regularized Diffusion Schr&ouml;dinger Bridge with Applications to Score-Based Generative Modeling

The implementation of regularized Schrodinger Bridge Diffusion was adapted based on IPF implementation by Valentin De Bortoli , James Thornton, Jeremy Heng, and Arnaud Doucet.

Original Article:
```
    @article{de2021diffusion,
              title={Diffusion Schr$\backslash$" odinger Bridge with Applications to Score-Based Generative Modeling},
              author={De Bortoli, Valentin and Thornton, James and Heng, Jeremy and Doucet, Arnaud},
              journal={arXiv preprint arXiv:2106.01357},
              year={2021}
            }
```

Installation
------------

This project can be installed from its git repository. 

  
1. Install:

    `conda env create -f conda.yaml`
    
    `conda activate bridge`



How to use this code?
---------------------

3. Train Networks:
  - 2d:  `python main.py dataset=2d model=Basic num_steps=20 num_iter=5000 +lam=<LAMBDA FACTOR>`


FUTURE WORKING: For future implementation: Whilst applying L1 regularization on MNIST and CELEBA can be done, identifcation of the ideal hyperparameters is still unkown and current work in progress.
  - mnist `python main.py dataset=stackedmnist num_steps=30 model=UNET num_iter=5000 data_dir=<insert filepath of data dir <local paths/data/>`
  - celeba `python main.py dataset=celeba num_steps=50 model=UNET num_iter=5000 data_dir=<insert filepath of data dir <local paths/data/>`

Checkpoints and sampled images will be saved to a newly created directory. If GPU has insufficient memory, then reduce cache size. 2D dataset should train on CPU. MNIST and CelebA was ran on 2 high-memory V100 GPUs.
    

References
----------

.. [1] Hans F&ouml;llmer
       *Random fields and diffusion processes*
       In: École d'été de Probabilités de Saint-Flour 1985-1987

.. [2] Christian Léonard 
       *A survey of the Schr&ouml;dinger problem and some of its connections with optimal transport*
       In: Discrete & Continuous Dynamical Systems-A 2014

.. [3] Yongxin Chen, Tryphon Georgiou and Michele Pavon
       *Optimal Transport in Systems and Control*
       In: Annual Review of Control, Robotics, and Autonomous Systems 2020

.. [4] Aapo Hyv&auml;rinen and Peter Dayan
       *Estimation of non-normalized statistical models by score matching*
       In: Journal of Machine Learning Research 2005

.. [5] Yang Song and Stefano Ermon
       *Generative modeling by estimating gradients of the data distribution*
       In: Advances in Neural Information Processing Systems 2019

.. [6] Jonathan Ho, Ajay Jain and Pieter Abbeel
       *Denoising diffusion probabilistic models*
       In: Advances in Neural Information Processing Systems 2020
