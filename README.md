# Improvments to DQNAlign

This code-base is built heavily upon the following repository https://github.com/syjqkrtk/DQNalign  

There are three versions of the codebase, one using Tensorflow 1.12 and one using PyTorch 1.13 and one using Pytorch 1.13 but includes an implementation with Noisy Layers. The TensorFlow version is in the TF folder and the PyTorch version is in the Torch folder. 

It is highly reccomended to use the torch version as it is more recent than TensorFlow 1.12 and is compatible with newer versions of Python

# Requirements
Works best on a linux environment with nucmer installed


Tensorflow version requirements
* Python == 3.5.4
* TensorFlow == 1.12

PyTorch version requirements  
* Python >= 3.10.5
* torch == 1.13
* torchvision == 0.14.0
* torchaudio == 0.13.0



# References
```bib
@misc{DQNAlign,
  doi = {10.48550/ARXIV.2010.13478},
  
  url = {https://arxiv.org/abs/2010.13478},
  
  Author = {Song, Yong Joon and Ji, Dong Jin and Seo, Hye In and Han, Gyu Bum and Cho, Dong Ho},
  
  keywords = {Quantitative Methods (q-bio.QM), Genomics (q-bio.GN), FOS: Biological sciences, FOS: Biological sciences},
  
  title = {Pairwise heuristic sequence alignment algorithm based on deep reinforcement learning},
  
  publisher = {arXiv},
  
  year = {2020},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


# Citation
To cite this codebase please use the github citation feature
