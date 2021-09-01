import setuptools

setuptools.setup(

     name='compressionNN',

     version='1.0',

     packages=['compressionNN',
                'compressionNN.test'],

     scripts=['compressionNN/compressed_nn.py', 
      'compressionNN/huffman.py',
      'compressionNN/nu_CWS.py',
      'compressionNN/nu_pruning_CWS.py',
      'compressionNN/nu_pruning_PWS.py', 
      'compressionNN/nu_pruning.py',
      'compressionNN/nu_PWS.py',
      'compressionNN/pruning_uCWS.py', 'compressionNN/pruning_uECSQ.py',
      'compressionNN/pruning_uPWS.py', 'compressionNN/pruning_uUQ.py',      
      'compressionNN/pruning.py',
      'compressionNN/sparse_huffman.py', 
      'compressionNN/sparse_huffman_only_data.py',
      'compressionNN/uCWS.py','compressionNN/uPWS.py',
      'compressionNN/uECSQ.py','compressionNN/uUQ.py'] ,

     author="Giosuè Marinò, Alessandro Petrini",

     author_email="giosumarin@gmail.com, alessandro.petrini@unimi.it",

     description="Compress neural networks, bruh",

     url="https://github.com/AnacletoLAB/sHAM",

     #packages=setuptools.find_packages(),

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: GPL-3.0",

         "Operating System :: Linux base",

     ],

 )
