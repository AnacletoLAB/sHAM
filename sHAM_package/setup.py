import setuptools

setuptools.setup(

     name='sHAM',

     version='1.0',

     packages=['sHAM',
                'sHAM.test'],

     scripts=['sHAM/compressed_nn.py', 
      'sHAM/huffman.py',
      'sHAM/nu_CWS.py',
      'sHAM/nu_pruning_CWS.py',
      'sHAM/nu_pruning_PWS.py', 
      'sHAM/nu_pruning.py',
      'sHAM/nu_PWS.py',
      'sHAM/pruning_uCWS.py', 'sHAM/pruning_uECSQ.py',
      'sHAM/pruning_uPWS.py', 'sHAM/pruning_uUQ.py',      
      'sHAM/pruning.py',
      'sHAM/sparse_huffman.py', 
      'sHAM/sparse_huffman_only_data.py',
      'sHAM/uCWS.py','sHAM/uPWS.py',
      'sHAM/uECSQ.py','sHAM/uUQ.py'] ,

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
