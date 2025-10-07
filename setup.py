from setuptools import setup

setup(name="DeepFV",  # Will be normalized to 'deepfv' on PyPI
      version='0.2.5',
      description='Fisher Vectors based on Gaussian Mixture Model with TensorFlow deep learning support',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/sidhomj/DeepFV',
      author='John-William Sidhom',
      author_email='',
      maintainer='John-William Sidhom',
      license='MIT',
      packages=['DeepFV'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
        'numpy>=1.13.3',
        'scipy',
        'scikit-learn>=0.24.0',
        'tensorflow>=2.0.0',
        'matplotlib>=3.0.0'
      ],
      python_requires='>=3.6',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
      ],
      keywords='fisher-vectors, gaussian-mixture-model, deep-learning, tensorflow, computer-vision',
      zip_safe=False)