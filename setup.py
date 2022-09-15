#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
	name='profile_t',
	version='1.0.0',
	description='Profile t calculation for symbolic expressions',
	author='Fabricio Olivetti de Franca, Gabriel Kronberger',
	author_email='folivetti@ufabc.edu.br, Gabriel.Kronberger@fh-hagenberg.at',
	url='https://github.com/folivetti/profile_t',
	packages=find_packages(),
    install_requires=['scipy>=1.9.0', 'numpy>=1.9.0', 'sympy>=1.7.0', 'matplotlib>=2.0.0'],
	)
