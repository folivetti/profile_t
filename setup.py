#!/usr/bin/env python

from setuptools import setup

setup(
	name='profile_t',
	version='1.0.0',
	description='Profile t calculation for symbolic expressions',
	author='Fabricio Olivetti de Franca, Gabriel Kronberger',
	author_email='folivetti@ufabc.edu.br, Gabriel.Kronberger@fh-hagenberg.at',
	url='https://github.com/folivetti/profile_t',
	packages=['profile_t'],
    install_requires=['scipy', 'numpy', 'sympy', 'matplotlib'],
	)