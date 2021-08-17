from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as readme_file:
    readme = readme_file.read()

requirements = [
    'numpy==1.20.3',
    'scipy==1.6.3',
    'torch>=1.7.1',
    'torchvision>=0.8.1',
    'tensorboard==2.5.0',
    'pandas==1.2.4',
    'scikit-learn==0.24.2',
    'scikit-image==0.18.1',
    'Pillow==8.2.0',
    'PyYAML==5.4.1',
]

setup(
    author='Conor Horgan',
    author_email='conor.horgan@kcl.ac.uk',
    python_requires='>=3.6, <3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
    ],
    description=(
        'spectrai: a deep learning framework for spectral data.'
    ),
    entry_points={
        'console_scripts': [
            'spectrai_train = spectrai.train:main',
            'spectrai_apply = spectrai.apply:main',
            'spectrai_evaluate = spectrai.evaluate:main',
            'spectrai_preview = spectrai.preview:main'
        ],
    },
    install_requires=requirements,
    license='MIT License',
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='spectrai',
    name='spectrai',
    packages=find_packages(include=['spectrai', 'spectrai.*']),
    setup_requires=[],
    test_suite='tests',
    tests_require=[],
    url='https://github.com/conor-horgan/spectrai',
    version='0.1.0',
)