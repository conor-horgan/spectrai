from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as readme_file:
    readme = readme_file.read()

requirements = [
    'numpy==1.23.3',
    'scipy==1.9.1',
    'torch>=1.11.0',
    'torchvision>=0.12.0',
    'tensorboard==2.10.0',
    'pandas==1.4.4',
    'scikit-learn==1.1.2',
    'scikit-image==0.19.3',
    'Pillow==9.2.0',
    'PyYAML==6.0',
]

setup(
    author='Conor Horgan',
    author_email='conor.horgan@kcl.ac.uk',
    python_requires='>=3.6, <3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
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
    license='Apache 2.0',
    license_files=('LICENSE'),
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
    version='0.1.4',
)