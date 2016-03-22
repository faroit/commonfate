import setuptools

if __name__ == "__main__":
    setuptools.setup(
        # Name of the project
        name='commonfate',

        # Version
        version="0.1.1",

        # Description
        description='Common Fate Transform and Model',

        url='https://github.com/aliutkus/commonfate',

        # Your contact information
        author='Antoine Liutkus',
        author_email='antoine.liutkus@inria.fr',

        # License
        license='BSD',

        # Packages in this project
        # find_packages() finds all these automatically for you
        packages=setuptools.find_packages(exclude=['tests']),

        # Dependencies, this installs the entire Python scientific
        # computations stack
        install_requires=[
            'numpy>=1.7',
            'scipy>=0.9',
            'tqdm',
        ],

        extras_require={
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
            'docs': [
                'sphinx',
                'sphinx_rtd_theme',
                'numpydoc',
            ]
        },

        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Telecommunications Industry',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis'
        ],

        zip_safe=False,
    )
