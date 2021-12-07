from setuptools import setup, find_namespace_packages


setup(
    name='vsdkx-yolo-facemask',
    url='https://github.com/natix-io/vsdkx-model-yolo-facemask.git',
    author='Helmut',
    author_email='helmut@natix.io',
    namespace_packages=['vsdkx', 'vsdkx.model'],
    packages=find_namespace_packages(include=['vsdkx*']),
    dependency_links=[
        'git+https://github.com/natix-io/vsdkx-core#egg=vsdkx-core'
    ],
    install_requires=[
        'vsdkx-core',
        'argparse',
        'opencv-python~=4.2.0.34',
        'pyyaml',
        'numpy==1.18.5',
        'tensorflow==2.3.1',

    ],
    version='1.0',
)
