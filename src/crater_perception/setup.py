from setuptools import setup

package_name = 'crater_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kaviarasu',
    maintainer_email='kaviarasu666@gmail.com',
    description='Crater detection for lunar navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crater_detector = crater_perception.crater_detector:main',
            'test_publisher = crater_perception.test_publisher:main',
            'evaluator = crater_perception.evaluation:main',
            'crater_visualizer = crater_perception.crater_visualizer:main',
        ],
    },
)