# setup.py

import setuptools

setuptools.setup(
    name="my_vertex_training_job",
    version="0.1",
    packages=setuptools.find_packages(),
    include_package_data=True,
    description="Example training application for Vertex AI using TensorFlow 2.1",
    author="Your Name",
    install_requires=[
        # List additional pip dependencies here
        # e.g., "pandas>=1.0.0", "tensorflow==2.1.0"
    ],
    entry_points={
        # Optionally specify console_scripts or similar if needed.
    },
)
