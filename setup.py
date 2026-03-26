from setuptools import find_packages, setup


VERSION = "0.0.1"

setup(
    name="Gram Newton-Schulz",
    version=VERSION,
    author="Jack Zhang, Noah Amsel, Berlin Chen, Tri Dao",
    description="Fast Newton-Schulz Algorithm with Kernels",
    url="https://github.com/Dao-AILab/gram-newton-schulz",
    packages=find_packages("./"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.7.1",
        "quack-kernels @ git+https://github.com/Dao-AILab/quack.git@490a300b09981fe9565c82ff64d5448a6bc1bb7d",
        "nvidia-cutlass-dsl==4.4.1",
    ],
)
