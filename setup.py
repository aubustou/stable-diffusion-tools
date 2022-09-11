from setuptools import setup

setup(
    name="stable-diffusion-tools",
    version="0.1",
    description="Tools for stable diffusion",
    author="aubustou",
    author_email="survivalfr@yahoo.fr",
    requires=[
        "pytorch",
        "tqdm",
        "diffusers",
        "transformers",
        "ftfy",
        "scipy",
        "pillow",
        "numpy",
        "ipython",
        "requests",
    ],
)
