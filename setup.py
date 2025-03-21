import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="retrieval-framework",
    version="0.1.0",
    author="Retrieval Framework Team",
    author_email="info@retrievalframework.org",
    description="A modular Python framework for document retrieval systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/retrievalframework/retrieval-framework",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0,<3.0.0",
        "python-dotenv>=1.0.0,<2.0.0",
        "tenacity>=8.0.0,<9.0.0",
        "numpy>=1.22.0,<2.0.0",
        "openai>=1.0.0,<2.0.0",
        "cohere-api>=4.0.0,<5.0.0",
        "pinecone-client>=2.2.0,<3.0.0",
        "groq>=0.4.0,<1.0.0",
        "loguru>=0.7.0,<1.0.0",
    ],
) 