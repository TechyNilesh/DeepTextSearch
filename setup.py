from setuptools import setup
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
  long_description_content_type="text/markdown",
  name = 'DeepTextSearch',         
  packages = ['DeepTextSearch'],
  version = '0.2',
  license='MIT',        
  description = 'Deep Text Search is an AI-powered multilingual text search and recommendation engine with state-of-the-art transformer-based multilingual text embedding (50+ languages).',
  long_description=README,
  author = 'Nilesh Verma',                   
  author_email = 'me@nileshverma.com',     
  url = 'https://github.com/TechyNilesh/DeepTextSearch',
  download_url = 'https://github.com/TechyNilesh/DeepTextSearch/archive/refs/tags/v_02.tar.gz',    
  keywords = ['Deep Text Search Engine', 'AI Text search', 'Text Search Python','Text Recommendation Engine'],   
  install_requires=[        
          'sentence_transformers',
          'pandas',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
