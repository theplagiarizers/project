.PHONY: install

install:
	python -m pip install --upgrade pip
	pip install -U pyopenssl cryptography
	pip install dvc==2.10.2 dvc-s3 pandas
	pip install --force-reinstall -v "fsspec==2022.11.0"