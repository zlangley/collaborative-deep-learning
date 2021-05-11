process_data:
	mkdir -p data/processed/citeulike-a
	mkdir -p data/processed/citeulike-t
	python src/data/preprocess.py

train:
	python src/main.py -v
