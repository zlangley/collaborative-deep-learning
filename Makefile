train:
	python src/main.py -v

features: clean bert bow mult

clean:
	rm -rf data/processed
	mkdir -p data/processed/citeulike-a
	mkdir -p data/processed/citeulike-t

bert:
	python scripts/compute_bert.py

bow:
	python scripts/compute_bow.py

mult:
	python scripts/transform_mult.py
