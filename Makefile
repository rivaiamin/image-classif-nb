# Setup venv image_classification_env
setup:
	python -m venv image_classification_env
	source image_classification_env/bin/activate
	pip install -r requirements.txt

train:
	python main.py

test:
	python main.py --test

clean:
	rm -rf __pycache__

feature_extraction:
	python feature_extraction.py