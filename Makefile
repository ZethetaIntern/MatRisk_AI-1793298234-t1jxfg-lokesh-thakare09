.PHONY: install train dashboard api test lint clean all
install:
	python -m pip install -r requirements.txt
train:
	PYTHONPATH=src python -m matrisk.train_all
dashboard:
	streamlit run app.py --server.port 8501
api:
	uvicorn matrisk.api:app --app-dir src --host 0.0.0.0 --port 8000
test:
	PYTHONPATH=src pytest -q --cov=src/matrisk --cov-report=term-missing
lint:
	python -m compileall -q src app.py tests
clean:
	rm -f models/*.joblib outputs/metrics.json outputs/dataset_summary.csv
all: train test
