.PHONY: install plot demo web

install:
	pip install -r requirements.txt

plot:
	python -m analysis.benchmarks --out examples/plots

demo:
	python -m apps.cli simulate --m 5 --t 3 --ber 0.05

web:
	streamlit run apps/web/app.py
