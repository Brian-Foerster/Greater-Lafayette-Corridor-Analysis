.PHONY: install test run viewer deploy clean

install:
	pip install -e .
	pip install -r requirements-dev.txt

test:
	python -m pytest tests/ -q

run:
	python scripts/optimized_corridor_search.py --iterations 25 --output 40
	python scripts/run_feedback_loop.py --all-scenarios --brt-compare --serve
	python scripts/apm_corridor_evaluation_integrated.py

viewer:
	python scripts/run_feedback_loop.py --all-scenarios --brt-compare --serve

deploy: run
	cp data/processed/corridor_viewer.html docs/viewer.html
	@echo "Viewer copied to docs/viewer.html — commit and push to update GitHub Pages"

clean:
	rm -rf data/processed/*.csv data/processed/*.json data/processed/*.html
	rm -rf data/processed/*.geojson data/processed/*.pkl
