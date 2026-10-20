.PHONY: format diagram lint dev

diagram:
	classpy sync
	PLANTUML_LIMIT_SIZE=16384 plantuml -tpng docs/diagrams/*.puml

format:
	docformatter --in-place --recursive --wrap-summaries 88 --wrap-descriptions 88 src/bmiemg
	black src/bmiemg
	black tests/

lint:
	poetry run pylint --disable=C src/bmiemg


dev: diagram format lint
