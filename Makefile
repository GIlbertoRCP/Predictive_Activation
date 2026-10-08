.PHONY: install test quick experiments report
install:
	pip install -r requirements.txt
test:
	PYTHONPATH=src python -m unittest discover -s tests -v
quick:            ## ~2 min smoke run of the whole pipeline
	PYTHONPATH=src python -m predictive_activation.experiments --quick
experiments:      ## full reproduction of every number in the README (~30 min on 2 cores)
	PYTHONPATH=src python -m predictive_activation.experiments
report:
	PYTHONPATH=src python -m predictive_activation.report
