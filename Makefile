.PHONY: download erp audioxcorr encoding prediction controls figures

download:
	python code/download.py

erp:
	python code/erp.py

audioxcorr:
	python code/audioxcorr.py

encoding:
	python code/encoding.py --modelname=gpt2-xl --layer=24
	python code/encoding.py --modelname=en_core_web_lg
	python code/encoding.py --modelname=syntactic
	python code/encoding.py --modelname=phonetic
	python code/encoding.py --modelname=spectral

prediction:
	python code/prediction.py --modelname=gpt2-xl --layer=24 --subject=03

controls:
	python code/controls.py --modelname=gpt2-xl --layer=24 --subject=03

figures:
	jupyter nbconvert --to notebook --inplace --execute notebooks/qualitychecks.ipynb
	jupyter nbconvert --to notebook --inplace --execute notebooks/encoding.ipynb
	jupyter nbconvert --to notebook --inplace --execute notebooks/prediction.ipynb
	jupyter nbconvert --to notebook --inplace --execute notebooks/controls.ipynb
