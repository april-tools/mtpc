#!/bin/bash

python figure1_tikz.py --csv mtpc2.csv > example.tex
pdflatex example.tex
mv example.pdf mtpc.pdf
rm example.*

for file in "mtpc3-evabyte-argmax-nolora" "mtpc3-evabyte-sampling-nolora" "mtpc3-llama-argmax-nolora" "mtpc3-llama-sampling-nolora"
do
	python figure1_tikz.py --csv "$file.csv" > example.tex
	pdflatex example.tex
	mv example.pdf "fig3-$file.pdf"
	rm example.*
done
