.PHONY: all clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
	cp build/main.pdf build/SAFEPlus-IJCAI2027-draft.pdf

clean:
	latexmk -C -outdir=build main.tex
	rm -f build/SAFEPlus-IJCAI2027-draft.pdf
