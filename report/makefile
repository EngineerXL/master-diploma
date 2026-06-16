all: main extra

main:
	latexmk -xelatex main.tex
	mv main.pdf output/

extra:
	latexmk -xelatex \
		consultant_review.tex \
		handout.tex \
		review.tex \
		supervisor_review.tex \
		task.tex
	mv \
		consultant_review.pdf \
		handout.pdf \
		review.pdf \
		supervisor_review.pdf \
		task.pdf \
		output/

clean:
	latexmk -c
	rm -rf output/*.pdf
