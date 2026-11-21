#!/bin/bash
# Build the camera-ready PDF (run from this directory).
set -e
pdflatex -interaction=nonstopmode paper.tex >/dev/null || true
bibtex paper >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null || true
pdflatex -interaction=nonstopmode paper.tex > build.log 2>&1 || true
grep -E "^!|undefined|Overfull" paper.log | sort | uniq || true
pdfinfo paper.pdf | grep Pages
