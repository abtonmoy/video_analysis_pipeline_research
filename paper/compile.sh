#!/bin/bash
# Compile AdaFrame paper with proper reference resolution

cd "$(dirname "$0")"

echo "=========================================="
echo "Compiling AdaFrame Paper"
echo "=========================================="
echo ""

# Check if pdflatex exists
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found!"
    echo "Install TeX Live: sudo apt-get install texlive-full"
    exit 1
fi

# Clean old files
echo "Cleaning old files..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

# Compile
echo "Running pdflatex (1/3)..."
pdflatex -interaction=nonstopmode mm2026.tex || { echo "First pdflatex failed"; exit 1; }

echo "Running bibtex..."
bibtex mm2026 || { echo "Bibtex failed (non-fatal)"; }

echo "Running pdflatex (2/3)..."
pdflatex -interaction=nonstopmode mm2026.tex || { echo "Second pdflatex failed"; exit 1; }

echo "Running pdflatex (3/3)..."
pdflatex -interaction=nonstopmode mm2026.tex || { echo "Third pdflatex failed"; exit 1; }

# Check result
if [ -f "mm2026.pdf" ]; then
    echo ""
    echo "=========================================="
    echo "✓ SUCCESS! Paper compiled: mm2026.pdf"
    echo "=========================================="
    ls -lh mm2026.pdf
    echo ""
    echo "PDF Info:"
    pdfinfo mm2026.pdf 2>/dev/null | grep -E "Pages:|File size:" || echo "pdfinfo not available"
else
    echo ""
    echo "=========================================="
    echo "✗ FAILED! Check errors above"
    echo "=========================================="
    exit 1
fi
