make
tmpFile=tmpFile
./generateModel 320 240 120 32 24 16 3 -40 40 -90 > $tmpFile.tex
latex $tmpFile
dvips $tmpFile.dvi -E -o $tmpFile.ps
epstool --copy --bbox $tmpFile.ps output.ps
rm $tmpFile*
display -alpha off output.ps
