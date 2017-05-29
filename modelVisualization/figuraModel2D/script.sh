make
tmpFile=tmpFile
./generateModel 320 240 32 24 4 80 -20 > $tmpFile.tex
latex $tmpFile
dvips $tmpFile.dvi -E -o $tmpFile.ps
epstool --copy --bbox $tmpFile.ps output.ps
rm $tmpFile*
display -alpha off output.ps

