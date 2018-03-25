#!/bin/bash

. header

infile01=$DDIR/sabah/vocab01.txt
tmpfile01=$DDIR/sabah/vocab01_words.txt
outfile01=$DDIR/sabah/vocab01_root_mapper01.txt

infile02=$DDIR/cumhuriyet/vocab01.txt
tmpfile02=$DDIR/cumhuriyet/vocab01_words.txt
outfile02=$DDIR/cumhuriyet/vocab01_root_mapper01.txt


cd $BDIR/MorphZemberek/out/artifacts/MorphZemberek_jar

cut -f 2 $infile01 > $tmpfile01
java -jar MorphZemberek.jar $tmpfile01 $outfile01

rm $tmpfile01

cut -f 2 $infile02 > $tmpfile02
java -jar MorphZemberek.jar $tmpfile02 $outfile02

rm $tmpfile02

