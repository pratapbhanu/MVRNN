#!/bin/bash
#pwd should be PROJECT_HOME i.e. root directory of project


pathToStanfordParser=`pwd`/ext/stanford-parser-2011-09-14
dataDir=`pwd`/data/corpus


# Run stanford parser for tree structures
echo
echo "Running Stanford Parser..."
java -mx600m -cp "$pathToStanfordParser/stanford-parser.jar:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -nthreads "$3" -outputFormat "penn" -sentences newline $pathToStanfordParser/grammar/englishPCFG.ser.gz  "$dataDir/$1" > "$dataDir/$2" &

if [ "$?" -ne 0 ]; then echo "stanford parser failed"; exit 1; fi


