mainCaffe.lua  It has the main structure of the CNN.

process.lua  It has all the functions implemented.

How to run the program the first time:

th mainCaffe.lua -initializeAll (all layers will freeze their weights but the last ones which would be initialized)

If we have already run the program and we have houseCaffe.t7:

th mainCaffe.lua -netwwork mainCaffe/houseCaffe.t7

The code posted in this site was originally from szagoruyko and train-on-housenumbers. Nevertheless, it has been modified accordingly to the goals.
