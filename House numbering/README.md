#House numbering
<p>This folder contains the files to run CNN using House numbering dataset.</p>

<p>mainCaffe.lua  It has the main structure of the CNN.</p>

<p>process.lua  It has all the functions implemented.</p>

How to run the program the first time:

<p>th mainCaffe.lua -initializeAll (all layers will freeze their weights but the last ones which would be initialized)</p>

<p>If we have already run the program and we have houseCaffe.t7:</p>

<p>th mainCaffe.lua -netwwork mainCaffe/houseCaffe.t7</p>

<p>The code posted in this site was originally from <a href="http://www.w3schools.com/html/">szagoruyko</a> and <a href="http://www.w3schools.com/html/">train-on-housenumbers</a>. Nevertheless, it has been modified accordingly to the goals.</p>
