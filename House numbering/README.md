#House numbering
<p>This folder contains the files to run CNN using House numbering dataset.</p>
<pre>
mainCaffe.lua  (It has the main structure of the CNN)
process.lua  (It has all the functions implemented)
</pre>
<u>How to run the program the first time:</u>
<pre>
th mainCaffe.lua -initializeAll (all layers will freeze their weights but the last ones which would be initialized)
</pre>
<u>If we have already run the program and we have houseCaffe.t7:</u>
<pre>
th mainCaffe.lua -netwwork mainCaffe/houseCaffe.t7
</pre>
<p>The code posted in this site was originally from <a href="https://github.com/szagoruyko/loadcaffe">szagoruyko</a> and <a href="https://github.com/torch/demos">train-on-housenumbers</a>. Nevertheless, it has been modified accordingly to the goals.</p>
