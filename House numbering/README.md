#House numbering
<p>This folder contains the files to run CNN using House numbering dataset.</p>
<pre>
mainCaffe.lua  
<small>(It has the main structure of the CNN)</small>
process.lua  
<small>(It has all the functions implemented)</small>
</pre>
<u>How to run the program the first time:</u>
<pre>
th mainCaffe.lua -initializeAll 
<small>(all layers will freeze their weights but the last ones which would be initialized)</small>
</pre>
<u>If we have already run the program and we have houseCaffe.t7:</u>
<pre>
th mainCaffe.lua -netwwork mainCaffe/houseCaffe.t7
</pre>
<p>The code posted in this site was originally from <a href="https://github.com/szagoruyko/loadcaffe">szagoruyko</a> and <a href="https://github.com/torch/demos">train-on-housenumbers</a>. Nevertheless, it has been modified accordingly to the goals.</p>
