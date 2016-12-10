#House numbering
<p>This folder contains the files to run CNN using House numbering dataset.</p>

 <p>&#9658; mainCaffe.lua</p>
(It has the main structure of the CNN)

<pre>process.lua</pre>
(It has all the functions implemented)

<h4>How to run the program the first time:</h4>
<pre>th mainCaffe.lua -initializeAll</pre>
(all layers will freeze their weights but the last ones which would be initialized)

<h4>If we have already run the program and we have houseCaffe.t7:</h4>
<pre>th mainCaffe.lua -netwwork mainCaffe/houseCaffe.t7</pre>

<p>The code posted in this site was originally from <a href="https://github.com/szagoruyko/loadcaffe">szagoruyko</a> and <a href="https://github.com/torch/demos">train-on-housenumbers</a>. Nevertheless, it has been modified accordingly to the goals.</p>
