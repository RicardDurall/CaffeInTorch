#House numbering
<p>This folder contains the files to run CNN using House numbering dataset. It has been used a pre-tained  AlexNet  model from <a href="https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet">Berkeley Vision and Learning Center</a>. Fine-tunning is applied only to the fully connected layers.</p>

 <p>&#9658; main.lua (It has the main structure of the CNN)</p>

<p>&#9658; process.lua (It has all the functions implemented)</p>


<h4>How to run the program the first time:</h4>
<pre>th mainCaffe.lua -initializeAll</pre>
(all layers will freeze their weights but the last ones which would be initialized)

<h4>If the program has already run once, then there is houseCaffe.t7 which will be load:</h4>
<pre>th mainCaffe.lua -netwwork mainCaffe/houseCaffe.t7</pre>

<p>The code posted in this site was originally from <a href="https://github.com/szagoruyko/loadcaffe">szagoruyko</a> and <a href="https://github.com/torch/demos">train-on-housenumbers</a>. Nevertheless, it has been modified accordingly to the goals.</p>
