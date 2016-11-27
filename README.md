# CaffeInTorch
<p>CNN implemented in Torch7 using pre-trained model from Caffe.</p>

<p><i>mainCaffe.lua</i> <q>  It has the main structure of the CNN</q>.</p>

<p><i>process.lua</i> <q>  It has all the functions implemented</q>.</p>

<p>How to run the program the first time:</p>

th mainCaffe.lua -initializeAll (all layers will freeze their weights but the last ones which would be initialized)

<p>If we have already run the program and we have houseCaffe.t7:</p>
th mainCaffe.lua -netwwork mainCaffe/houseCaffe.t7

<p>The code posted in this site was originally from <a href="https://github.com/szagoruyko/loadcaffe">szagoruyko</a> and <a href="https://github.com/torch/demos"> train-on-housenumbers</a>. Nevertheless, it has been modified accordingly to the goals.</p>
<p></p>
