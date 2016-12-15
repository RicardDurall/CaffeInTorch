#Places25
<p>This folder contains the files to run CNN using a modified  <a href="http://places.csail.mit.edu/">PLaces205 dataset</a>. In our case, there are 25 different classes to classify ('airport_terminal','aqueduct','bookstore','bridge','castle','cathedral','cemetery','gas_station','harbor','highway','igloo','jail_cell','lighthouse','office_building','palace','parking_lot','residential_neighborhood','skyscraper','slum','snowfield','stadium','subway_station/platform','supermarket','swimming_pool/outdoor','windmill'), all of them originally from <a href="http://places.csail.mit.edu/">PLaces205 dataset</a> It has been used a pre-tained  AlexNet  model from <a href="https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet">Berkeley Vision and Learning Center</a>. Fine-tunning is applied after pool2 layer.</p>

<p>&#9658; main.lua (It has the main structure of the CNN)</p>

<p>&#9658; process.lua (It has all the functions implemented)</p>


<h4>How to run the program the first time:</h4>
<pre>th main.lua -initializeAll</pre>

<h4>If the program has already run once, then there is placesCaffe.t7 which will be load:</h4>
<pre>th main.lua -netwwork main/places25Caffe.t7</pre>

<p>The code posted in this site was originally from <a href="https://github.com/szagoruyko/loadcaffe">szagoruyko</a> and <a href="https://github.com/torch/demos">train-on-housenumbers</a>. Nevertheless, it has been modified accordingly to the goals.</p>
