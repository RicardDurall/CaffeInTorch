#Places10
<p>This folder contains the files to run CNN using a modified  <a href="http://places.csail.mit.edu/">PLaces205 dataset</a> and some <a href="https://github.com/RicardDurall/CaffeInTorch/blob/master/Places/Project.md">results</a> extacted them . In our case, there are 10 different classes to classify:


<table style="width:100%">
  <tr>
    <td>airport_terminal</td>
    <td>lighthouse</td>
    <td>cemetery</td>
    <td>swimming_pool</td>
    <td>windmill</td>
  </tr>
    <tr>
    <td>bookstore</td>
    <td>castle</td>
    <td>gas_station</td>
    <td>highway</td>
    <td>skyscraper</td>
  </tr>
</table>
All of them are originally from <a href="http://places.csail.mit.edu/">PLaces205 dataset</a> It has been used a pre-tained  AlexNet  model from <a href="https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet">Berkeley Vision and Learning Center</a>. Fine-tunning is applied after block 4.</p>

<img src="https://github.com/RicardDurall/CaffeInTorch/blob/master/Places/Imgaes/network.PNG" alt="CNN" style="width:304px;height:228px;">

<h4>Files:</h4>

<p>&#9658; mainCaffe.lua (It has the main structure of the CNN)</p>
aaa

<p>&#9658; process.lua (It has all the functions implemented)</p>


<h4>How to run the program the first time:</h4>
<pre>th mainCaffe.lua -initializeAll</pre>

<h4>If the program has already run once, then there is places25Caffe.t7 which will be load:</h4>
<pre>th mainCaffe.lua -netwwork mainCaffe/places25Caffe.t7</pre>

<p>The code posted in this site was originally from <a href="https://github.com/szagoruyko/loadcaffe">szagoruyko</a> and <a href="https://github.com/torch/demos">train-on-housenumbers</a>. Nevertheless, it has been modified accordingly to the goals.</p>
