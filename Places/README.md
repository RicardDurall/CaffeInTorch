#Places25
<p>This folder contains the files to run CNN using a modified  <a href="http://places.csail.mit.edu/">PLaces205 dataset</a>. In our case, there are 25 different classes to classify:
<table style="width:100%">
  <tr>
    <td>airport_terminal</td>
    <td>aqueduct</td>
    <td>bookstore</td>
    <td>abridge</td>
    <td>castle</td>
  </tr>
    <tr>
    <td>cathedral</td>
    <td>cemetery</td>
    <td>gas_station</td>
    <td>harbor</td>
    <td>highway</td>
  </tr>
    <tr>
    <td>igloo</td>
    <td>jail_cell</td>
    <td>lighthouse</td>
    <td>office_building</td>
    <td>palace</td>
  </tr>  <tr>
    <td>parking_lot</td>
    <td>residential_neighborhood</td>
    <td>skyscraper</td>
    <td>slum</td>
    <td>snowfield</td>
  </tr>
    <tr>
    <td>stadium</td>
    <td>subway_station</td>
    <td>supermarket</td>
    <td>swimming_pool</td>
    <td>windmill</td>
  </tr>
</table>
All of them are originally from <a href="http://places.csail.mit.edu/">PLaces205 dataset</a> It has been used a pre-tained  AlexNet  model from <a href="https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet">Berkeley Vision and Learning Center</a>. Fine-tunning is applied after pool2 layer.</p>

<p>&#9658; main.lua (It has the main structure of the CNN)</p>

<p>&#9658; process.lua (It has all the functions implemented)</p>


<h4>How to run the program the first time:</h4>
<pre>th main.lua -initializeAll</pre>

<h4>If the program has already run once, then there is places25Caffe.t7 which will be load:</h4>
<pre>th main.lua -netwwork main/places25Caffe.t7</pre>

<p>The code posted in this site was originally from <a href="https://github.com/szagoruyko/loadcaffe">szagoruyko</a> and <a href="https://github.com/torch/demos">train-on-housenumbers</a>. Nevertheless, it has been modified accordingly to the goals.</p>
