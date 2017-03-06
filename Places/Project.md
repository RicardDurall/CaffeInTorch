#Project
<p>This folder contains sevreal experiments performed over the CNN. In our case, as we mentioned beofre, there are 10 different classes to classify:
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

<img src="https://github.com/RicardDurall/CaffeInTorch/blob/master/Places/network.PNG" alt="CNN" style="width:304px;height:228px;">

<h4>Accuracy results of the CNN:</h4>

<img src="https://github.com/RicardDurall/CaffeInTorch/blob/master/Places/graph.PNG" alt="CNN" style="width:304px;height:228px;">
