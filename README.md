# BPTC+NOSO
This repository is the official implementation of BPTC+NOSO: Backpropagation of errors based on temporal code with neurons that only spike once at the most.
BPTC is backpropagation of errors based on temporal code, which is mathematically rigorous given that no approximations of any gradient evaluations are used. When combined with neurons that spike once at the most (NOSOs), BPTC+NOSO highlights the following advantages of learning efficiency: (i) computational complexity for learning is independent of the input encoding length, and (ii) only few NOSOs are active during learning and inference periods, leading to large reduction in computational complexity. 

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

### Dataset
MNIST: dataset

[N-MNIST](https://www.garrickorchard.com/datasets/n-mnist) dataset loader was adapted from [PySNN](https://github.com/BasBuller/PySNN) and modified.

## Training
To train a folded SNN (FCN) using BPTC on MNIST, run this command:
```train
cd SNN_folded
python main_FCN_folded_MNIST.py
```

To train unfolded SNNs (FCN and CNN) using BPTC on MNIST and N-MNIST, run this command:
```train
cd SNN_unfolded
python main_unfolded.py --task <mnist or nmist> --network <fcn or cnn> --eval_mode False
```

## Evaluation
To evaluate a folded SNN (FCN) on MNIST, run this command:
```evaluation
cd SNN_folded
python evaluation_FCN_folded_MNIST.py
```

To evaluate unfolded SNNs (FCN and CNN) on MNIST or N-MNIST, run this command:
```evaluation
cd SNN_unfolded
python main_unfolded.py --task <mnist or nmist> --network <fcn or cnn> --eval_mode True
```

## Results
Our model achieves the following performance on: 
MNIST dataset
| Method   | Network                    | Accuracy (%) | Average # spikes per neuron and sample (learning)  |
| -------- |--------------------------- | ------------ |----------------------------------------------------|
| BPTC+NOSO| 784-400-10                 | 98.09%       |  0.340                                             |
| BPTC+NOSO| 28x28-12C5-2MP-64C5-2MP-10 | 99.01%       |  0.209                                             |

N-MNIST dataset
| Method   | Network                      | Accuracy (%) | Average # spikes per neuron and sample (learning)  |
| -------- |----------------------------- | ------------ |----------------------------------------------------|
| BPTC+NOSO| 34x34x2-800-10               | 97.22%       |  0.234                                             |
| BPTC+NOSO| 34x34x2-12C5-2MP-64C5-2MP-10 | 98.67%       |  0.214                                             |
