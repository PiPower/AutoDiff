Automatic differentiation library for deep learning purposes.
Library exposes api for creating computation graphs, that will be compiled 
to sequence of ops and run on GPU

Example of creating graph that will be a 2d convolution
```c++
GlorotUniform* weight_init = new GlorotUniform(newDepth, newDepth, TensorType::float32);
weight = new Variable({newDepth, newDepth, 3, 3}, weight_init);
model = new Conv2D(model, weight, {1,1}, {1,1} );
model = new Activation(model, ActivationType::relu);
```

Result of training resnet on mnist dataset (test/MnistModel/Resnet.cpp)

<img width="300"  src="https://github.com/PiPower/AutoDiff/blob/master/resnet.png">