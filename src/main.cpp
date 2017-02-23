#include <iostream>
#include <Network.h>
#include <BackpropagationTrainer.h>

int main() {
  unsigned short *layers = new unsigned short[3]{2, 3, 1};
  Cortex::Network *net = new Cortex::Network(layers, 3);
  Cortex::BackpropagationTrainer *trainer = new Cortex::BackpropagationTrainer();
  trainer->setNetwork(net);

  float **inputSet = new float*[4]{
    new float[2]{1.0f, 1.0f},
    new float[2]{1.0f, 1.0f},
    new float[2]{1.0f, 0.0f},
    new float[2]{0.0f, 1.0f}
  };

  float **desiredSet = new float*[4]{
    new float[1]{0.0f},
    new float[1]{0.0f},
    new float[1]{1.0f},
    new float[1]{1.0f}
  };

  trainer->train(inputSet, desiredSet, 4);
  float *output = new float[1] {1.0f};

  for(int i = 0; i < 4; i++) {
    net->run(inputSet[i], output);
    std::cout << inputSet[i][0] << ", " << inputSet[i][1] << " = " << output[0] << std::endl;
  }

  for(int i = 0; i < 4; i++) {
    delete[] inputSet[i];
    delete[] desiredSet[i];
  }

  delete trainer;
  delete net;
  delete[] inputSet;
  delete[] desiredSet;
  delete[] output;

  return 0;
}
