#ifndef _CORTEX_BACKPROPAGATIONTRAINER_H
#define _CORTEX_BACKPROPAGATIONTRAINER_H
#include "Network.h"

namespace Cortex {

  class BackpropagationTrainer {
  public:
  BackpropagationTrainer();
  ~BackpropagationTrainer();
  void setNetwork(Network *network);
  float train(float **input, float **desired, unsigned short length);
  void run(float *input, float *desiredOutput);
  void clear();
  float score(float *desired);
  private:
  Network *network;
  float **errors;
  float *output;
  void calculateOutputError(float *desiredOutput);
  void calculateRecurrentErrorLayer(unsigned short layerIndex);
  void calculateRecurrentErrorNeuron(unsigned short layerIndex, unsigned short neuronIndex);
  void adjustWeights();
  void adjustBiasWeights();
  };
}

#endif
