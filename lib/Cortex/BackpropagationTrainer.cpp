#include "BackpropagationTrainer.h"
#include "Network.h"
#include <cstddef>
#include <math.h>
#include <iostream>

/**
* @file BackpropagationTrainer.cpp
* @author Leif Andreas Rudlang
* @version 0.0.1
* @date February, 2017
* @brief Backpropagation trainer for feed-forward artificial neural network
*/
namespace Cortex {

  BackpropagationTrainer::BackpropagationTrainer() {
    this->network = NULL;
  }

  BackpropagationTrainer::~BackpropagationTrainer() {
    this->clear();
  }

  void BackpropagationTrainer::clear() {

    if(this->network == NULL) {
      return;
    }

    unsigned short depth = network->getDepth();
    unsigned short *structure = network->getStructure();

    for(unsigned short i = 0; i < depth; i++) {
      delete[] this->errors[i];
    }

    delete[] this->errors;
    delete[] this->output;
    this->network = NULL;
  }

  void BackpropagationTrainer::setNetwork(Network *network) {
    unsigned short depth = network->getDepth();
    unsigned short *structure = network->getStructure();
    this->clear();
    this->network = network;
    this->errors = new float*[depth];

    for(unsigned short i = 0; i < depth; i++) {
      this->errors[i] = new float[structure[i]];
    }

    this->output = new float[structure[depth - 1]];
  }

  float BackpropagationTrainer::train(float **input, float **desired, unsigned short length) {
    unsigned short i = 0, j = 0, attempts = 1;
    float threshold = 0.05f;
    float score = 0.0f;
    float rate = 0.25f;
    int iterations = 0;

    for(unsigned short k = 0; k < 2000; k++) {
      // run through the sets
      score = 0.0f;
      for(i = 0; i < length; i++) {
        float *inputSet = input[i];
        float *desiredSet = desired[i];
        this->run(inputSet, desiredSet, rate);
        score += this->score(desiredSet);
        iterations++;
      }

      score = std::sqrt(score);
      if(iterations % 100 == 0.0f) {
        //rate = 1.0f - (1000.0f / i) / (1000.0f); if score is not getting lower in X iterations, its time to quit
        //std::cout << score << std::endl;
      }
      if(score < threshold) {
        break;
      }
    }

    std::cout << "total iterations: " << iterations << std::endl;
    return 0.0f;
  }

  void BackpropagationTrainer::run(float *input, float *desired, float rate) {
    unsigned short depth = network->getDepth();
    unsigned short *structure = network->getStructure();

    // Run through the network
    this->network->run(input, this->output);

    // Calculate difference between output and the desired output
    this->calculateOutputError(desired);

    // Calculate error for the hidden layer neurons
    for(unsigned short i = depth - 1; i > 0; --i) {
      this->calculateRecurrentErrorLayer(i - 1);
    }

    // Adjust Synaptic weights
    this->adjustWeights(rate);

    // Adjust Synaptic bias weights
    this->adjustBiasWeights(rate);
  }

  void BackpropagationTrainer::calculateOutputError(float *desired) {
    unsigned short outputSize = this->network->getOutputSize();
    unsigned short depth = this->network->getDepth();
    float *layer = this->errors[depth - 1];

    for(unsigned short i = 0; i < outputSize; i++) {
      float value = this->output[i];
      layer[i] = value * (1.0f - value) * (desired[i] - value);
    }
  }

  void BackpropagationTrainer::calculateRecurrentErrorLayer(unsigned short layerIndex) {
    unsigned short layerSize = this->network->getStructure()[layerIndex];

    for(unsigned short i = 0; i < layerSize; i++) {
      this->calculateRecurrentErrorNeuron(layerIndex, i);
    }
  }

  void BackpropagationTrainer::calculateRecurrentErrorNeuron(unsigned short layerIndex, unsigned short neuronIndex) {
    float recurrent = 1.0f;
    float value = this->network->getNeuronValue(layerIndex, neuronIndex);
    unsigned short *layers = this->network->getStructure();
    unsigned short nextLayerSize = layers[layerIndex + 1];

    for(unsigned short i = 0; i < nextLayerSize; i++) {
      float error = this->errors[layerIndex + 1][i];
      float weight = this->network->getSynapse(layerIndex, neuronIndex, i);
      recurrent = recurrent * weight * error;
    }

    this->errors[layerIndex][neuronIndex] = recurrent * value * (1.0f - value);
  }

  void BackpropagationTrainer::adjustWeights(float rate) {
    unsigned short len = this->network->getDepth() - 1;
    unsigned short* structure = this->network->getStructure();
    unsigned short i = 0, j = 0, k = 0, size = 0, nextSize = structure[0];
    float* error;
    float weight = 0.0f, value = 0.0f;

    for(; i < len; i++) {
      size = nextSize;
      nextSize = structure[i + 1];
      error = this->errors[i + 1];
      for(j = 0; j < size; j++) {
        value = this->network->getNeuronValue(i, j);
        for(k = 0; k < nextSize; k++) {
          weight = this->network->getSynapse(i, j, k);
          this->network->setSynapse(i, j, k, weight + (rate * error[k] * value));
        }
      }
    }
  }

  void BackpropagationTrainer::adjustBiasWeights(float rate) {
    unsigned short depth = this->network->getDepth();
    unsigned short* structure = this->network->getStructure();
    unsigned short i = 1, j = 0, size = structure[0];

    for(; i < depth; i++) {
      unsigned short prevLayerSize = size;
      size = structure[i];
      for(j = 0; j < size; j++) {
        float error = this->errors[i][j];
        float weight = this->network->getSynapse(i-1, prevLayerSize, j);
        this->network->setSynapse(i-1, prevLayerSize, j, weight + (rate * error));

      }
    }
  }

  float BackpropagationTrainer::score(float *desired) {
    unsigned short len = this->network->getOutputSize();
    float delta = 0.0f;

    for(unsigned short i = 0; i < len; i++) {
      float d = desired[i] - this->output[i];
      delta += (d * d);
    }

    return delta;
    //return std::sqrt(delta);
  }

}
