#include "Network.h"
#include <math.h>
#include <iostream>

/**
* @file Network.cpp
* @author Leif Andreas Rudlang
* @version 0.0.1
* @date February, 2017
* @brief High-performance Feed-forward artificial neural network
*/
namespace Cortex {

  Network::Network(unsigned short *structure, unsigned short depth, float **valueMatrice, float **weightMatrice) {
    this->structure = structure;
    this->depth = depth;
    this->valueMatrice = valueMatrice;
    this->weightMatrice = weightMatrice;
  }

  Network::Network(unsigned short *structure, unsigned short depth) {
    this->structure = structure;
    this->depth = depth;
    this->valueMatrice = new float*[depth];
    this->weightMatrice = new float*[depth - 1];

    for(unsigned short i = 1; i < depth; i++) {
      unsigned short layerSize = structure[i];
      this->valueMatrice[i] = new float[layerSize];
      this->weightMatrice[i - 1] = new float[layerSize * (structure[i - 1] + 1)];
    }

    this->identity();
  }

  Network::~Network() {
    for(unsigned short i = 1; i < this->depth - 1; i++) {
      delete[] this->valueMatrice[i];
      delete[] this->weightMatrice[i - 1];
    }

    delete[] this->structure;
    delete[] this->valueMatrice;
    delete[] this->weightMatrice;
  }

  /**
  * Initialize the synaptic weights
  * @TODO method to initialize the weights using a "seed"
  */
  void Network::identity() {
    unsigned short i = 0, j = 0, len = this->depth - 1;
    for(; i < len; i++) {
      float *weights = this->weightMatrice[i];
      unsigned short layerSize = (this->structure[i] + 1) * this->structure[i + 1];
      for(j = 0; j < layerSize; j++) {
        weights[j] = 0.5f; // TODO, need function to "Seed" weights, can this be used by calculating complexity?
      }
    }
  }

  void Network::run(float *input, float *output) {
    unsigned short i = 1, j = 0;
    this->valueMatrice[0] = input;
    this->valueMatrice[this->depth - 1] = output;

    for(; i < this->depth; i++) {
      unsigned short layerSize = this->structure[i];

      for(j = 0; j < layerSize; j++) {
        this->valueMatrice[i][j] = this->runNeuron(i, j);
      }
    }
  }

  float Network::runNeuron(unsigned short layer, unsigned short neuron) {
    unsigned short prevLayerIndex = layer - 1;
    unsigned short prevLayerSize = this->structure[prevLayerIndex];
    float *prevLayer = this->valueMatrice[prevLayerIndex];
    float sum = Network::BIAS * this->getSynapse(prevLayerIndex, prevLayerSize, neuron);

    for(unsigned short i = 0; i < prevLayerSize; i++) {
      sum += prevLayer[i] * this->getSynapse(prevLayerIndex, i, neuron);
    }

    return this->hyperbolicTangent(sum);
  }

  float Network::fTanh(float x) {
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    float r = a / b;
    return r > 1.0f ? 1.0f : r < -1.0f ? -1.0f : r;
  }

  float Network::hyperbolicTangent(float sum) {
    return this->fTanh(sum);
  }

  unsigned short* Network::getStructure() {
    return this->structure;
  }

  unsigned short Network::getDepth() {
    return this->depth;
  }

  unsigned short Network::getOutputSize() {
    return this->structure[this->depth - 1];
  }

  float Network::getNeuronValue(unsigned short layer, unsigned short neuron) {
    return this->valueMatrice[layer][neuron];
  }

  float Network::getSynapse(unsigned short layer, unsigned short output, unsigned short input) {
    short stride = this->structure[layer] + 1;
    return this->weightMatrice[layer][output + stride * input];
  }

  void Network::setSynapse(unsigned short layer, unsigned short output, unsigned short input, float value) {
    short stride = this->structure[layer] + 1;
    this->weightMatrice[layer][output + stride * input] = value;
  }
}
