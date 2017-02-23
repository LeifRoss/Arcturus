#ifndef _CORTEX_NETWORK_H
#define _CORTEX_NETWORK_H

namespace Cortex {
  class Network {
  public:
    Network(unsigned short *structure, unsigned short depth);
    Network(unsigned short *structure, unsigned short depth, float **valueMatrice, float **weightMatrice);
    ~Network();
    void run(float *input, float *output);
    void identity();
    unsigned short* getStructure();
    float getSynapse(unsigned short layer, unsigned short output, unsigned short input);
    void setSynapse(unsigned short layer, unsigned short output, unsigned short input, float value);
    float getNeuronValue(unsigned short layer, unsigned short neuron);
    unsigned short getDepth();
    unsigned short getOutputSize();
    const float BIAS = 1.0f;
  private:
    unsigned short *structure;
    unsigned short depth;
    float **valueMatrice;
    float **weightMatrice;
    float runNeuron(unsigned short layer, unsigned short neuron);
    float sigmoid(float sum);
    float hyperbolicTangent(float sum);
    float fTanh(float value);
    float inverseSigmoid(float sum);
  };
}

#endif
