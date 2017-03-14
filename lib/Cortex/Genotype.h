#ifndef _CORTEX_GENOTYPE_H
#define _CORTEX_GENOTYPE_H

#include "Network.h"

namespace Cortex {

  class Genotype {
  public:
    Genotype(char *sequence);
    ~Genotype();
    int initNetwork(Network *net);
    int interpolate(Genotype *target, float balance);
    int mutate(float rate);
    Genotype* clone();
  private:
    char *geneSequence;
  };
}

#endif
