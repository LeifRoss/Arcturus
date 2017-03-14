#include "Genotype.h"
#include "Network.h"


/**
* Genesequence structure char[] ????? See the google drawings chart for more up to date description
* header (Maintains structure of entire phenotype):
* - structure of super-network
*   - amount of sub-networks
*   - inter-sub-network connections
*
* body (For each individual network):
* - topology
* - activator types
* - synaptic weights
*
*/
namespace Cortex {

  Genotype::Genotype(char *sequence) {
    this->geneSequence = sequence;
  }

  Genotype::~Genotype() {
    delete[] this->geneSequence;
  }

  int Genotype::mutate(float rate) {
    return 0;
  }

  int Genotype::initNetwork(Network *net) {
    return 0;
  }

  int Genotype::interpolate(Genotype *target, float balance) {
    return 0;
  }

  Genotype* Genotype::clone() {
    return 0;
  }

}
