add_library (
  Cortex
  Network.h Network.cpp
  BackpropagationTrainer.h BackpropagationTrainer.cpp
  Genotype.h Genotype.cpp
)

install (TARGETS Cortex DESTINATION bin)
install (FILES Network.h DESTINATION include)
install (FILES BackpropagationTrainer.h DESTINATION include)
install (FILES Genotype.h DESTINATION include)
