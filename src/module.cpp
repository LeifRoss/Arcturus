#include <nan.h>
using namespace v8;


void Init(Local<Object> exports, Local<Object> module) {
  //NODE_SET_METHOD(module, "exports", etcetc);
}


NODE_MODULE(addon, Init)
