#pragma once
namespace znn {
namespace phi {

class Layer {
    public:
        virtual void forward(void)=0; //TODO: remove this dummy
        virtual ~Layer() =0;
};

}
}
