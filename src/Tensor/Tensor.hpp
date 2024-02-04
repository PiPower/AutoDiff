#include <vector>

class Tensor
{
public:
    Tensor(std::vector<unsigned int> dim = {});
    std::vector<unsigned int> dim;
};