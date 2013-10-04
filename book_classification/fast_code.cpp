#include <unordered_map>
#include <cmath>
#include <iostream>

class MyHash
{
public:
    std::size_t operator()(long s) const 
    {
        return 1664525*s + 1013904223;
    }
};

extern "C" void pairwise_entropies_window(
  double *output, long output_size,
  const long *words, long words_size,
  const double *weights) {
	
    auto temp = std::unordered_map<long, double, MyHash>();
    temp.reserve(words_size);
    const long center = words[words_size / 2];

    for (int i = 0; i < words_size; ++i) {
        //std::cerr << words[i] << ' ' << weights[i] << '\n';
        temp[words[i]] += weights[i];
    }

    //return;
    for (auto it = temp.begin(); it != temp.end(); ++it) {
        //std::cerr << it->first << ' ' << it->second << '\n';
        const long index = (center*1664525 + it->first) % output_size;
        output[index] += it->second * std::log(it->second);
    }
}