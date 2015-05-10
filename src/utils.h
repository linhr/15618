#ifndef _UTILS_H
#define _UTILS_H

#include <algorithm>
#include <iterator>
#include <iostream>

template <typename T>
void print_vector(const vector<T> &v) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

#endif
