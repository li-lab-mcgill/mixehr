#include <unordered_map>
#include <algorithm>
#include <vector>
#include <list>
#include <random>
#include <chrono>
#include <ctime>
#include <iostream>
#include <string>
#include <assert.h>

using namespace std;

template <class T>
inline void hash_combine(size_t & seed, const T & v)
{
	hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std
{
template<typename S, typename T> struct hash<pair<S, T>>
{
	inline size_t operator()(const pair<S, T> & v) const
	{
		size_t seed = 0;
		::hash_combine(seed, v.first);
		::hash_combine(seed, v.second);
		return seed;
	}
};
}

